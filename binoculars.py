#!/usr/bin/env python3
"""
Binoculars-style scoring for markdown text using two llama.cpp (GGUF) models,
loaded sequentially to reduce VRAM usage.

Metrics:
  - logPPL(M1): observer log perplexity over the text
  - logXPPL(M1,M2): cross log perplexity (cross-entropy) of M2 under M1's predictive distribution
  - B = logPPL / logXPPL   (Binoculars ratio)

Notes:
  - This implementation requires full per-token logits, so we set logits_all=True.
  - llama-cpp-python allocates a logits buffer of shape (n_ctx, n_vocab) when logits_all=True.
    For memory sanity, this script auto-sets n_ctx = token_count by default (unless you override).
  - Both models should share tokenizer + vocab alignment (ideally same family/base vs instruct sibling).
"""

from __future__ import annotations

import argparse
import bisect
import ctypes
from datetime import datetime
import difflib
import gc
import hashlib
import json
import os
import re
import socket
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, List, Set

import numpy as np
import llama_cpp
from llama_cpp import Llama


# ----------------------------
# Utilities
# ----------------------------

SPELL_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*(?:[-/][A-Za-z]+(?:['’][A-Za-z]+)*)*")
SYNONYM_TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*$")
SYNONYM_OPTION_COUNT = 9
SYNONYM_GRID_COLUMNS = 3
_ENGLISH_WORDLIST_PATHS = [
    "/usr/share/dict/american-english",
    "/usr/share/dict/british-english",
    "/usr/share/dict/words",
    "/usr/dict/words",
]
_ENGLISH_WORDS_CACHE: Optional[Set[str]] = None
_LLAMA_SUPPRESSED_LOG_PATTERNS = (
    re.compile(r"^llama_context:\s*n_batch is less than GGML_KQ_MASK_PAD - increasing to \d+$"),
    re.compile(r"^llama_context:\s*n_ctx_per_seq \(\d+\) > n_ctx_train \(\d+\) -- possible training context overflow$"),
    re.compile(
        r"^llama_context:\s*n_ctx_per_seq \(\d+\) < n_ctx_train \(\d+\) -- the full capacity of the model will not be utilized$"
    ),
)
_LLAMA_LOG_CALLBACK = None
_LLAMA_LOG_CONFIGURED = False
_LLAMA_LAST_LOG_LEVEL = 1
# ggml log levels (from llama.cpp / llama-cpp-python):
# 0=NONE, 1=INFO, 2=WARN, 3=ERROR, 4=DEBUG, 5=CONT(previous level)
_LLAMA_LOG_LEVEL_ERROR = 3
_LOCAL_SYNONYM_FALLBACK: Dict[str, List[str]] = {
    "quick": ["fast", "rapid", "swift", "speedy", "brisk", "hasty"],
    "slow": ["gradual", "leisurely", "sluggish", "unhurried", "plodding", "delayed"],
    "good": ["great", "solid", "decent", "worthy", "favorable", "beneficial"],
    "bad": ["poor", "awful", "weak", "inferior", "unfavorable", "harmful"],
    "big": ["large", "huge", "vast", "major", "sizable", "substantial"],
    "small": ["little", "tiny", "compact", "minor", "modest", "slight"],
    "smart": ["clever", "sharp", "bright", "astute", "savvy", "intelligent"],
    "easy": ["simple", "straightforward", "effortless", "light", "smooth", "clear"],
    "hard": ["difficult", "tough", "challenging", "demanding", "strenuous", "severe"],
    "clear": ["plain", "explicit", "evident", "obvious", "distinct", "transparent"],
    "show": ["display", "reveal", "present", "illustrate", "demonstrate", "exhibit"],
    "use": ["apply", "utilize", "employ", "leverage", "adopt", "operate"],
    "make": ["create", "build", "produce", "form", "craft", "generate"],
    "change": ["alter", "modify", "revise", "adjust", "shift", "transform"],
    "help": ["assist", "support", "aid", "enable", "facilitate", "improve"],
    "idea": ["concept", "notion", "thought", "insight", "proposal", "theme"],
    "important": ["key", "vital", "critical", "central", "major", "essential"],
    "interesting": ["engaging", "intriguing", "compelling", "notable", "fascinating", "captivating"],
}
_WORDNET_MODULE: Optional[Any] = None
_WORDNET_READY: Optional[bool] = None


def should_suppress_llama_context_log(text: str) -> bool:
    msg = (text or "").strip()
    if not msg:
        return False
    return any(pat.match(msg) for pat in _LLAMA_SUPPRESSED_LOG_PATTERNS)


def _is_synonym_word_char(ch: str) -> bool:
    return bool(ch) and (ch.isalpha() or ch in {"'", "’", "-"})


def extract_click_word_span(text: str, char_pos: int) -> Optional[Tuple[int, int, str]]:
    """
    Return (start, end, word) for the word around char_pos.
    Accepts apostrophes and hyphens inside words.
    """
    raw = str(text or "")
    if not raw:
        return None
    n = len(raw)
    pos = max(0, min(n - 1, int(char_pos)))
    if not _is_synonym_word_char(raw[pos]):
        if pos > 0 and _is_synonym_word_char(raw[pos - 1]):
            pos -= 1
        else:
            return None

    start = pos
    while start > 0 and _is_synonym_word_char(raw[start - 1]):
        start -= 1
    end = pos + 1
    while end < n and _is_synonym_word_char(raw[end]):
        end += 1

    while start < end and raw[start] in {"'", "’", "-"}:
        start += 1
    while end > start and raw[end - 1] in {"'", "’", "-"}:
        end -= 1
    if end <= start:
        return None

    word = raw[start:end]
    if not SYNONYM_TOKEN_RE.match(word):
        return None
    return (start, end, word)


def _normalize_synonym_candidate(word: str) -> Optional[str]:
    w = str(word or "").strip().replace("’", "'").lower()
    if not w:
        return None
    w = w.replace("_", " ")
    w = re.sub(r"\s+", " ", w)
    if " " in w:
        return None
    w = w.strip("'-.")
    if not w:
        return None
    if not SYNONYM_TOKEN_RE.match(w):
        return None
    return w


def dedupe_synonym_candidates(
    base_word: str,
    candidates: List[str],
    max_items: int = SYNONYM_OPTION_COUNT,
) -> List[str]:
    base = _normalize_synonym_candidate(base_word)
    out: List[str] = []
    seen: Set[str] = set()
    for cand in candidates:
        norm = _normalize_synonym_candidate(cand)
        if not norm:
            continue
        if base and norm == base:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
        if len(out) >= max(1, int(max_items)):
            break
    return out


def apply_word_case_from_template(template: str, candidate: str) -> str:
    src = str(template or "")
    dst = str(candidate or "")
    if not dst:
        return dst
    if src.isupper():
        return dst.upper()
    if len(src) > 1 and src[0].isupper() and src[1:].islower():
        return dst.capitalize()
    if src[:1].isupper():
        return dst[:1].upper() + dst[1:]
    return dst


def _is_ascii_alpha_word(word: str) -> bool:
    return bool(re.fullmatch(r"[a-z]+", str(word or "")))


def _is_vowel(ch: str) -> bool:
    return ch in {"a", "e", "i", "o", "u"}


def _looks_cvc_ending(word: str) -> bool:
    w = str(word or "")
    if len(w) < 3:
        return False
    c1, c2, c3 = w[-3], w[-2], w[-1]
    if not (c1.isalpha() and c2.isalpha() and c3.isalpha()):
        return False
    if _is_vowel(c1) or (not _is_vowel(c2)) or _is_vowel(c3):
        return False
    if c3 in {"w", "x", "y"}:
        return False
    # Keep doubling conservative to avoid odd forms.
    return len(w) <= 5


def detect_inflection_pattern(word: str) -> str:
    w = _normalize_synonym_candidate(word)
    if not w:
        return "base"
    if len(w) >= 5 and w.endswith("ing"):
        return "ing"
    if len(w) >= 4 and w.endswith("ied"):
        return "ied"
    if len(w) >= 4 and w.endswith("ed"):
        return "ed"
    if len(w) >= 4 and w.endswith("ies"):
        return "ies"
    if len(w) >= 5 and w.endswith("est"):
        return "est"
    if len(w) >= 4 and w.endswith("er"):
        return "er"
    if len(w) >= 4 and w.endswith("ly"):
        return "ly"
    if len(w) >= 3 and w.endswith("es"):
        return "es"
    if len(w) >= 3 and w.endswith("s") and not w.endswith("ss"):
        return "s"
    return "base"


def inflect_candidate_to_pattern(base_candidate: str, pattern: str) -> str:
    w = _normalize_synonym_candidate(base_candidate)
    if not w:
        return ""
    if pattern == "base":
        return w
    if not _is_ascii_alpha_word(w):
        return w
    if pattern == "ing":
        if w.endswith("ing"):
            return w
        if w.endswith("ie"):
            return w[:-2] + "ying"
        if w.endswith("e") and not w.endswith(("ee", "oe", "ye")):
            return w[:-1] + "ing"
        if _looks_cvc_ending(w):
            return w + w[-1] + "ing"
        return w + "ing"
    if pattern in {"ed", "ied"}:
        if w.endswith("ed"):
            return w
        if w.endswith("e"):
            return w + "d"
        if w.endswith("y") and len(w) >= 2 and not _is_vowel(w[-2]):
            return w[:-1] + "ied"
        if _looks_cvc_ending(w):
            return w + w[-1] + "ed"
        return w + "ed"
    if pattern in {"s", "es", "ies"}:
        if w.endswith("s"):
            return w
        if w.endswith("y") and len(w) >= 2 and not _is_vowel(w[-2]):
            return w[:-1] + "ies"
        if w.endswith(("s", "x", "z", "sh", "ch", "o")):
            return w + "es"
        return w + "s"
    if pattern == "ly":
        if w.endswith("ly"):
            return w
        if w.endswith("y") and len(w) >= 2 and not _is_vowel(w[-2]):
            return w[:-1] + "ily"
        if w.endswith("ic"):
            return w + "ally"
        return w + "ly"
    if pattern == "er":
        if w.endswith("er"):
            return w
        if w.endswith("y") and len(w) >= 2 and not _is_vowel(w[-2]):
            return w[:-1] + "ier"
        if w.endswith("e"):
            return w + "r"
        if _looks_cvc_ending(w):
            return w + w[-1] + "er"
        return w + "er"
    if pattern == "est":
        if w.endswith("est"):
            return w
        if w.endswith("y") and len(w) >= 2 and not _is_vowel(w[-2]):
            return w[:-1] + "iest"
        if w.endswith("e"):
            return w + "st"
        if _looks_cvc_ending(w):
            return w + w[-1] + "est"
        return w + "est"
    return w


def _get_wordnet_module() -> Optional[Any]:
    global _WORDNET_MODULE, _WORDNET_READY
    if _WORDNET_READY is False:
        return None
    if _WORDNET_READY is None:
        try:
            from nltk.corpus import wordnet as wn  # type: ignore
        except Exception:
            _WORDNET_READY = False
            _WORDNET_MODULE = None
            return None
        _WORDNET_MODULE = wn
        _WORDNET_READY = True
    return _WORDNET_MODULE


def _lookup_synonyms_wordnet(word: str) -> List[str]:
    wn = _get_wordnet_module()
    if wn is None:
        return []
    out: List[str] = []
    try:
        for syn in wn.synsets(word):
            lemmas = syn.lemma_names()
            for lemma in lemmas:
                out.append(str(lemma))
                if len(out) >= 80:
                    return out
    except Exception:
        return []
    return out


def _lookup_synonyms_datamuse(word: str, timeout_s: float = 1.2) -> List[str]:
    encoded = urllib.parse.quote_plus(str(word or "").strip())
    if not encoded:
        return []
    url = f"https://api.datamuse.com/words?ml={encoded}&max={max(24, SYNONYM_OPTION_COUNT * 4)}"
    req = urllib.request.Request(
        url=url,
        headers={"User-Agent": "binoculars-synonyms/1.0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=max(0.5, float(timeout_s))) as resp:
            payload = resp.read()
    except Exception:
        return []
    try:
        parsed = json.loads(payload.decode("utf-8", errors="replace"))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    out: List[str] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        w = item.get("word")
        if isinstance(w, str):
            out.append(w)
    return out


def lookup_synonyms_fast(word: str, max_items: int = SYNONYM_OPTION_COUNT) -> List[str]:
    base = _normalize_synonym_candidate(word)
    if not base:
        return []
    candidates: List[str] = []
    candidates.extend(_LOCAL_SYNONYM_FALLBACK.get(base, []))
    candidates.extend(_lookup_synonyms_wordnet(base))
    out = dedupe_synonym_candidates(base, candidates, max_items=max_items)
    if len(out) >= max_items:
        return out[:max_items]
    remote = _lookup_synonyms_datamuse(base)
    out = dedupe_synonym_candidates(base, out + remote, max_items=max_items)
    return out[:max_items]


def _emit_llama_log_text(text: str) -> None:
    if not text:
        return
    if should_suppress_llama_context_log(text):
        return
    try:
        sys.stderr.write(text)
        sys.stderr.flush()
    except Exception:
        pass


def _llama_log_callback(level: int, text: bytes, user_data: ctypes.c_void_p) -> None:
    del user_data
    if not text:
        return
    global _LLAMA_LAST_LOG_LEVEL
    effective_level = _LLAMA_LAST_LOG_LEVEL if level == 5 else level
    if level != 5:
        _LLAMA_LAST_LOG_LEVEL = level
    if effective_level < _LLAMA_LOG_LEVEL_ERROR:
        # Keep stderr clean: drop INFO/WARN/DEBUG/CONT noise.
        return
    chunk = text.decode("utf-8", errors="replace")
    if not chunk:
        return
    _emit_llama_log_text(chunk)


def configure_llama_log_filtering() -> None:
    global _LLAMA_LOG_CALLBACK, _LLAMA_LOG_CONFIGURED
    if _LLAMA_LOG_CONFIGURED:
        return
    _LLAMA_LOG_CONFIGURED = True

    env_raw = str(os.environ.get("BINOCULARS_SUPPRESS_LLAMA_CONTEXT_WARNINGS", "1")).strip().lower()
    if env_raw in {"0", "false", "no", "off"}:
        return

    log_set = getattr(llama_cpp, "llama_log_set", None)
    cb_factory = getattr(llama_cpp, "llama_log_callback", None)
    if log_set is None or cb_factory is None:
        return

    try:
        _LLAMA_LOG_CALLBACK = cb_factory(_llama_log_callback)
        log_set(_LLAMA_LOG_CALLBACK, None)
    except Exception:
        _LLAMA_LOG_CALLBACK = None


def load_english_words() -> Set[str]:
    global _ENGLISH_WORDS_CACHE
    if _ENGLISH_WORDS_CACHE is not None:
        return _ENGLISH_WORDS_CACHE

    words: Set[str] = set()
    for path in _ENGLISH_WORDLIST_PATHS:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for raw in f:
                    w = raw.strip()
                    if not w or w.startswith("#"):
                        continue
                    words.add(w.lower())
        except Exception:
            continue

    # Minimal fallback so spellcheck remains functional even when system dictionaries are absent.
    if not words:
        words = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "he",
            "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "were", "with", "you",
            "i", "we", "they", "this", "not", "but", "if", "then", "their", "there", "his", "her",
        }

    _ENGLISH_WORDS_CACHE = words
    return _ENGLISH_WORDS_CACHE


def _word_candidates(word: str) -> Set[str]:
    w = word.replace("’", "'").lower()
    out: Set[str] = {w}
    if w.endswith("'s") and len(w) > 2:
        out.add(w[:-2])
    if w.endswith("s'") and len(w) > 2:
        out.add(w[:-1])
    if w.endswith("in'") and len(w) > 3:
        out.add(w[:-1] + "g")
    return {c for c in out if c}


def is_word_spelled_correctly(word: str, dictionary: Set[str]) -> bool:
    if not word:
        return True
    if any(ch.isdigit() for ch in word):
        return True

    normalized = word.replace("’", "'")
    if normalized.isupper() and len(normalized) <= 6:
        return True
    if re.search(r"[a-z][A-Z]|[A-Z][a-z].*[A-Z]", normalized):
        # Likely identifier/acronym-like token.
        return True

    parts = re.split(r"[-/]", normalized)
    for part in parts:
        if not part:
            continue
        if len(part) <= 1:
            continue
        candidates = _word_candidates(part)
        if not any(c in dictionary for c in candidates):
            return False
    return True


def find_misspelled_spans(text: str, dictionary: Set[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    if not dictionary:
        return spans
    for m in SPELL_WORD_RE.finditer(text):
        tok = m.group(0)
        if not is_word_spelled_correctly(tok, dictionary):
            spans.append((m.start(), m.end()))
    return spans


def changed_span_in_new_text(old_text: str, new_text: str) -> Tuple[int, int]:
    """
    Return the changed span [start, end) within new_text between two versions.
    If there is no insertion/replacement content in new_text (for example pure deletion),
    end can equal start.
    """
    if old_text == new_text:
        return (0, 0)

    old_len = len(old_text)
    new_len = len(new_text)

    start = 0
    common_prefix = min(old_len, new_len)
    while start < common_prefix and old_text[start] == new_text[start]:
        start += 1

    old_end = old_len
    new_end = new_len
    while old_end > start and new_end > start and old_text[old_end - 1] == new_text[new_end - 1]:
        old_end -= 1
        new_end -= 1

    return (start, new_end)


def logsumexp_1d(x: np.ndarray) -> float:
    """Stable logsumexp for 1D array."""
    m = float(np.max(x))
    if not np.isfinite(m):
        m = 0.0
    s = float(np.sum(np.exp(x - m)))
    return m + float(np.log(s)) if s > 0.0 else m


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_text(path: Optional[str]) -> str:
    if path is None or path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def json_dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def filter_llama_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only kwargs that are accepted by Llama.__init__ (common subset).
    Unknown keys in config won't break the script.
    """
    allowed = {
        # Model params
        "model_path", "n_gpu_layers", "split_mode", "main_gpu", "tensor_split",
        "vocab_only", "use_mmap", "use_mlock", "kv_overrides",
        # Context params
        "seed", "n_ctx", "n_batch", "n_ubatch", "n_threads", "n_threads_batch",
        "rope_scaling_type", "pooling_type", "rope_freq_base", "rope_freq_scale",
        "yarn_ext_factor", "yarn_attn_factor", "yarn_beta_fast", "yarn_beta_slow", "yarn_orig_ctx",
        "logits_all", "embedding", "offload_kqv", "flash_attn", "op_offload", "swa_full",
        # Sampling params
        "no_perf", "last_n_tokens_size",
        # LoRA
        "lora_base", "lora_scale", "lora_path",
        # Backend
        "numa",
        # Chat format
        "chat_format", "chat_handler",
        # Speculative
        "draft_model",
        # Tokenizer override
        "tokenizer",
        # KV cache quantization
        "type_k", "type_v",
        # Misc
        "spm_infill", "verbose",
    }
    return {k: v for k, v in cfg.items() if k in allowed}


@dataclass
class TextConfig:
    add_bos: bool = True
    special_tokens: bool = False
    # If set, allow truncation to this many tokens (0 means no truncation).
    max_tokens: int = 0


@dataclass
class CacheConfig:
    dir: str = ""
    dtype: str = "float32"     # float32 or float16
    keep: bool = False         # keep cache files for debugging


@dataclass
class RewriteLLMConfig:
    endpoint_url: str
    model: str
    api_key: str = ""
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer "
    request_path: str = "/chat/completions"
    timeout_s: float = 12.0
    max_tokens: int = 180
    temperature: float = 0.78
    top_p: float = 0.95
    context_chars_each_side: int = 1800
    context_paragraphs_each_side: int = 2
    context_window_max_chars: int = 5200
    extra_headers: Dict[str, str] = field(default_factory=dict)
    extra_body: Dict[str, Any] = field(default_factory=dict)


def default_master_config_path() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "config.binoculars.json")


def default_rewrite_llm_config_path() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "config.binoculars.llm.json")


def _as_string_map(obj: Any) -> Dict[str, str]:
    if not isinstance(obj, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in obj.items():
        key = str(k).strip()
        if not key:
            continue
        out[key] = str(v)
    return out


def _as_object_map(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {}
    return dict(obj)


def _build_chat_completions_url(endpoint_url: str, request_path: str) -> str:
    base = str(endpoint_url).strip()
    if not base:
        return ""
    low = base.lower()
    if low.endswith("/chat/completions"):
        return base
    req = str(request_path or "").strip() or "/chat/completions"
    if not req.startswith("/"):
        req = "/" + req
    return base.rstrip("/") + req


def load_optional_rewrite_llm_config(
    path: Optional[str] = None,
) -> Tuple[Optional[RewriteLLMConfig], Optional[str]]:
    cfg_path = str(path or default_rewrite_llm_config_path()).strip()
    if not cfg_path:
        return None, None
    if not os.path.isfile(cfg_path):
        return None, None

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        return None, f"LLM config parse error ({cfg_path}): {exc}"

    if not isinstance(raw, dict):
        return None, f"LLM config must be a JSON object ({cfg_path})."

    cfg_raw = raw.get("llm") if isinstance(raw.get("llm"), dict) else raw
    if not isinstance(cfg_raw, dict):
        return None, f"LLM config 'llm' section must be an object ({cfg_path})."

    if not bool(cfg_raw.get("enabled", True)):
        return None, None

    endpoint_url = str(cfg_raw.get("endpoint_url", cfg_raw.get("endpoint", cfg_raw.get("url", "")))).strip()
    model = str(cfg_raw.get("model", "")).strip()
    if not endpoint_url or not model:
        return None, f"LLM config missing endpoint_url or model ({cfg_path})."

    api_key = str(cfg_raw.get("api_key", "")).strip()
    api_key_env = str(cfg_raw.get("api_key_env", "")).strip()
    if not api_key and api_key_env:
        api_key = str(os.environ.get(api_key_env, "")).strip()
    if not api_key and bool(cfg_raw.get("allow_openai_api_key_env", True)):
        api_key = str(os.environ.get("OPENAI_API_KEY", "")).strip()

    try:
        timeout_s = float(cfg_raw.get("timeout_s", 12.0))
    except Exception:
        timeout_s = 12.0
    if timeout_s <= 0:
        timeout_s = 12.0

    try:
        max_tokens = int(cfg_raw.get("max_tokens", 180))
    except Exception:
        max_tokens = 180
    max_tokens = max(24, min(max_tokens, 1200))

    try:
        temperature = float(cfg_raw.get("temperature", 0.78))
    except Exception:
        temperature = 0.78
    temperature = max(0.0, min(temperature, 2.0))

    try:
        top_p = float(cfg_raw.get("top_p", 0.95))
    except Exception:
        top_p = 0.95
    top_p = max(0.01, min(top_p, 1.0))

    try:
        context_chars_each_side = int(cfg_raw.get("context_chars_each_side", 1800))
    except Exception:
        context_chars_each_side = 1800
    context_chars_each_side = max(300, min(context_chars_each_side, 20000))

    try:
        context_paragraphs_each_side = int(cfg_raw.get("context_paragraphs_each_side", 2))
    except Exception:
        context_paragraphs_each_side = 2
    context_paragraphs_each_side = max(0, min(context_paragraphs_each_side, 8))

    try:
        context_window_max_chars = int(cfg_raw.get("context_window_max_chars", 5200))
    except Exception:
        context_window_max_chars = 5200
    context_window_max_chars = max(900, min(context_window_max_chars, 50000))

    request_path = str(cfg_raw.get("request_path", "/chat/completions")).strip() or "/chat/completions"
    api_key_header = str(cfg_raw.get("api_key_header", "Authorization")).strip() or "Authorization"
    api_key_prefix = str(cfg_raw.get("api_key_prefix", "Bearer ")).strip()
    if api_key_prefix and not api_key_prefix.endswith(" "):
        api_key_prefix += " "

    extra_headers = _as_string_map(cfg_raw.get("extra_headers", cfg_raw.get("headers", {})))
    extra_body = _as_object_map(cfg_raw.get("extra_body", cfg_raw.get("body", cfg_raw.get("params", {}))))

    cfg = RewriteLLMConfig(
        endpoint_url=endpoint_url,
        model=model,
        api_key=api_key,
        api_key_header=api_key_header,
        api_key_prefix=api_key_prefix,
        request_path=request_path,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        context_chars_each_side=context_chars_each_side,
        context_paragraphs_each_side=context_paragraphs_each_side,
        context_window_max_chars=context_window_max_chars,
        extra_headers=extra_headers,
        extra_body=extra_body,
    )
    return cfg, None


def _post_json_request(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout_s: float,
) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=max(1.0, float(timeout_s))) as resp:
        body = resp.read()
        text = body.decode("utf-8", errors="replace")
        parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response was not a JSON object.")
    return parsed


def _generate_rewrites_via_openai_compatible(
    llm_cfg: RewriteLLMConfig,
    system_prompt: str,
    user_prompt: str,
    option_count: int,
) -> List[str]:
    n_opts = max(1, int(option_count))
    url = _build_chat_completions_url(llm_cfg.endpoint_url, llm_cfg.request_path)
    if not url:
        raise ValueError("LLM endpoint URL is empty.")

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    for k, v in llm_cfg.extra_headers.items():
        headers[str(k)] = str(v)
    if llm_cfg.api_key:
        headers[llm_cfg.api_key_header] = f"{llm_cfg.api_key_prefix}{llm_cfg.api_key}"

    rewrites: List[str] = []
    seen: Set[str] = set()
    temperatures = [llm_cfg.temperature, min(1.25, llm_cfg.temperature + 0.20)]

    for temp in temperatures:
        if len(rewrites) >= n_opts:
            break
        want = max(1, n_opts - len(rewrites))
        payload: Dict[str, Any] = dict(llm_cfg.extra_body)
        payload.setdefault("model", llm_cfg.model)
        payload["messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload.setdefault("temperature", float(temp))
        payload.setdefault("top_p", float(llm_cfg.top_p))
        payload.setdefault("max_tokens", int(llm_cfg.max_tokens))
        payload.setdefault("n", int(want))
        payload.setdefault("stream", False)

        response = _post_json_request(url, payload, headers, llm_cfg.timeout_s)
        choices = response.get("choices")
        if not isinstance(choices, list):
            continue
        for ch in choices:
            txt = _extract_completion_text({"choices": [ch]})
            cand = _normalize_generated_rewrite(txt)
            if not cand:
                continue
            key = _semantic_text_key(cand)
            if not key or key in seen:
                continue
            seen.add(key)
            rewrites.append(cand)
            if len(rewrites) >= n_opts:
                break

    if not rewrites:
        raise ValueError("No rewrite candidates returned by external LLM.")
    while len(rewrites) < n_opts:
        rewrites.append(rewrites[-1])
    return rewrites[:n_opts]


def load_master_config_detailed(path: str) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Master config must be a JSON object.")

    default_label = str(cfg.get("default", "")).strip()
    profiles_raw = cfg.get("profiles")
    if not isinstance(profiles_raw, dict) or not profiles_raw:
        raise ValueError("Master config must contain a non-empty 'profiles' object.")

    profiles: Dict[str, Dict[str, Any]] = {}
    for label, entry_raw in profiles_raw.items():
        key = str(label).strip()
        if not key:
            continue

        cfg_path = ""
        max_tokens_override: Optional[int] = None

        if isinstance(entry_raw, str):
            cfg_path = entry_raw.strip()
        elif isinstance(entry_raw, dict):
            cfg_path = str(entry_raw.get("path", "")).strip()
            if "max_tokens" in entry_raw and entry_raw.get("max_tokens") is not None:
                try:
                    max_tokens_override = int(entry_raw.get("max_tokens"))
                except Exception as exc:
                    raise ValueError(
                        f"Master config profile '{key}' has invalid max_tokens={entry_raw.get('max_tokens')!r}."
                    ) from exc
                if max_tokens_override < 0:
                    raise ValueError(
                        f"Master config profile '{key}' has max_tokens={max_tokens_override}; expected >= 0."
                    )
        else:
            continue

        if not cfg_path:
            continue

        entry: Dict[str, Any] = {"path": cfg_path}
        if max_tokens_override is not None:
            entry["max_tokens"] = max_tokens_override
        profiles[key] = entry

    if not profiles:
        raise ValueError("Master config 'profiles' has no valid entries.")

    if not default_label:
        raise ValueError("Master config must define a non-empty 'default' profile label.")
    if default_label not in profiles:
        raise ValueError(
            f"Master config default='{default_label}' is not present in profiles: "
            f"{', '.join(sorted(profiles.keys()))}"
        )

    return default_label, profiles


def load_master_config(path: str) -> Tuple[str, Dict[str, str]]:
    default_label, profiles_detailed = load_master_config_detailed(path)
    profiles: Dict[str, str] = {label: str(entry["path"]) for label, entry in profiles_detailed.items()}
    return default_label, profiles


def resolve_profile_config_path(master_cfg_path: str, profile_label: Optional[str]) -> Tuple[str, str]:
    label, cfg_path, _max_tokens_override = resolve_profile_config(master_cfg_path, profile_label)
    return label, cfg_path


def resolve_profile_config(
    master_cfg_path: str,
    profile_label: Optional[str],
) -> Tuple[str, str, Optional[int]]:
    if not os.path.isfile(master_cfg_path):
        raise ValueError(f"Master config file not found: {master_cfg_path}")

    default_label, profiles = load_master_config_detailed(master_cfg_path)
    selected_label = (profile_label or "").strip() or default_label
    if selected_label not in profiles:
        raise ValueError(
            f"Unknown --config profile '{selected_label}'. Available profiles: {', '.join(sorted(profiles.keys()))}"
        )

    selected = profiles[selected_label]
    cfg_path = str(selected["path"])
    if not os.path.isfile(cfg_path):
        raise ValueError(f"Config profile '{selected_label}' points to missing file: {cfg_path}")

    max_tokens_override_raw = selected.get("max_tokens")
    max_tokens_override: Optional[int]
    if max_tokens_override_raw is None:
        max_tokens_override = None
    else:
        max_tokens_override = int(max_tokens_override_raw)
        if max_tokens_override < 0:
            raise ValueError(
                f"Config profile '{selected_label}' has max_tokens={max_tokens_override}; expected >= 0."
            )

    return selected_label, cfg_path, max_tokens_override


def load_config(path: str) -> Tuple[Dict[str, Any], TextConfig, CacheConfig]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "observer" not in cfg or "performer" not in cfg:
        raise ValueError("Config must contain top-level keys: 'observer' and 'performer'.")

    tcfg_raw = cfg.get("text", {}) or {}
    ccfg_raw = cfg.get("cache", {}) or {}

    tcfg = TextConfig(
        add_bos=bool(tcfg_raw.get("add_bos", True)),
        special_tokens=bool(tcfg_raw.get("special_tokens", False)),
        max_tokens=int(tcfg_raw.get("max_tokens", 0)),
    )

    ccfg = CacheConfig(
        dir=str(ccfg_raw.get("dir", "")).strip(),
        dtype=str(ccfg_raw.get("dtype", "float32")).strip().lower(),
        keep=bool(ccfg_raw.get("keep", False)),
    )

    if ccfg.dtype not in ("float32", "float16"):
        raise ValueError("cache.dtype must be 'float32' or 'float16'.")

    return cfg, tcfg, ccfg


def build_llama_instance(model_cfg: Dict[str, Any]) -> Llama:
    configure_llama_log_filtering()
    return Llama(**model_cfg)


def close_llama(model: Optional[Llama]) -> None:
    if model is None:
        return
    try:
        model.close()  # llama-cpp-python provides close() to free memory
    except Exception:
        pass


def tokenize_with_vocab_only(model_path: str, text_bytes: bytes, tcfg: TextConfig) -> List[int]:
    """
    Load only vocabulary/tokenizer to tokenize without heavy VRAM/weights usage.
    """
    tok = None
    try:
        configure_llama_log_filtering()
        tok = Llama(
            model_path=model_path,
            vocab_only=True,
            n_gpu_layers=0,
            n_ctx=32,
            verbose=False,
        )
        return tok.tokenize(text_bytes, add_bos=tcfg.add_bos, special=tcfg.special_tokens)
    finally:
        close_llama(tok)
        del tok
        gc.collect()


def maybe_truncate_tokens(tokens: List[int], max_tokens: int) -> List[int]:
    if max_tokens and max_tokens > 0 and len(tokens) > max_tokens:
        # Keep the first max_tokens tokens (simple truncation).
        return tokens[:max_tokens]
    return tokens


def estimate_char_end_for_token_limit(
    text: str,
    model: Llama,
    tcfg: TextConfig,
    token_limit: int,
) -> int:
    """
    Return the largest character offset such that tokenizing text[:offset]
    yields at most token_limit tokens (under current tokenizer settings).
    """
    if token_limit <= 0:
        return len(text)
    if not text:
        return 0

    lo = 0
    hi = len(text)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        prefix_tokens = model.tokenize(
            text[:mid].encode("utf-8", errors="replace"),
            add_bos=tcfg.add_bos,
            special=tcfg.special_tokens,
        )
        n = len(prefix_tokens)
        if n <= token_limit:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def infer_n_ctx(model_section: Dict[str, Any], needed: int) -> int:
    """
    Determine n_ctx:
      - if config specifies a positive int n_ctx: use it (must be >= needed)
      - if missing or 0: auto (use needed)
    """
    n_ctx = int(model_section.get("n_ctx", 0) or 0)
    if n_ctx <= 0:
        return needed
    if n_ctx < needed:
        raise ValueError(f"Configured n_ctx={n_ctx} is smaller than required tokens={needed}. "
                         f"Increase n_ctx or set n_ctx=0 for auto.")
    return n_ctx


def compute_transition_losses(scores: np.ndarray, tokens: List[int]) -> np.ndarray:
    """
    Compute per-transition negative log-likelihoods:
      loss[i] = -log p(token[i+1] | prefix up to i)
    """
    if len(tokens) < 2:
        raise ValueError("Need at least 2 tokens to compute perplexity (a transition).")

    n = len(tokens)
    losses = np.empty(n - 1, dtype=np.float64)

    # scores[i] predicts token[i+1]
    for i in range(n - 1):
        logits = scores[i].astype(np.float64, copy=False)
        logZ = logsumexp_1d(logits)
        nxt = tokens[i + 1]
        losses[i] = logZ - float(logits[nxt])

    return losses


def compute_logppl_from_scores(scores: np.ndarray, tokens: List[int]) -> Tuple[float, float]:
    """
    Compute observer logPPL:
      logPPL = - mean_{i=0..N-2} log p(token[i+1] | prefix up to i)
    Returns: (logPPL, ppl)
    """
    losses = compute_transition_losses(scores, tokens)
    logppl = float(np.mean(losses))
    ppl = float(np.exp(logppl))
    return logppl, ppl


def save_logits_memmap(path: str, arr: np.ndarray, dtype: np.dtype) -> None:
    """
    Save logits array to a disk-backed memmap (raw .dat file).
    """
    mm = np.memmap(path, dtype=dtype, mode="w+", shape=arr.shape)
    # copy in a memory-conscious way
    np.copyto(mm, arr.astype(dtype, copy=False))
    mm.flush()
    del mm


def load_logits_memmap(path: str, shape: Tuple[int, int], dtype: np.dtype) -> np.memmap:
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def compute_cross_logxppl(
    logits1_mm: np.memmap,
    scores2: np.ndarray,
    tokens: List[int],
) -> Tuple[float, float]:
    """
    Compute cross logXPPL (Binoculars cross-perplexity term):

      E_{p1}[log p2] at each position i:
        p1 = softmax(l1)
        log p2 = l2 - logZ2
        E = dot(p1, l2) - logZ2

      logXPPL = - mean_i E

    Returns: (logXPPL, XPPL)
    """
    if len(tokens) < 2:
        raise ValueError("Need at least 2 tokens to compute cross-perplexity.")

    n = len(tokens)
    total_expected_logp2 = 0.0
    count = 0

    for i in range(n - 1):
        # NumPy 2 can reject np.array(..., copy=False) for memmap rows.
        # asarray preserves the "no unnecessary copy" behavior while allowing copies when required.
        l1 = np.asarray(logits1_mm[i], dtype=np.float64)
        l2 = scores2[i].astype(np.float64, copy=False)

        logZ1 = logsumexp_1d(l1)
        p1 = np.exp(l1 - logZ1)  # softmax(l1)

        logZ2 = logsumexp_1d(l2)
        expected_logp2 = float(np.dot(p1, l2) - logZ2)

        total_expected_logp2 += expected_logp2
        count += 1

    logxppl = -total_expected_logp2 / max(count, 1)
    xppl = float(np.exp(logxppl))
    return logxppl, xppl


def split_markdown_paragraph_spans(text: str) -> List[Tuple[int, int]]:
    """
    Return non-empty paragraph-like spans split on blank-line boundaries.
    Spans are character-indexed [start, end).
    """
    spans: List[Tuple[int, int]] = []
    start = 0
    for m in re.finditer(r"\n\s*\n+", text):
        chunk = text[start:m.start()]
        if chunk.strip():
            lead = len(chunk) - len(chunk.lstrip())
            trail = len(chunk.rstrip())
            spans.append((start + lead, start + trail))
        start = m.end()

    tail = text[start:]
    if tail.strip():
        lead = len(tail) - len(tail.lstrip())
        trail = len(tail.rstrip())
        spans.append((start + lead, start + trail))

    return spans


def build_excerpt(text: str, start: int, end: int, max_chars: int = 140) -> str:
    excerpt = " ".join(text[start:end].split())
    if len(excerpt) <= max_chars:
        return excerpt
    return excerpt[: max_chars - 3] + "..."


def analyze_low_perplexity_paragraphs(
    text: str,
    tcfg: TextConfig,
    obs_model: Llama,
    tokens: List[int],
    transition_losses: np.ndarray,
    top_k: int,
) -> Dict[str, Any]:
    """
    Identify paragraph spans that disproportionately lower document-level logPPL.
    Ranking metric:
      delta_doc_logPPL_if_removed = logPPL_without_span - logPPL_full
    Positive values indicate the span is easier than average and lowers total perplexity.
    """
    spans = split_markdown_paragraph_spans(text)
    if not spans:
        return {
            "unit": "paragraph",
            "total_paragraphs": 0,
            "paragraphs_with_transitions": 0,
            "top_low_perplexity_hotspots": [],
        }

    needed_ctx = len(tokens)
    total_transitions = len(tokens) - 1
    total_loss = float(np.sum(transition_losses))
    doc_logppl = float(total_loss / max(total_transitions, 1))

    # Precompute token lengths for all span boundaries.
    boundaries = sorted({0, *[s for s, _ in spans], *[e for _, e in spans]})
    prefix_token_lens: Dict[int, int] = {}
    for pos in boundaries:
        p = int(max(0, min(len(text), pos)))
        prefix_tokens = obs_model.tokenize(
            text[:p].encode("utf-8", errors="replace"),
            add_bos=tcfg.add_bos,
            special=tcfg.special_tokens,
        )
        prefix_token_lens[p] = len(prefix_tokens)

    rows: List[Dict[str, Any]] = []
    for para_id, (char_start, char_end) in enumerate(spans, start=1):
        tok_start = min(prefix_token_lens[char_start], needed_ctx)
        tok_end = min(prefix_token_lens[char_end], needed_ctx)

        # transition i predicts token i+1, so target token range is [tok_start, tok_end)
        t_start = max(1, tok_start)
        t_end = min(tok_end, needed_ctx)
        if t_end <= t_start:
            continue

        span_losses = transition_losses[t_start - 1 : t_end - 1]
        c = int(span_losses.shape[0])
        if c <= 0:
            continue

        span_loss_sum = float(np.sum(span_losses))
        span_logppl = float(span_loss_sum / c)

        if total_transitions > c:
            without_logppl = float((total_loss - span_loss_sum) / (total_transitions - c))
            delta_if_removed = float(without_logppl - doc_logppl)
        else:
            without_logppl = doc_logppl
            delta_if_removed = 0.0

        rows.append(
            {
                "paragraph_id": para_id,
                "char_start": char_start,
                "char_end": char_end,
                "char_len": char_end - char_start,
                "token_start": tok_start,
                "token_end": tok_end,
                "transitions": c,
                "logPPL": span_logppl,
                "delta_vs_doc_logPPL": float(span_logppl - doc_logppl),
                "doc_logPPL_if_removed": without_logppl,
                "delta_doc_logPPL_if_removed": delta_if_removed,
                "excerpt": build_excerpt(text, char_start, char_end),
            }
        )

    rows.sort(key=lambda r: r["delta_doc_logPPL_if_removed"], reverse=True)
    k = max(1, int(top_k))
    top_rows = rows[:k]

    return {
        "unit": "paragraph",
        "total_paragraphs": len(spans),
        "paragraphs_with_transitions": len(rows),
        "doc_logPPL": doc_logppl,
        "ranking_metric": "delta_doc_logPPL_if_removed",
        "top_low_perplexity_hotspots": top_rows,
    }


def compute_paragraph_profile(
    text: str,
    tcfg: TextConfig,
    obs_model: Llama,
    tokens: List[int],
    transition_losses: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute paragraph-level observer perplexity profile over the analyzed token window.
    """
    spans = split_markdown_paragraph_spans(text)
    if not spans:
        return {
            "unit": "paragraph",
            "total_paragraphs": 0,
            "paragraphs_with_transitions": 0,
            "doc_logPPL": 0.0,
            "rows": [],
        }

    needed_ctx = len(tokens)
    total_transitions = len(tokens) - 1
    total_loss = float(np.sum(transition_losses))
    doc_logppl = float(total_loss / max(total_transitions, 1))

    boundaries = sorted({0, *[s for s, _ in spans], *[e for _, e in spans]})
    prefix_token_lens: Dict[int, int] = {}
    for pos in boundaries:
        p = int(max(0, min(len(text), pos)))
        prefix_tokens = obs_model.tokenize(
            text[:p].encode("utf-8", errors="replace"),
            add_bos=tcfg.add_bos,
            special=tcfg.special_tokens,
        )
        prefix_token_lens[p] = len(prefix_tokens)

    rows: List[Dict[str, Any]] = []
    for para_id, (char_start, char_end) in enumerate(spans, start=1):
        tok_start = min(prefix_token_lens[char_start], needed_ctx)
        tok_end = min(prefix_token_lens[char_end], needed_ctx)

        # transition i predicts token i+1, so target token range is [tok_start, tok_end)
        t_start = max(1, tok_start)
        t_end = min(tok_end, needed_ctx)
        if t_end <= t_start:
            continue

        span_losses = transition_losses[t_start - 1 : t_end - 1]
        c = int(span_losses.shape[0])
        if c <= 0:
            continue

        span_loss_sum = float(np.sum(span_losses))
        span_logppl = float(span_loss_sum / c)

        if total_transitions > c:
            without_logppl = float((total_loss - span_loss_sum) / (total_transitions - c))
            delta_if_removed = float(without_logppl - doc_logppl)
        else:
            without_logppl = doc_logppl
            delta_if_removed = 0.0

        rows.append(
            {
                "paragraph_id": para_id,
                "char_start": char_start,
                "char_end": char_end,
                "char_len": char_end - char_start,
                "token_start": tok_start,
                "token_end": tok_end,
                "transitions": c,
                "logPPL": span_logppl,
                "delta_vs_doc_logPPL": float(span_logppl - doc_logppl),
                "doc_logPPL_if_removed": without_logppl,
                "delta_doc_logPPL_if_removed": delta_if_removed,
                "excerpt": build_excerpt(text, char_start, char_end),
            }
        )

    return {
        "unit": "paragraph",
        "total_paragraphs": len(spans),
        "paragraphs_with_transitions": len(rows),
        "doc_logPPL": doc_logppl,
        "rows": rows,
    }


def build_heatmap_markdown(
    text: str,
    source_label: str,
    paragraph_profile: Dict[str, Any],
    top_k: int,
    observer_logppl: float,
    observer_ppl: float,
    performer_logppl: float,
    performer_ppl: float,
    logxppl: float,
    xppl: float,
    binoculars_score: float,
) -> str:
    return build_heatmap_output_markdown(
        text=text,
        source_label=source_label,
        paragraph_profile=paragraph_profile,
        top_k=top_k,
        observer_logppl=observer_logppl,
        observer_ppl=observer_ppl,
        performer_logppl=performer_logppl,
        performer_ppl=performer_ppl,
        logxppl=logxppl,
        xppl=xppl,
        binoculars_score=binoculars_score,
    )


def _prepare_heatmap_annotations(
    rows: List[Dict[str, Any]],
    top_k: int,
    observer_logppl: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    rows = list(rows)
    k = max(1, int(top_k))
    low_rows = sorted(rows, key=lambda r: r["logPPL"])[:k]
    low_ids = {r["paragraph_id"] for r in low_rows}
    high_rows = [r for r in sorted(rows, key=lambda r: r["logPPL"], reverse=True) if r["paragraph_id"] not in low_ids][:k]

    selected: Dict[int, Dict[str, Any]] = {}
    for row in low_rows:
        info = dict(row)
        info["label"] = "LOW"
        info["color"] = "#b71c1c"
        info["css_class"] = "binoculars-heat-low"
        selected[row["paragraph_id"]] = info
    for row in high_rows:
        info = dict(row)
        info["label"] = "HIGH"
        info["color"] = "#0b7a0b"
        info["css_class"] = "binoculars-heat-high"
        selected[row["paragraph_id"]] = info

    # Number notes by document order (start -> end), not by LOW/HIGH bucket.
    annotations: Dict[int, Dict[str, Any]] = {}
    endnotes: List[Dict[str, Any]] = []
    note_index = 1
    for row in sorted(rows, key=lambda r: r["char_start"]):
        info = selected.get(row["paragraph_id"])
        if info is None:
            continue
        note_info = dict(info)
        note_info["note_num"] = note_index
        if observer_logppl != 0.0:
            note_info["pct_contribution"] = (note_info["delta_doc_logPPL_if_removed"] / observer_logppl) * 100.0
        else:
            note_info["pct_contribution"] = float("nan")
        annotations[row["paragraph_id"]] = note_info
        endnotes.append(note_info)
        note_index += 1

    return low_rows, high_rows, annotations, endnotes


def build_heatmap_output_markdown(
    text: str,
    source_label: str,
    paragraph_profile: Dict[str, Any],
    top_k: int,
    observer_logppl: float,
    observer_ppl: float,
    performer_logppl: float,
    performer_ppl: float,
    logxppl: float,
    xppl: float,
    binoculars_score: float,
) -> str:
    rows = list(paragraph_profile.get("rows", []))
    if not rows:
        return (
            "# Perplexity Heatmap\n\n"
            f"Source: `{source_label}`\n\n"
            "No paragraph spans were available in the analyzed token window.\n"
        )

    low_rows, high_rows, annotations, endnotes = _prepare_heatmap_annotations(
        rows=rows,
        top_k=top_k,
        observer_logppl=observer_logppl,
    )

    out: List[str] = []
    out.append("# Perplexity Heatmap")
    out.append("")
    out.append(f"Source: `{source_label}`")
    out.append("")
    out.append("## Summary")
    out.append(f"- Observer logPPL: `{observer_logppl:.6f}` (PPL `{observer_ppl:.3f}`)")
    out.append(f"- Performer logPPL: `{performer_logppl:.6f}` (PPL `{performer_ppl:.3f}`)")
    out.append(f"- Cross logXPPL: `{logxppl:.6f}` (XPPL `{xppl:.3f}`)")
    out.append(f"- Binoculars score B (high is more human-like): `{binoculars_score:.6f}`")
    out.append(f"- Red sections: {len(low_rows)} lowest paragraph logPPL")
    out.append(f"- Green sections: {len(high_rows)} highest paragraph logPPL")
    out.append("")
    out.append("## Text")
    out.append("")

    text_parts: List[str] = []
    pos = 0
    for row in sorted(rows, key=lambda r: r["char_start"]):
        s = row["char_start"]
        e = row["char_end"]
        if s > pos:
            text_parts.append(_normalize_console_text(text[pos:s]))
        ann = annotations.get(row["paragraph_id"])
        seg = _normalize_console_text(text[s:e])
        if ann is None:
            text_parts.append(seg)
        else:
            note_ref = f"[[#Notes Table|[{ann['note_num']}]]]"
            text_parts.append(
                f"<span class=\"{ann['css_class']}\" style=\"color: {ann['color']};\">"
                f"{seg}</span>{note_ref}"
            )
        pos = e
    if pos < len(text):
        text_parts.append(_normalize_console_text(text[pos:]))

    out.append("".join(text_parts))

    out.append("")
    out.append("## Notes Table")
    out.append("")
    out.append("| Index | Label | % contribution | Paragraph | logPPL | delta_vs_doc | delta_if_removed | Transitions | Chars | Tokens |")
    out.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for info in endnotes:
        pct = info["pct_contribution"]
        pct_str = f"{pct:+.2f}%" if np.isfinite(pct) else "n/a"
        out.append(
            f"| {info['note_num']} "
            f"| {info['label']} "
            f"| {pct_str} "
            f"| {info['paragraph_id']} "
            f"| {info['logPPL']:.6f} "
            f"| {info['delta_vs_doc_logPPL']:+.6f} "
            f"| {info['delta_doc_logPPL_if_removed']:+.6f} "
            f"| {info['transitions']} "
            f"| {info['char_start']}:{info['char_end']} "
            f"| {info['token_start']}:{info['token_end']} |"
        )
    out.append("")

    return "\n".join(out).rstrip() + "\n"


def _normalize_markdown_hardbreaks(text: str) -> str:
    # Normalize markdown hard-break markers expressed as trailing backslashes.
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\\[ \t]*\n", "\n", t)
    t = re.sub(r"\\[ \t]*$", "", t)
    # Also strip stray "\ " artifacts that can appear before quoted continuations.
    t = re.sub(r"(?<=\S)\\(?=\s)", "", t)
    return t


def _normalize_console_text(text: str) -> str:
    # Console output should not show markdown hard-break markers as literal backslashes.
    t = _normalize_markdown_hardbreaks(text)
    return t


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _visible_len(s: str) -> int:
    return len(ANSI_RE.sub("", s))


def _wrap_ansi_line(line: str, width: int) -> List[str]:
    if width <= 0 or _visible_len(line) <= width:
        return [line]

    tokens = re.findall(r"\S+\s*", line)
    if not tokens:
        return [line]

    lines: List[str] = []
    cur = ""
    for tok in tokens:
        if not cur:
            cur = tok
            continue
        if _visible_len(cur) + _visible_len(tok) <= width:
            cur += tok
        else:
            lines.append(cur.rstrip())
            cur = tok.lstrip()
    if cur:
        lines.append(cur.rstrip())

    return lines or [line]


def _wrap_console_text(text: str, width: int) -> str:
    out_lines: List[str] = []
    for raw_line in text.split("\n"):
        if raw_line.strip() == "":
            if out_lines and out_lines[-1] != "":
                out_lines.append("")
            continue
        out_lines.extend(_wrap_ansi_line(raw_line, width))
    while out_lines and out_lines[-1] == "":
        out_lines.pop()
    return "\n".join(out_lines)


def _console_target_width() -> int:
    cols = shutil.get_terminal_size(fallback=(120, 24)).columns
    target = int(cols * 0.85)
    return max(72, target)


def _truncate_cell(value: str, max_len: int) -> str:
    if max_len <= 1:
        return value[:max_len]
    if len(value) <= max_len:
        return value
    return value[: max_len - 1] + "…"


def _draw_console_table(headers: List[str], rows: List[List[str]], max_width: Optional[int] = None) -> str:
    widths: List[int] = []
    for idx, h in enumerate(headers):
        max_cell = len(h)
        for row in rows:
            max_cell = max(max_cell, len(row[idx]))
        widths.append(max_cell)

    if max_width is not None:
        # Minimal readable widths for compact heatmap table columns.
        mins = [3, 3, 7, 4, 7, 7, 7]
        total = sum(widths) + (3 * len(widths)) + 1
        if total > max_width:
            widths = [max(widths[i], mins[i]) for i in range(len(widths))]
            total = sum(widths) + (3 * len(widths)) + 1
            while total > max_width:
                changed = False
                for i in range(len(widths) - 1, -1, -1):
                    if widths[i] > mins[i]:
                        widths[i] -= 1
                        total -= 1
                        changed = True
                        if total <= max_width:
                            break
                if not changed:
                    break

    def border(left: str, mid: str, right: str, fill: str = "─") -> str:
        return left + mid.join(fill * (w + 2) for w in widths) + right

    def row_line(cells: List[str]) -> str:
        out_cells = []
        for i, c in enumerate(cells):
            c_fmt = _truncate_cell(c, widths[i])
            if i in {0, 3}:  # index, paragraph
                out_cells.append(c_fmt.rjust(widths[i]))
            elif i in {4, 5, 6}:  # numeric score columns
                out_cells.append(c_fmt.rjust(widths[i]))
            else:
                out_cells.append(c_fmt.ljust(widths[i]))
        return "│ " + " │ ".join(out_cells) + " │"

    out: List[str] = []
    out.append(border("┌", "┬", "┐"))
    out.append(row_line(headers))
    out.append(border("├", "┼", "┤"))
    for r in rows:
        out.append(row_line(r))
    out.append(border("└", "┴", "┘"))
    return "\n".join(out)


def _wrap_line(line: str, width: int) -> List[str]:
    if len(line) <= width:
        return [line]
    words = re.findall(r"\S+\s*", line)
    if not words:
        return [line]
    out: List[str] = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
            continue
        if len(cur) + len(w) <= width:
            cur += w
        else:
            out.append(cur.rstrip())
            cur = w.lstrip()
    if cur:
        out.append(cur.rstrip())
    return out


def _format_summary_lines(lines: List[str], width: int) -> List[str]:
    out: List[str] = []
    for line in lines:
        wrapped = _wrap_line(line, width)
        out.extend(wrapped)
    return out


def _build_console_text_body(
    text: str,
    rows: List[Dict[str, Any]],
    annotations: Dict[int, Dict[str, Any]],
    red: str,
    green: str,
    reset: str,
) -> str:
    text_parts: List[str] = []
    pos = 0
    for row in sorted(rows, key=lambda r: r["char_start"]):
        s = row["char_start"]
        e = row["char_end"]
        if s > pos:
            text_parts.append(_normalize_console_text(text[pos:s]))
        ann = annotations.get(row["paragraph_id"])
        seg = _normalize_console_text(text[s:e])
        if ann is None:
            text_parts.append(seg)
        else:
            color = red if ann["label"] == "LOW" else green
            text_parts.append(f"{color}{seg}{reset}[{ann['note_num']}]")
        pos = e
    if pos < len(text):
        text_parts.append(_normalize_console_text(text[pos:]))
    return "".join(text_parts)


def _build_console_table_rows(endnotes: List[Dict[str, Any]]) -> List[List[str]]:
    table_rows: List[List[str]] = []
    for info in endnotes:
        pct = info["pct_contribution"]
        pct_str = f"{pct:+.2f}%" if np.isfinite(pct) else "n/a"
        table_rows.append(
            [
                str(info["note_num"]),
                info["label"],
                pct_str,
                str(info["paragraph_id"]),
                f"{info['logPPL']:.6f}",
                f"{info['delta_vs_doc_logPPL']:+.6f}",
                f"{info['delta_doc_logPPL_if_removed']:+.6f}",
            ]
        )
    return table_rows


def _build_console_table(max_width: int, endnotes: List[Dict[str, Any]]) -> str:
    headers = [
        "Idx",
        "Lbl",
        "%Contrib",
        "Para",
        "logPPL",
        "dDoc",
        "dRm",
    ]
    table_rows = _build_console_table_rows(endnotes)
    return _draw_console_table(headers, table_rows, max_width=max_width)


def _build_summary_block(
    observer_logppl: float,
    observer_ppl: float,
    performer_logppl: float,
    performer_ppl: float,
    logxppl: float,
    xppl: float,
    binoculars_score: float,
    low_count: int,
    high_count: int,
    width: int,
) -> List[str]:
    lines = [
        f"- Observer logPPL: {observer_logppl:.6f} (PPL {observer_ppl:.3f})",
        f"- Performer logPPL: {performer_logppl:.6f} (PPL {performer_ppl:.3f})",
        f"- Cross logXPPL: {logxppl:.6f} (XPPL {xppl:.3f})",
        f"- Binoculars score B (high is more human-like): {binoculars_score:.6f}",
        f"- Red sections: {low_count} lowest paragraph logPPL",
        f"- Green sections: {high_count} highest paragraph logPPL",
    ]
    return _format_summary_lines(lines, width)


def build_heatmap_output_console(
    text: str,
    source_label: str,
    paragraph_profile: Dict[str, Any],
    top_k: int,
    observer_logppl: float,
    observer_ppl: float,
    performer_logppl: float,
    performer_ppl: float,
    logxppl: float,
    xppl: float,
    binoculars_score: float,
    force_color: Optional[bool] = None,
) -> str:
    rows = list(paragraph_profile.get("rows", []))
    if not rows:
        return (
            "Perplexity Heatmap\n\n"
            f"Source: {source_label}\n\n"
            "No paragraph spans were available in the analyzed token window.\n"
        )

    low_rows, high_rows, annotations, endnotes = _prepare_heatmap_annotations(
        rows=rows,
        top_k=top_k,
        observer_logppl=observer_logppl,
    )

    if force_color is None:
        # Default to colorized console output for heatmap readability.
        use_color = True
    else:
        use_color = force_color

    red = "\033[31m" if use_color else ""
    green = "\033[32m" if use_color else ""
    reset = "\033[0m" if use_color else ""
    width = _console_target_width()

    out: List[str] = []
    out.append("Perplexity Heatmap")
    out.append("")
    out.extend(_wrap_line(f"Source: {source_label}", width))
    out.append("")
    out.append("Summary")
    out.extend(
        _build_summary_block(
            observer_logppl=observer_logppl,
            observer_ppl=observer_ppl,
            performer_logppl=performer_logppl,
            performer_ppl=performer_ppl,
            logxppl=logxppl,
            xppl=xppl,
            binoculars_score=binoculars_score,
            low_count=len(low_rows),
            high_count=len(high_rows),
            width=width,
        )
    )
    out.append("")
    out.append("Text")
    out.append("")

    text_body = _build_console_text_body(
        text=text,
        rows=rows,
        annotations=annotations,
        red=red,
        green=green,
        reset=reset,
    )
    out.append(_wrap_console_text(text_body, width))
    out.append("")
    out.append("Notes Table")
    out.append("")
    out.append(_build_console_table(max_width=width, endnotes=endnotes))
    out.append("")

    return "\n".join(out)


def infer_heatmap_output_path(input_path: Optional[str]) -> str:
    if input_path and input_path != "-":
        src_dir = os.path.dirname(input_path) or "."
        stem = os.path.splitext(os.path.basename(input_path))[0]
        return os.path.join(src_dir, f"{stem}_heatmap.md")
    return "heatmap.md"


def backup_existing_file(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = f"{path}.{stamp}.bak"
    idx = 1
    while os.path.exists(candidate):
        idx += 1
        candidate = f"{path}.{stamp}.{idx}.bak"

    os.replace(path, candidate)
    return candidate


# ----------------------------
# Main pipeline
# ----------------------------

def analyze_text_document(
    cfg_path: str,
    text: str,
    input_label: str,
    diagnose_paragraphs: bool,
    diagnose_top_k: int,
    need_paragraph_profile: bool,
    text_max_tokens_override: Optional[int] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    cfg, tcfg, ccfg = load_config(cfg_path)
    if text_max_tokens_override is not None:
        if int(text_max_tokens_override) < 0:
            raise ValueError(f"text_max_tokens_override={text_max_tokens_override} must be >= 0.")
        tcfg.max_tokens = int(text_max_tokens_override)

    observer_section = cfg["observer"]
    performer_section = cfg["performer"]

    obs_path = observer_section.get("model_path")
    perf_path = performer_section.get("model_path")
    if not obs_path or not perf_path:
        raise ValueError("observer.model_path and performer.model_path are required.")

    text_bytes = text.encode("utf-8", errors="replace")

    # Tokenize with vocab_only model(s) to decide n_ctx and ensure tokenizer alignment.
    tokens_obs_full = tokenize_with_vocab_only(obs_path, text_bytes, tcfg)
    tokens_obs = maybe_truncate_tokens(tokens_obs_full, tcfg.max_tokens)

    tokens_perf_full = tokenize_with_vocab_only(perf_path, text_bytes, tcfg)
    tokens_perf = maybe_truncate_tokens(tokens_perf_full, tcfg.max_tokens)

    if tokens_obs != tokens_perf:
        raise ValueError(
            "Tokenizer mismatch: the two models do not tokenize the input identically. "
            "Use two models from the same family/tokenizer (e.g., base + instruct sibling)."
        )

    tokens = tokens_obs
    truncated_by_limit = len(tokens_obs_full) > len(tokens)
    if len(tokens) < 2:
        raise ValueError("Text is too short after tokenization (need at least 2 tokens).")

    needed_ctx = len(tokens)

    if ccfg.dir:
        cache_dir = ccfg.dir
        ensure_dir(cache_dir)
        temp_dir_ctx = None
    else:
        temp_dir_ctx = tempfile.TemporaryDirectory(prefix="binoculars_cache_")
        cache_dir = temp_dir_ctx.name

    cache_dtype = np.float16 if ccfg.dtype == "float16" else np.float32
    logits1_path = os.path.join(cache_dir, "observer_logits.dat")

    paragraph_profile: Optional[Dict[str, Any]] = None
    diag_result: Optional[Dict[str, Any]] = None

    obs = None
    perf = None
    try:
        n_ctx_obs = infer_n_ctx(observer_section, needed_ctx)

        obs_cfg = dict(observer_section)
        obs_cfg["model_path"] = obs_path
        obs_cfg["n_ctx"] = n_ctx_obs
        obs_cfg["logits_all"] = True
        obs_cfg.setdefault("verbose", False)

        obs_cfg = filter_llama_kwargs(obs_cfg)
        obs = build_llama_instance(obs_cfg)

        obs.reset()
        obs.eval(tokens)

        n_vocab_obs = obs.n_vocab()
        scores1 = obs.scores[:needed_ctx, :n_vocab_obs]
        observer_losses = compute_transition_losses(scores1, tokens)
        logppl_obs = float(np.mean(observer_losses))
        ppl_obs = float(np.exp(logppl_obs))

        if diagnose_paragraphs or need_paragraph_profile:
            paragraph_profile = compute_paragraph_profile(
                text=text,
                tcfg=tcfg,
                obs_model=obs,
                tokens=tokens,
                transition_losses=observer_losses,
            )
            analyzed_char_end = len(text)
            if truncated_by_limit:
                analyzed_char_end = estimate_char_end_for_token_limit(
                    text=text,
                    model=obs,
                    tcfg=tcfg,
                    token_limit=len(tokens),
                )
            paragraph_profile["analyzed_char_end"] = int(max(0, min(len(text), analyzed_char_end)))
            paragraph_profile["truncated_by_limit"] = bool(truncated_by_limit)
            paragraph_profile["analyzed_tokens"] = int(len(tokens))
            paragraph_profile["tokens_before_limit"] = int(len(tokens_obs_full))

        if diagnose_paragraphs and paragraph_profile is not None:
            rows = list(paragraph_profile.get("rows", []))
            rows.sort(key=lambda r: r["delta_doc_logPPL_if_removed"], reverse=True)
            diag_result = {
                "unit": paragraph_profile.get("unit", "paragraph"),
                "total_paragraphs": paragraph_profile.get("total_paragraphs", 0),
                "paragraphs_with_transitions": paragraph_profile.get("paragraphs_with_transitions", 0),
                "doc_logPPL": paragraph_profile.get("doc_logPPL", logppl_obs),
                "ranking_metric": "delta_doc_logPPL_if_removed",
                "top_low_perplexity_hotspots": rows[: max(1, int(diagnose_top_k))],
            }

        save_arr = scores1[:needed_ctx - 1, :n_vocab_obs]
        save_logits_memmap(logits1_path, save_arr, cache_dtype)

        close_llama(obs)
        del obs
        gc.collect()
        obs = None

        n_ctx_perf = infer_n_ctx(performer_section, needed_ctx)

        perf_cfg = dict(performer_section)
        perf_cfg["model_path"] = perf_path
        perf_cfg["n_ctx"] = n_ctx_perf
        perf_cfg["logits_all"] = True
        perf_cfg.setdefault("verbose", False)

        perf_cfg = filter_llama_kwargs(perf_cfg)
        perf = build_llama_instance(perf_cfg)

        perf.reset()
        perf.eval(tokens)

        n_vocab_perf = perf.n_vocab()
        if n_vocab_perf != n_vocab_obs:
            raise ValueError(
                f"Vocab size mismatch: observer n_vocab={n_vocab_obs}, performer n_vocab={n_vocab_perf}. "
                "Binoculars cross-entropy requires aligned vocab."
            )

        scores2 = perf.scores[:needed_ctx, :n_vocab_perf]
        logppl_perf, ppl_perf = compute_logppl_from_scores(scores2, tokens)

        logits1_mm = load_logits_memmap(
            logits1_path,
            shape=(needed_ctx - 1, n_vocab_obs),
            dtype=cache_dtype,
        )
        logxppl, xppl = compute_cross_logxppl(logits1_mm, scores2, tokens)
        del logits1_mm

        B = float(logppl_obs / logxppl) if logxppl != 0.0 else float("inf")

        result: Dict[str, Any] = {
            "input": {
                "path": input_label,
                "chars": len(text),
                "tokens": len(tokens),
                "tokens_before_limit": len(tokens_obs_full),
                "transitions": len(tokens) - 1,
                "markdown_preserved": True,
                "max_tokens_limit": int(tcfg.max_tokens),
                "truncated_by_limit": bool(truncated_by_limit),
            },
            "observer": {
                "model_path": obs_path,
                "logPPL": logppl_obs,
                "PPL": ppl_obs,
            },
            "performer": {
                "model_path": perf_path,
                "logPPL": logppl_perf,
                "PPL": ppl_perf,
            },
            "cross": {
                "logXPPL": logxppl,
                "XPPL": xppl,
            },
            "binoculars": {
                "score": B,
                "definition": "B = logPPL(observer) / logXPPL(observer, performer)",
            },
            "cache": {
                "dir": cache_dir,
                "dtype": ccfg.dtype,
                "kept": ccfg.keep,
            },
        }
        if diag_result is not None:
            result["diagnostics"] = {"low_perplexity_spans": diag_result}

        return result, paragraph_profile

    finally:
        close_llama(obs)
        close_llama(perf)
        try:
            del obs
        except Exception:
            pass
        try:
            del perf
        except Exception:
            pass
        gc.collect()

        if not ccfg.keep:
            try:
                if os.path.exists(logits1_path):
                    os.remove(logits1_path)
            except Exception:
                pass
            if temp_dir_ctx is not None:
                temp_dir_ctx.cleanup()


def _extract_completion_text(response: Any) -> str:
    if not isinstance(response, dict):
        return ""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    msg = first.get("message")
    if isinstance(msg, dict):
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
    text = first.get("text", "")
    if isinstance(text, str):
        return text
    return ""


def _normalize_generated_rewrite(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""

    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            body = lines[1:-1]
            if body:
                s = "\n".join(body).strip()

    s = re.sub(r"^\s*(?:option\s*)?\d+\s*[\)\].:-]\s*", "", s, flags=re.IGNORECASE)
    if len(s) >= 2 and ((s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'"))):
        s = s[1:-1].strip()
    return s


def _semantic_text_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _score_observer_logppl_batch(
    cfg_path: str,
    texts: List[str],
    text_max_tokens_override: Optional[int] = None,
) -> List[Dict[str, float]]:
    if not texts:
        return []

    cfg, tcfg, _ccfg = load_config(cfg_path)
    if text_max_tokens_override is not None:
        if int(text_max_tokens_override) < 0:
            raise ValueError(f"text_max_tokens_override={text_max_tokens_override} must be >= 0.")
        tcfg.max_tokens = int(text_max_tokens_override)

    observer_section = cfg["observer"]
    obs_path = observer_section.get("model_path")
    if not obs_path:
        raise ValueError("observer.model_path is required.")

    tokens_per_text: List[List[int]] = []
    for text in texts:
        tok_full = tokenize_with_vocab_only(obs_path, text.encode("utf-8", errors="replace"), tcfg)
        tok = maybe_truncate_tokens(tok_full, tcfg.max_tokens)
        if len(tok) < 2:
            raise ValueError("Local rewrite scoring text is too short after tokenization.")
        tokens_per_text.append(tok)

    needed_ctx = max(len(toks) for toks in tokens_per_text)
    n_ctx_obs = infer_n_ctx(observer_section, needed_ctx)

    obs_cfg = dict(observer_section)
    obs_cfg["model_path"] = obs_path
    obs_cfg["n_ctx"] = n_ctx_obs
    obs_cfg["logits_all"] = True
    obs_cfg.setdefault("verbose", False)
    obs_cfg = filter_llama_kwargs(obs_cfg)

    obs = None
    out: List[Dict[str, float]] = []
    try:
        obs = build_llama_instance(obs_cfg)
        n_vocab_obs = obs.n_vocab()
        for toks in tokens_per_text:
            obs.reset()
            obs.eval(toks)
            scores = obs.scores[: len(toks), :n_vocab_obs]
            logppl, _ppl = compute_logppl_from_scores(scores, toks)
            out.append(
                {
                    "logPPL": float(logppl),
                    "transitions": int(len(toks) - 1),
                }
            )
        return out
    finally:
        close_llama(obs)
        try:
            del obs
        except Exception:
            pass
        gc.collect()


def _choose_rewrite_window_bounds(
    text: str,
    span_start: int,
    span_end: int,
    neighbor_paragraphs: int = 1,
    pad_chars: int = 120,
    max_window_chars: int = 2600,
) -> Tuple[int, int]:
    n = len(text)
    start = max(0, min(n, int(span_start)))
    end = max(start, min(n, int(span_end)))
    if n <= 0:
        return (0, 0)

    spans = split_markdown_paragraph_spans(text)
    hit_idx: Optional[int] = None
    for i, (s, e) in enumerate(spans):
        if end > s and start < e:
            hit_idx = i
            break

    if hit_idx is None or not spans:
        w_start = max(0, start - 900)
        w_end = min(n, end + 900)
    else:
        left_idx = max(0, hit_idx - max(0, int(neighbor_paragraphs)))
        right_idx = min(len(spans) - 1, hit_idx + max(0, int(neighbor_paragraphs)))
        w_start = spans[left_idx][0]
        w_end = spans[right_idx][1]

    w_start = max(0, w_start - max(0, int(pad_chars)))
    w_end = min(n, w_end + max(0, int(pad_chars)))

    max_chars = max(300, int(max_window_chars))
    if (w_end - w_start) > max_chars:
        center = (start + end) // 2
        half = max_chars // 2
        w_start = max(0, center - half)
        w_end = min(n, w_start + max_chars)
        w_start = max(0, w_end - max_chars)

    return (w_start, w_end)


def _clip_text_middle(text: str, max_chars: int) -> str:
    s = str(text or "")
    cap = max(80, int(max_chars))
    if len(s) <= cap:
        return s
    head = max(24, (cap - 9) // 2)
    tail = max(24, cap - head - 9)
    return s[:head] + "\n[...]\n" + s[-tail:]


def _paragraph_index_for_span(
    spans: List[Tuple[int, int]],
    start: int,
    end: int,
) -> Optional[int]:
    for i, (s, e) in enumerate(spans):
        if end > s and start < e:
            return i
    return None


def _build_rewrite_prompts(
    full_text: str,
    span_start: int,
    span_end: int,
    llm_cfg: Optional[RewriteLLMConfig],
) -> Tuple[str, str, str]:
    start = max(0, min(len(full_text), int(span_start)))
    end = max(start, min(len(full_text), int(span_end)))
    target = full_text[start:end]
    if not target.strip():
        raise ValueError("Selected low-perplexity span is empty.")

    if llm_cfg is None:
        context_chars_each_side = 1800
        context_paragraphs_each_side = 2
        context_window_max_chars = 5200
    else:
        context_chars_each_side = int(llm_cfg.context_chars_each_side)
        context_paragraphs_each_side = int(llm_cfg.context_paragraphs_each_side)
        context_window_max_chars = int(llm_cfg.context_window_max_chars)

    spans = split_markdown_paragraph_spans(full_text)
    para_idx = _paragraph_index_for_span(spans, start, end)
    prev_para = ""
    next_para = ""
    if para_idx is not None and spans:
        if para_idx > 0:
            ps, pe = spans[para_idx - 1]
            prev_para = full_text[ps:pe].strip()
        if para_idx + 1 < len(spans):
            ns, ne = spans[para_idx + 1]
            next_para = full_text[ns:ne].strip()

    win_start, win_end = _choose_rewrite_window_bounds(
        full_text,
        start,
        end,
        neighbor_paragraphs=context_paragraphs_each_side,
        pad_chars=220,
        max_window_chars=context_window_max_chars,
    )

    left_start = max(win_start, start - context_chars_each_side)
    right_end = min(win_end, end + context_chars_each_side)
    left = full_text[left_start:start]
    right = full_text[end:right_end]

    window_marked = (
        full_text[win_start:start]
        + "\n[[TARGET SPAN START]]\n"
        + full_text[start:end]
        + "\n[[TARGET SPAN END]]\n"
        + full_text[end:win_end]
    )

    left = _clip_text_middle(left.strip(), min(context_chars_each_side, 3600))
    right = _clip_text_middle(right.strip(), min(context_chars_each_side, 3600))
    prev_para = _clip_text_middle(prev_para, 1200)
    next_para = _clip_text_middle(next_para, 1200)
    window_marked = _clip_text_middle(window_marked.strip(), context_window_max_chars)

    system_prompt = (
        "You rewrite exactly one markdown span from a document. Preserve meaning, facts, timeline, and voice. "
        "Do not add explanations or labels. Output only the rewritten span text. "
        "Return the complete rewritten span, including any lines left unchanged."
    )

    user_prompt = (
        "Rewrite the TARGET span so it sounds naturally authored while staying semantically equivalent.\n\n"
        "GLOBAL CONSTRAINTS:\n"
        "- Preserve named entities, claims, and implied facts.\n"
        "- Preserve tense, perspective, and speaker intent.\n"
        "- Preserve dialogue structure and punctuation style when present.\n"
        "- Preserve paragraph and line-break structure.\n"
        "- Do not omit lines; if a line is not rewritten, include it unchanged.\n"
        "- Keep similar length (roughly +/-30%).\n"
        "- No preface, no bullets, no numbering; return only rewritten TARGET span text.\n\n"
        "PREVIOUS PARAGRAPH:\n"
        f"{prev_para}\n\n"
        "CONTEXT BEFORE TARGET:\n"
        f"{left}\n\n"
        "TARGET SPAN:\n"
        f"{target}\n\n"
        "CONTEXT AFTER TARGET:\n"
        f"{right}\n\n"
        "NEXT PARAGRAPH:\n"
        f"{next_para}\n\n"
        "WINDOW WITH TARGET MARKERS (for local coherence):\n"
        f"{window_marked}"
    )

    return system_prompt, user_prompt, target


def _generate_rewrites_via_internal_performer(
    cfg_path: str,
    system_prompt: str,
    user_prompt: str,
    target_text: str,
    option_count: int,
    status_callback: Optional[Callable[[str], None]] = None,
) -> List[str]:
    n_opts = max(1, int(option_count))
    cfg, _tcfg, _ccfg = load_config(cfg_path)
    performer_section = cfg["performer"]
    perf_path = performer_section.get("model_path")
    if not perf_path:
        raise ValueError("performer.model_path is required.")

    perf_cfg = dict(performer_section)
    perf_cfg["model_path"] = perf_path
    configured_n_ctx = int(perf_cfg.get("n_ctx", 0) or 0)
    perf_cfg["logits_all"] = False
    perf_cfg.setdefault("verbose", False)
    try:
        target_tokens_preview = tokenize_with_vocab_only(
            perf_path,
            target_text.encode("utf-8", errors="replace"),
            TextConfig(add_bos=False, special_tokens=False, max_tokens=0),
        )
        target_tok_len = len(target_tokens_preview)
    except Exception:
        target_tok_len = max(12, len(target_text) // 4)
    max_out_tokens = min(220, max(48, int(target_tok_len * 1.35) + 20))

    prompt_chars = len(system_prompt) + len(user_prompt) + len(target_text) + 240
    prompt_tok_est = max(256, prompt_chars // 3)
    n_ctx_needed = min(4096, max(1536, prompt_tok_est + max_out_tokens + 192))
    if configured_n_ctx <= 0:
        perf_cfg["n_ctx"] = n_ctx_needed
    else:
        perf_cfg["n_ctx"] = max(int(configured_n_ctx), n_ctx_needed)
    perf_cfg = filter_llama_kwargs(perf_cfg)

    perf = None
    try:
        if status_callback is not None:
            status_callback(f"Loading internal performer model (n_ctx={perf_cfg.get('n_ctx')})...")
        perf = build_llama_instance(perf_cfg)

        temperatures = [0.66, 0.82, 0.98, 0.74, 0.90]
        rewrites: List[str] = []
        seen: Set[str] = set()

        for idx, temp in enumerate(temperatures, start=1):
            if len(rewrites) >= n_opts:
                break
            if status_callback is not None:
                status_callback(f"Generating option candidates (pass {idx}/{len(temperatures)})...")
            raw_text = ""
            try:
                if hasattr(perf, "create_chat_completion"):
                    resp = perf.create_chat_completion(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=float(temp),
                        top_p=0.95,
                        max_tokens=max_out_tokens,
                        repeat_penalty=1.08,
                    )
                    raw_text = _extract_completion_text(resp)
                else:
                    resp = perf.create_completion(
                        prompt=f"{system_prompt}\n\n{user_prompt}\n\nRewrite:\n",
                        temperature=float(temp),
                        top_p=0.95,
                        max_tokens=max_out_tokens,
                        repeat_penalty=1.08,
                    )
                    raw_text = _extract_completion_text(resp)
            except Exception:
                continue

            candidate = _normalize_generated_rewrite(raw_text)
            if not candidate:
                continue
            key = _semantic_text_key(candidate)
            if not key or key in seen:
                continue
            seen.add(key)
            rewrites.append(candidate)

        if not rewrites:
            raise ValueError("Could not generate rewrite options with internal performer model.")

        while len(rewrites) < n_opts:
            rewrites.append(rewrites[-1])
        return rewrites[:n_opts]
    finally:
        close_llama(perf)
        try:
            del perf
        except Exception:
            pass
        gc.collect()


def generate_rewrite_candidates_for_span(
    cfg_path: str,
    full_text: str,
    span_start: int,
    span_end: int,
    option_count: int = 3,
    status_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[List[str], str, Optional[str]]:
    n_opts = max(1, int(option_count))
    start = max(0, min(len(full_text), int(span_start)))
    end = max(start, min(len(full_text), int(span_end)))

    llm_cfg, llm_cfg_issue = load_optional_rewrite_llm_config()
    system_prompt, user_prompt, target = _build_rewrite_prompts(
        full_text=full_text,
        span_start=start,
        span_end=end,
        llm_cfg=llm_cfg,
    )
    if llm_cfg is not None:
        try:
            if status_callback is not None:
                status_callback("Trying external rewrite LLM...")
            rewrites = _generate_rewrites_via_openai_compatible(
                llm_cfg=llm_cfg,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                option_count=n_opts,
            )
            return rewrites, "external-llm", None
        except (urllib.error.URLError, urllib.error.HTTPError, socket.timeout, TimeoutError, OSError, ValueError) as exc:
            if status_callback is not None:
                status_callback("External LLM unavailable; falling back to internal model...")
            fallback_reason = f"external LLM failed ({type(exc).__name__}: {exc})"
        except Exception as exc:
            if status_callback is not None:
                status_callback("External LLM failed; falling back to internal model...")
            fallback_reason = f"external LLM failed ({type(exc).__name__}: {exc})"
        rewrites = _generate_rewrites_via_internal_performer(
            cfg_path=cfg_path,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            target_text=target,
            option_count=n_opts,
            status_callback=status_callback,
        )
        return rewrites, "internal-fallback", fallback_reason

    if llm_cfg_issue and status_callback is not None:
        status_callback("LLM config invalid; using internal model.")

    rewrites = _generate_rewrites_via_internal_performer(
        cfg_path=cfg_path,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        target_text=target,
        option_count=n_opts,
        status_callback=status_callback,
    )
    return rewrites, "internal", llm_cfg_issue


def estimate_rewrite_b_impact_options(
    cfg_path: str,
    full_text: str,
    span_start: int,
    span_end: int,
    rewrites: List[str],
    base_doc_b: float,
    base_doc_observer_logppl: float,
    base_doc_cross_logxppl: float,
    base_doc_transitions: int,
    text_max_tokens_override: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not rewrites:
        return []

    start = max(0, min(len(full_text), int(span_start)))
    end = max(start, min(len(full_text), int(span_end)))
    win_start, win_end = _choose_rewrite_window_bounds(full_text, start, end)
    base_local = full_text[win_start:win_end]
    rel_start = max(0, min(len(base_local), start - win_start))
    rel_end = max(rel_start, min(len(base_local), end - win_start))

    local_texts: List[str] = [base_local]
    for rw in rewrites:
        local_texts.append(base_local[:rel_start] + rw + base_local[rel_end:])

    local_scores = _score_observer_logppl_batch(
        cfg_path=cfg_path,
        texts=local_texts,
        text_max_tokens_override=text_max_tokens_override,
    )
    if len(local_scores) != len(local_texts):
        raise ValueError("Internal error: local rewrite scoring did not produce expected score count.")

    base_local_logppl = float(local_scores[0]["logPPL"])
    base_local_transitions = int(local_scores[0]["transitions"])
    doc_transitions = max(1, int(base_doc_transitions))
    base_local_total = base_local_logppl * float(base_local_transitions)

    out: List[Dict[str, Any]] = []
    for rw, sc in zip(rewrites, local_scores[1:]):
        cand_local_logppl = float(sc["logPPL"])
        cand_local_transitions = int(sc["transitions"])
        cand_local_total = cand_local_logppl * float(cand_local_transitions)
        delta_doc_observer = (cand_local_total - base_local_total) / float(doc_transitions)
        approx_observer = float(base_doc_observer_logppl + delta_doc_observer)
        if np.isfinite(base_doc_cross_logxppl) and base_doc_cross_logxppl != 0.0:
            approx_b = float(approx_observer / base_doc_cross_logxppl)
        else:
            approx_b = float("inf")
        if np.isfinite(approx_b) and np.isfinite(base_doc_b):
            delta_b = float(approx_b - base_doc_b)
        else:
            delta_b = float("nan")
        out.append(
            {
                "text": rw,
                "approx_B": approx_b,
                "delta_B": delta_b,
                "approx_observer_logPPL": approx_observer,
                "local_logPPL": cand_local_logppl,
                "local_transitions": cand_local_transitions,
            }
        )
    def _rank_key(opt: Dict[str, Any]) -> Tuple[int, float, int, float]:
        raw_delta = opt.get("delta_B", float("nan"))
        raw_b = opt.get("approx_B", float("nan"))
        try:
            delta_b = float(raw_delta)
        except Exception:
            delta_b = float("nan")
        try:
            approx_b = float(raw_b)
        except Exception:
            approx_b = float("nan")

        has_delta = 1 if np.isfinite(delta_b) else 0
        has_b = 1 if np.isfinite(approx_b) else 0
        delta_key = delta_b if has_delta else float("-inf")
        b_key = approx_b if has_b else float("-inf")
        return (has_delta, delta_key, has_b, b_key)

    out.sort(key=_rank_key, reverse=True)
    return out


def launch_gui(
    cfg_path: str,
    gui_input_path: str,
    top_k: int,
    text_max_tokens_override: Optional[int] = None,
) -> int:
    try:
        import tkinter as tk
        import tkinter.font as tkfont
        from tkinter import filedialog, messagebox
    except Exception as exc:
        raise ValueError("Tkinter is required for --gui mode but could not be imported.") from exc

    src_path = os.path.abspath(gui_input_path)
    if not os.path.isfile(src_path):
        raise ValueError(f"--gui file not found: {src_path}")

    with open(src_path, "r", encoding="utf-8") as f:
        initial_text = _normalize_markdown_hardbreaks(f.read())
    home_dir = os.path.expanduser("~")
    if not os.path.isdir(home_dir):
        home_dir = os.path.dirname(src_path) or "."

    try:
        # className influences WM_CLASS on Linux/X11 docks/taskbars.
        root = tk.Tk(className="Binoculars")
    except Exception:
        try:
            root = tk.Tk()
        except Exception as exc:
            raise ValueError("Unable to start GUI window. Check your display environment.") from exc
    try:
        root.wm_class("Binoculars")
    except Exception:
        pass
    try:
        root.tk.call("tk", "appname", "Binoculars")
    except Exception:
        pass
    try:
        root.iconname("Binoculars")
    except Exception:
        pass

    def maximize_root_window() -> None:
        # Ubuntu and different WMs expose maximize via different APIs.
        try:
            root.attributes("-zoomed", True)
            return
        except Exception:
            pass
        try:
            root.state("zoomed")
            return
        except Exception:
            pass
        try:
            root.update_idletasks()
            root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")
        except Exception:
            root.geometry("1600x1000")

    def choose_preview_font_family() -> str:
        preferred = ("Noto Sans", "Ubuntu", "DejaVu Sans", "Cantarell", "TkDefaultFont")
        try:
            available = set(tkfont.families(root))
        except Exception:
            return "TkDefaultFont"
        for family in preferred:
            if family in available:
                return family
        return "TkDefaultFont"

    def create_owl_icon_image() -> Optional[Any]:
        """
        Build a small built-in owl icon (big eyes) so GUI launches with a recognizable app icon
        without requiring external image assets.
        """
        try:
            size = 64
            icon = tk.PhotoImage(width=size, height=size)

            def rect(x1: int, y1: int, x2: int, y2: int, color: str) -> None:
                # Tk's PhotoImage rectangle upper bounds are exclusive.
                icon.put(color, to=(x1, y1, x2 + 1, y2 + 1))

            def circle(cx: int, cy: int, radius: int, color: str) -> None:
                r2 = radius * radius
                for y in range(cy - radius, cy + radius + 1):
                    if y < 0 or y >= size:
                        continue
                    for x in range(cx - radius, cx + radius + 1):
                        if x < 0 or x >= size:
                            continue
                        dx = x - cx
                        dy = y - cy
                        if (dx * dx) + (dy * dy) <= r2:
                            icon.put(color, to=(x, y))

            outline = "#1c1c1c"
            feather_outer = "#7c7c7c"
            feather_inner = "#9b9b9b"
            face = "#c8c8c8"
            eye_white = "#f0f0f0"
            pupil = "#151515"
            beak = "#b1b1b1"
            wing = "#676767"
            badge_outer = "#d0d0d0"
            badge_inner = "#bdbdbd"

            # Circular badge background so icon reads well at small sizes.
            circle(32, 32, 30, badge_outer)
            circle(32, 32, 27, badge_inner)

            # Owl body and head.
            circle(32, 36, 20, outline)
            circle(32, 36, 18, feather_outer)
            circle(32, 29, 16, feather_inner)

            # Ears.
            rect(18, 12, 24, 20, outline)
            rect(40, 12, 46, 20, outline)
            rect(19, 13, 23, 19, feather_outer)
            rect(41, 13, 45, 19, feather_outer)

            # Face mask.
            circle(32, 31, 12, face)

            # Big eyes.
            circle(24, 30, 9, outline)
            circle(40, 30, 9, outline)
            circle(24, 30, 7, eye_white)
            circle(40, 30, 7, eye_white)
            circle(24, 31, 4, pupil)
            circle(40, 31, 4, pupil)
            circle(22, 28, 2, "#d7d7d7")
            circle(38, 28, 2, "#d7d7d7")

            # Beak.
            rect(30, 35, 34, 39, outline)
            rect(31, 36, 33, 38, beak)

            # Wings.
            rect(14, 34, 21, 49, wing)
            rect(43, 34, 50, 49, wing)
            rect(14, 34, 15, 49, outline)
            rect(20, 34, 21, 49, outline)
            rect(43, 34, 44, 49, outline)
            rect(49, 34, 50, 49, outline)

            # Belly and feet.
            rect(26, 42, 38, 53, "#a7a7a7")
            rect(27, 43, 37, 52, "#b8b8b8")
            rect(24, 54, 29, 58, outline)
            rect(35, 54, 40, 58, outline)

            return icon
        except Exception:
            return None

    root.title(f"Binoculars - {os.path.basename(src_path)}")
    app_icon = create_owl_icon_image()
    if app_icon is not None:
        try:
            root.iconphoto(True, app_icon)
            # Keep a strong reference; Tk can drop icons if image is GC'd.
            root._binoculars_app_icon = app_icon  # type: ignore[attr-defined]
        except Exception:
            pass
    root.geometry("1200x800")
    root.configure(bg="#000000")
    maximize_root_window()
    preview_font_family = choose_preview_font_family()
    preview_font_size = 13

    toolbar = tk.Frame(root, bg="#000000")
    toolbar.pack(side="top", fill="x", padx=10, pady=8)

    split_pane = tk.PanedWindow(
        root,
        orient="horizontal",
        bg="#000000",
        bd=0,
        sashwidth=8,
        showhandle=False,
    )
    split_pane.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 8))

    editor_frame = tk.Frame(split_pane, bg="#000000")
    preview_frame = tk.Frame(split_pane, bg="#2a2a2a")
    split_pane.add(editor_frame, minsize=420)
    split_pane.add(preview_frame, minsize=280)

    status_var = tk.StringVar(value="Ready. Press Analyze to score and highlight this document.")
    status_label = tk.Label(
        root,
        textvariable=status_var,
        anchor="w",
        bg="#1f8a84",
        fg="#ecfffd",
        padx=8,
        pady=6,
    )
    status_label.pack(side="bottom", fill="x")

    debug_var = tk.StringVar(value="")
    debug_label = tk.Label(
        toolbar,
        textvariable=debug_var,
        anchor="e",
        bg="#000000",
        fg="#8ea3b8",
        padx=4,
    )
    debug_label.pack(side="right", padx=(8, 0))

    line_bars = tk.Canvas(
        editor_frame,
        width=66,
        bg="#000000",
        highlightthickness=0,
        borderwidth=0,
        relief="flat",
        takefocus=0,
    )
    line_bars.pack(side="left", fill="y")

    line_number_font = tkfont.nametofont("TkFixedFont")
    line_numbers = tk.Canvas(
        editor_frame,
        width=56,
        bg="#0a0a0a",
        relief="flat",
        takefocus=0,
        highlightthickness=0,
        borderwidth=0,
    )
    line_numbers.pack(side="left", fill="y")

    text_widget = tk.Text(
        editor_frame,
        wrap="word",
        undo=True,
        bg="#000000",
        fg="#efefef",
        insertbackground="#efefef",
        relief="flat",
        padx=12,
        pady=12,
    )
    scroll_y = tk.Scrollbar(editor_frame, orient="vertical", command=text_widget.yview)
    text_widget.configure(yscrollcommand=scroll_y.set)
    scroll_y.pack(side="right", fill="y")
    text_widget.pack(side="left", fill="both", expand=True)

    preview_body = tk.Frame(preview_frame, bg="#2a2a2a")
    preview_body.pack(side="top", fill="both", expand=True)

    preview_text = tk.Text(
        preview_body,
        wrap="word",
        state="disabled",
        takefocus=0,
        bg="#2a2a2a",
        fg="#e4e4e4",
        insertbackground="#e4e4e4",
        font=(preview_font_family, preview_font_size),
        relief="flat",
        padx=12,
        pady=12,
    )
    preview_scroll = tk.Scrollbar(preview_body, orient="vertical")
    preview_text.configure(yscrollcommand=preview_scroll.set)
    preview_scroll.pack(side="right", fill="y")
    preview_text.pack(side="left", fill="both", expand=True)

    synonym_panel = tk.Frame(
        preview_frame,
        bg="#1f4548",
        bd=0,
        relief="flat",
        highlightthickness=1,
        highlightbackground="#2a6568",
    )
    synonym_panel.pack(side="bottom", fill="x", padx=8, pady=(0, 8))
    synonym_header_var = tk.StringVar(value="Synonyms: click a word in the left pane.")
    synonym_header = tk.Label(
        synonym_panel,
        textvariable=synonym_header_var,
        anchor="w",
        justify="left",
        bg="#1f4548",
        fg="#e9ffff",
        padx=10,
        pady=6,
        font=("TkDefaultFont", 10, "bold"),
    )
    synonym_header.pack(side="top", fill="x")

    synonym_grid = tk.Frame(synonym_panel, bg="#1f4548")
    synonym_grid.pack(side="top", fill="x", padx=8, pady=(0, 4))
    for col_idx in range(SYNONYM_GRID_COLUMNS):
        synonym_grid.grid_columnconfigure(col_idx, weight=1)
    synonym_item_vars: List[Any] = [tk.StringVar(value="") for _ in range(SYNONYM_OPTION_COUNT)]
    synonym_item_labels: List[Any] = []
    synonym_item_width = 17 if SYNONYM_GRID_COLUMNS >= 3 else 26
    for i in range(SYNONYM_OPTION_COUNT):
        row = i // SYNONYM_GRID_COLUMNS
        col = i % SYNONYM_GRID_COLUMNS
        lbl = tk.Label(
            synonym_grid,
            textvariable=synonym_item_vars[i],
            anchor="w",
            justify="left",
            bg="#1f4548",
            fg="#e8f6f8",
            width=synonym_item_width,
            padx=6,
            pady=1,
            font=("TkDefaultFont", 9),
        )
        lbl.grid(row=row, column=col, sticky="w", padx=(0, 8), pady=1)
        synonym_item_labels.append(lbl)

    synonym_btn_frame = tk.Frame(synonym_panel, bg="#1f4548")
    synonym_btn_frame.pack(side="bottom", fill="x", padx=8, pady=(0, 8))
    synonym_buttons: List[Any] = []

    preview_text.tag_configure("md_h1", font=(preview_font_family, preview_font_size + 7, "bold"), spacing3=7)
    preview_text.tag_configure("md_h2", font=(preview_font_family, preview_font_size + 4, "bold"), spacing3=6)
    preview_text.tag_configure("md_h3", font=(preview_font_family, preview_font_size + 2, "bold"), spacing3=5)
    preview_text.tag_configure("md_bold", font=(preview_font_family, preview_font_size, "bold"))
    preview_text.tag_configure("md_italic", font=(preview_font_family, preview_font_size, "italic"))
    preview_text.tag_configure("md_code_inline", font=("TkFixedFont", 11), background="#1f1f1f")
    preview_text.tag_configure("md_code_block", font=("TkFixedFont", 11), background="#1f1f1f")
    preview_text.tag_configure("md_quote", foreground="#c9d1d9")
    preview_text.tag_configure("md_list", foreground="#e4e4e4")
    preview_text.tag_configure("md_hr", foreground="#8a8a8a")
    preview_text.tag_configure("md_link", foreground="#9fc3ff", underline=1)
    preview_text.tag_configure("preview_heat_low", background="#3a2323")
    preview_text.tag_configure("preview_heat_high", background="#223a2a")
    preview_text.tag_configure("preview_sel_low", background="#4a3434")
    preview_text.tag_configure("preview_sel_high", background="#31483a")
    preview_text.tag_configure("preview_sel_neutral", background="#606060")
    preview_text.tag_configure("preview_active_line", background="#3d3d3d")
    # Make unscored preview lines visibly distinct from analyzed content.
    preview_text.tag_configure("preview_unscored", foreground="#bcc4cf", background="#414a56")

    text_widget.tag_configure("edited", foreground="#ffd54f")
    text_widget.tag_configure("misspelled", underline=1, underlinefg="#ff4d4d")
    # Keep unscored source text dimmer with a cool background tint.
    text_widget.tag_configure("unscored", foreground="#8f9aaa", background="#111821")
    english_words = load_english_words()

    # Central GUI state store. Grouped logically:
    # - analysis metrics/status
    # - scheduled UI jobs
    # - synonym/rewrite transient state
    # - rendering caches (line map/contributions/priors)
    state: Dict[str, Any] = {
        "baseline_text": initial_text,
        "prev_text": initial_text,
        "last_saved_text": initial_text,
        "segment_tags": [],
        "segment_labels": {},
        "segment_infos": {},
        "prior_bg_tags": [],
        "prior_counter": 0,
        "last_b_score": None,
        "has_analysis": False,
        "b_score_stale": False,
        "last_analysis_status_core": "",
        "last_analysis_metrics": None,
        "analyzed_char_end": 0,
        "analysis_chunks": [],
        "analysis_chunk_id_seq": 0,
        "analysis_covered_until": 0,
        "analysis_next_available": False,
        "analyze_next_visible": False,
        "spell_version": 0,
        "last_spell_spans": None,
        "tooltip": None,
        "pending_edit_job": None,
        "pending_line_numbers_job": None,
        "pending_line_bars_job": None,
        "pending_spell_job": None,
        "pending_preview_job": None,
        "pending_focus_job": None,
        "pending_synonym_job": None,
        "synonym_request_id": 0,
        "synonym_options": [],
        "synonym_target_word": "",
        "synonym_target_start_idx": None,
        "synonym_target_end_idx": None,
        "synonym_cache": {},
        "preview_line_map": {},
        "line_contrib_map": {},
        "line_contrib_max_abs": 0.0,
        "line_contrib_last_line": 0,
        "line_contrib_top_lines": set(),
        "prior_line_contrib_maps": [],
        "analyze_cursor_idx": "1.0",
        "analyze_yview_top": 0.0,
        "line_count": 0,
        "sash_initialized": False,
        "internal_update": False,
        "analyzing": False,
        "progress_popup": None,
        "save_popup": None,
        "pending_status_restore_job": None,
        "undo_action": None,
        "rewrite_popup": None,
        "rewrite_request_id": 0,
        "rewrite_busy": False,
        "clear_priors_visible": False,
        "open_dialog_dir": home_dir,
        "open_dialog_resize_bindings_installed": False,
        "debug_enabled": bool(os.environ.get("BINOCULARS_GUI_DEBUG")),
        "preview_view_offset_lines": int(os.environ.get("BINOCULARS_PREVIEW_VIEW_OFFSET_LINES", "-3")),
    }
    if not state.get("debug_enabled", False):
        debug_label.pack_forget()

    def current_text() -> str:
        return text_widget.get("1.0", "end-1c")

    def merge_char_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        valid: List[Tuple[int, int]] = []
        for start_raw, end_raw in intervals:
            try:
                s = int(start_raw)
                e = int(end_raw)
            except Exception:
                continue
            if e <= s:
                continue
            valid.append((s, e))
        if not valid:
            return []
        valid.sort(key=lambda x: (x[0], x[1]))
        merged: List[Tuple[int, int]] = [valid[0]]
        for s, e in valid[1:]:
            ps, pe = merged[-1]
            if s <= pe:
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))
        return merged

    def analysis_chunk_list() -> List[Dict[str, Any]]:
        chunks_obj = state.get("analysis_chunks")
        if isinstance(chunks_obj, list):
            return chunks_obj
        return []

    def get_scored_intervals(text_snapshot: str) -> List[Tuple[int, int]]:
        text_len = len(text_snapshot)
        raw: List[Tuple[int, int]] = []
        for chunk in analysis_chunk_list():
            try:
                s = int(chunk.get("char_start", 0))
                e = int(chunk.get("analyzed_char_end", s))
            except Exception:
                continue
            s = max(0, min(text_len, s))
            e = max(s, min(text_len, e))
            if e > s:
                raw.append((s, e))
        return merge_char_intervals(raw)

    def get_unscored_intervals(text_snapshot: str) -> List[Tuple[int, int]]:
        text_len = len(text_snapshot)
        if not state.get("has_analysis") and not analysis_chunk_list():
            return []
        scored = get_scored_intervals(text_snapshot)
        if not scored:
            return [(0, text_len)] if text_len > 0 else []
        out: List[Tuple[int, int]] = []
        cursor = 0
        for s, e in scored:
            if s > cursor:
                out.append((cursor, s))
            cursor = max(cursor, e)
        if cursor < text_len:
            out.append((cursor, text_len))
        return out

    def shift_profile_to_global_char_offsets(
        profile: Optional[Dict[str, Any]], base_char_start: int
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(profile, dict):
            return None
        out = dict(profile)
        shifted_rows: List[Dict[str, Any]] = []
        for row in profile.get("rows", []):
            if not isinstance(row, dict):
                continue
            shifted = dict(row)
            try:
                shifted["char_start"] = int(base_char_start + int(row.get("char_start", 0)))
                shifted["char_end"] = int(base_char_start + int(row.get("char_end", row.get("char_start", 0))))
            except Exception:
                continue
            shifted_rows.append(shifted)
        out["rows"] = shifted_rows
        raw_end = profile.get("analyzed_char_end")
        if raw_end is not None:
            try:
                out["analyzed_char_end"] = int(base_char_start + int(raw_end))
            except Exception:
                pass
        return out

    def merge_chunk_descriptor(new_chunk: Dict[str, Any]) -> None:
        try:
            ns = int(new_chunk.get("char_start", 0))
            ne = int(new_chunk.get("analyzed_char_end", ns))
        except Exception:
            return
        if ne <= ns:
            return
        kept: List[Dict[str, Any]] = []
        for old_chunk in analysis_chunk_list():
            try:
                os_ = int(old_chunk.get("char_start", 0))
                oe = int(old_chunk.get("analyzed_char_end", os_))
            except Exception:
                continue
            # Replace any overlapping coverage interval with the newly analyzed chunk.
            if ne <= os_ or ns >= oe:
                kept.append(old_chunk)
        kept.append(new_chunk)
        kept.sort(key=lambda c: int(c.get("char_start", 0)))
        state["analysis_chunks"] = kept

    def recompute_chunk_coverage_state(text_snapshot: str) -> None:
        text_len = len(text_snapshot)
        merged = get_scored_intervals(text_snapshot)
        contiguous = 0
        for s, e in merged:
            if s > contiguous:
                break
            contiguous = max(contiguous, e)
        contiguous = max(0, min(text_len, contiguous))
        state["analysis_covered_until"] = int(contiguous)
        state["analyzed_char_end"] = int(contiguous)
        state["analysis_next_available"] = bool(merged) and contiguous < text_len

    def line_range_to_char_range(start_line: int, end_line: int) -> Tuple[int, int]:
        s_line = max(1, int(start_line))
        e_line = max(s_line, int(end_line))
        start_idx = f"{s_line}.0"
        end_idx = f"{e_line}.0 lineend+1c"
        return (char_offset_for_index(start_idx), char_offset_for_index(end_idx))

    def visible_line_range() -> Optional[Tuple[int, int]]:
        try:
            top_idx = text_widget.index("@0,0")
            top_line = int(str(top_idx).split(".")[0])
        except Exception:
            return None
        line_idx = top_idx
        bottom_line = top_line
        try:
            canvas_h = max(1, int(text_widget.winfo_height()))
        except Exception:
            canvas_h = 1
        while True:
            dline = text_widget.dlineinfo(line_idx)
            if dline is None:
                break
            y = float(dline[1])
            h = float(dline[3])
            if y > canvas_h:
                break
            try:
                bottom_line = int(str(line_idx).split(".")[0])
            except Exception:
                pass
            line_idx = text_widget.index(f"{line_idx}+1line")
        return (top_line, max(top_line, bottom_line))

    def chunk_overlap_with_span(chunk: Dict[str, Any], span_start: int, span_end: int) -> int:
        try:
            cs = int(chunk.get("char_start", 0))
            ce = int(chunk.get("analyzed_char_end", cs))
        except Exception:
            return 0
        start = max(cs, int(span_start))
        end = min(ce, int(span_end))
        return max(0, end - start)

    def resolve_active_chunk(text_snapshot: Optional[str] = None) -> Optional[Dict[str, Any]]:
        chunks = analysis_chunk_list()
        if not chunks:
            return None
        txt = current_text() if text_snapshot is None else text_snapshot
        txt_len = len(txt)

        sel_pair = selection_index_pair()
        if sel_pair is not None:
            sel_s = char_offset_for_index(sel_pair[0])
            sel_e = char_offset_for_index(sel_pair[1])
            if sel_e > sel_s:
                scored_best: Optional[Dict[str, Any]] = None
                scored_best_ov = 0
                for chunk in chunks:
                    ov = chunk_overlap_with_span(chunk, sel_s, sel_e)
                    if ov > scored_best_ov:
                        scored_best = chunk
                        scored_best_ov = ov
                if scored_best is not None and scored_best_ov > 0:
                    return scored_best

        vis = visible_line_range()
        cursor_line = 1
        try:
            cursor_line = int(str(text_widget.index("insert")).split(".")[0])
        except Exception:
            cursor_line = 1
        cursor_visible = vis is not None and vis[0] <= cursor_line <= vis[1]
        if cursor_visible:
            cursor_char = char_offset_for_index(text_widget.index("insert"))
            for chunk in chunks:
                try:
                    cs = int(chunk.get("char_start", 0))
                    ce = int(chunk.get("analyzed_char_end", cs))
                except Exception:
                    continue
                if cs <= cursor_char < ce:
                    return chunk

        if vis is not None:
            vis_s, vis_e = line_range_to_char_range(vis[0], vis[1])
            vis_s = max(0, min(txt_len, vis_s))
            vis_e = max(vis_s, min(txt_len, vis_e))
            best_vis: Optional[Dict[str, Any]] = None
            best_vis_ov = 0
            for chunk in chunks:
                ov = chunk_overlap_with_span(chunk, vis_s, vis_e)
                if ov > best_vis_ov:
                    best_vis = chunk
                    best_vis_ov = ov
            if best_vis is not None and best_vis_ov > 0:
                return best_vis

        anchor = char_offset_for_index(text_widget.index("insert"))
        best_chunk = None
        best_dist = None
        for chunk in chunks:
            try:
                cs = int(chunk.get("char_start", 0))
                ce = int(chunk.get("analyzed_char_end", cs))
            except Exception:
                continue
            if cs <= anchor <= ce:
                return chunk
            dist = min(abs(anchor - cs), abs(anchor - ce))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_chunk = chunk
        return best_chunk

    def chunk_metrics_from_descriptor(chunk: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(chunk, dict):
            return None
        metrics_obj = chunk.get("metrics")
        metrics = metrics_obj if isinstance(metrics_obj, dict) else None
        if metrics is None:
            return None
        out = dict(metrics)
        try:
            out["chunk_char_start"] = int(chunk.get("char_start", 0))
            out["chunk_char_end"] = int(chunk.get("analyzed_char_end", out["chunk_char_start"]))
        except Exception:
            pass
        return out

    def update_analysis_status_from_active_chunk() -> None:
        if not state.get("has_analysis"):
            return
        text_snapshot = current_text()
        active_chunk = resolve_active_chunk(text_snapshot)
        metrics = chunk_metrics_from_descriptor(active_chunk)
        if metrics is None:
            return
        set_analyze_next_visible(bool(state.get("analysis_next_available")))
        chunk_start = int(metrics.get("chunk_char_start", 0))
        chunk_end = int(metrics.get("chunk_char_end", chunk_start))
        chunk_start_line = line_number_for_char_end(text_snapshot, chunk_start + 1 if chunk_start < len(text_snapshot) else chunk_start)
        chunk_end_line = line_number_for_char_end(text_snapshot, max(chunk_start + 1, chunk_end))
        status_core = (
            "Binocular score B (high is more human-like): "
            f"{float(metrics.get('binoculars_score', float('nan'))):.6f} | "
            f"Chunk lines: {chunk_start_line}-{chunk_end_line} | "
            f"Observer logPPL: {float(metrics.get('observer_logPPL', float('nan'))):.6f} | "
            f"Performer logPPL: {float(metrics.get('performer_logPPL', float('nan'))):.6f} | "
            f"Cross logXPPL: {float(metrics.get('cross_logXPPL', float('nan'))):.6f}"
        )
        state["last_analysis_status_core"] = status_core
        state["last_analysis_metrics"] = {
            "binoculars_score": float(metrics.get("binoculars_score", float("nan"))),
            "observer_logPPL": float(metrics.get("observer_logPPL", float("nan"))),
            "performer_logPPL": float(metrics.get("performer_logPPL", float("nan"))),
            "cross_logXPPL": float(metrics.get("cross_logXPPL", float("nan"))),
            "transitions": int(metrics.get("transitions", 0)),
            "last_line": int(chunk_end_line),
            "chunk_char_start": int(chunk_start),
            "chunk_char_end": int(chunk_end),
        }
        suffix = analysis_stale_suffix()
        if bool(state.get("analysis_next_available")):
            suffix += " | Analyze Next available."
        status_var.set(status_core + suffix)

    def analysis_stale_suffix() -> str:
        if state.get("has_analysis") and state.get("b_score_stale"):
            return " | B score is stale for current edits/rewrites. Run Analyze for exact B."
        return ""

    def refresh_analysis_status_line() -> None:
        update_analysis_status_from_active_chunk()

    def cancel_pending_status_restore() -> None:
        pending_restore = state.get("pending_status_restore_job")
        if pending_restore is None:
            return
        try:
            root.after_cancel(pending_restore)
        except Exception:
            pass
        state["pending_status_restore_job"] = None

    def show_transient_status_then_restore_stats(message: str, duration_ms: int = 8000) -> None:
        """
        Show a temporary status message, then restore the current analysis metrics line.
        This is used for non-blocking workflow events (save, clear priors, delete, undo).
        """
        cancel_pending_status_restore()
        status_var.set(message)
        core = str(state.get("last_analysis_status_core", "")).strip()
        if not core:
            return

        expected = message

        def restore() -> None:
            state["pending_status_restore_job"] = None
            # Do not clobber newer status updates.
            if status_var.get() != expected:
                return
            refresh_analysis_status_line()

        state["pending_status_restore_job"] = root.after(max(100, int(duration_ms)), restore)

    def mark_analysis_stale() -> None:
        if not state.get("has_analysis"):
            return
        state["b_score_stale"] = True
        recompute_chunk_coverage_state(current_text())
        set_analyze_next_visible(bool(state.get("analysis_next_available")))
        refresh_analysis_status_line()

    def has_unsaved_changes() -> bool:
        saved_snapshot = str(state.get("last_saved_text", ""))
        return current_text() != saved_snapshot

    def sync_save_button_state(enabled_base: bool = True) -> None:
        # Save is enabled only when edits are allowed and content differs from last saved/opened snapshot.
        save_enabled = bool(enabled_base) and has_unsaved_changes()
        try:
            save_btn.configure(state="normal" if save_enabled else "disabled")
        except Exception:
            pass

    def sync_undo_button_state(enabled_base: bool = True) -> None:
        # Undo is single-level and only available when a tracked operation exists.
        undo_enabled = bool(state.get("undo_action")) and bool(enabled_base)
        try:
            undo_btn.configure(state="normal" if undo_enabled else "disabled")
        except Exception:
            pass

    def clear_one_level_undo() -> None:
        state["undo_action"] = None
        sync_undo_button_state(enabled_base=(not bool(state.get("analyzing"))))

    def remember_one_level_undo(
        operation_label: str,
        start_idx: str,
        old_text: str,
        new_text: str,
    ) -> None:
        """
        Capture one reversible text mutation.
        We intentionally keep one level only to keep behavior predictable and fast.
        """
        state["undo_action"] = {
            "label": str(operation_label),
            "start_idx": str(start_idx),
            "old_text": str(old_text),
            "new_text": str(new_text),
            "after_text": current_text(),
        }
        sync_undo_button_state(enabled_base=(not bool(state.get("analyzing"))))

    def on_undo() -> None:
        """
        Restore the most recent tracked mutation if the document has not changed since.
        Guarding on full-text equality prevents applying stale coordinates to a changed doc.
        """
        action_obj = state.get("undo_action")
        action = action_obj if isinstance(action_obj, dict) else None
        if not action:
            show_transient_status_then_restore_stats(
                f"Undo unavailable: no tracked operation.{analysis_stale_suffix()}",
                duration_ms=8000,
            )
            return
        if state.get("analyzing") or state.get("rewrite_busy"):
            return

        current = current_text()
        expected_after = str(action.get("after_text", ""))
        if current != expected_after:
            clear_one_level_undo()
            show_transient_status_then_restore_stats(
                f"Undo unavailable: document changed since last tracked operation.{analysis_stale_suffix()}",
                duration_ms=8000,
            )
            return

        start_idx = str(action.get("start_idx", "1.0"))
        old_text = str(action.get("old_text", ""))
        new_text = str(action.get("new_text", ""))
        label = str(action.get("label", "operation"))
        try:
            end_idx = text_widget.index(f"{start_idx}+{len(new_text)}c")
            existing = text_widget.get(start_idx, end_idx)
            if existing != new_text:
                clear_one_level_undo()
                show_transient_status_then_restore_stats(
                    f"Undo unavailable: tracked range no longer matches.{analysis_stale_suffix()}",
                    duration_ms=8000,
                )
                return

            text_widget.focus_set()
            text_widget.mark_set("insert", start_idx)
            text_widget.delete(start_idx, end_idx)
            if old_text:
                text_widget.insert(start_idx, old_text)
            restored_end = text_widget.index(f"{start_idx}+{len(old_text)}c")
            if old_text:
                text_widget.tag_add("edited", start_idx, restored_end)
            text_widget.tag_raise("edited")
            text_widget.tag_raise("misspelled")
            queue_line_numbers_refresh(delay_ms=0)
            queue_line_bars_refresh(delay_ms=0)
            state["spell_version"] = int(state.get("spell_version", 0)) + 1
            queue_spellcheck(delay_ms=0)
            queue_preview_render(delay_ms=0)
            queue_preview_focus_sync(delay_ms=0)
            mark_analysis_stale()
            clear_one_level_undo()
            show_transient_status_then_restore_stats(
                f"Undo applied: {label}.{analysis_stale_suffix()}",
                duration_ms=1800,
            )
        except Exception as exc:
            clear_one_level_undo()
            show_transient_status_then_restore_stats(
                f"Undo failed: {exc}{analysis_stale_suffix()}",
                duration_ms=8000,
            )

    def cancel_pending_synonym_lookup() -> None:
        pending_syn = state.get("pending_synonym_job")
        if pending_syn is None:
            return
        try:
            root.after_cancel(pending_syn)
        except Exception:
            pass
        state["pending_synonym_job"] = None

    def clear_synonym_panel(message: str = "Synonyms: click a word in the left pane.") -> None:
        synonym_header_var.set(message)
        for i in range(SYNONYM_OPTION_COUNT):
            synonym_item_vars[i].set("")
        for btn in synonym_buttons:
            try:
                btn.configure(state="disabled")
            except Exception:
                pass
        state["synonym_options"] = []
        state["synonym_target_word"] = ""
        state["synonym_target_start_idx"] = None
        state["synonym_target_end_idx"] = None

    def apply_synonym_choice(choice_idx: int, _event: Optional[Any] = None) -> str:
        options_obj = state.get("synonym_options")
        options = options_obj if isinstance(options_obj, list) else []
        if choice_idx < 0 or choice_idx >= len(options):
            return "break"

        start_idx = state.get("synonym_target_start_idx")
        end_idx = state.get("synonym_target_end_idx")
        target_word = str(state.get("synonym_target_word", "") or "")
        if not start_idx or not end_idx or not target_word:
            return "break"

        replacement = str(options[choice_idx] or "").strip()
        if not replacement:
            return "break"

        try:
            current_span = text_widget.get(str(start_idx), str(end_idx))
        except Exception:
            show_transient_status_then_restore_stats(
                "Synonym target changed. Click the word again.",
                duration_ms=5000,
            )
            return "break"
        if _normalize_synonym_candidate(current_span) != _normalize_synonym_candidate(target_word):
            show_transient_status_then_restore_stats(
                "Synonym target changed. Click the word again.",
                duration_ms=5000,
            )
            return "break"

        text_widget.focus_set()
        text_widget.mark_set("insert", str(start_idx))
        text_widget.delete(str(start_idx), str(end_idx))
        text_widget.insert(str(start_idx), replacement)
        new_end = text_widget.index(f"{start_idx}+{len(replacement)}c")
        text_widget.tag_add("edited", str(start_idx), new_end)
        text_widget.tag_raise("edited")
        text_widget.tag_raise("misspelled")
        queue_line_numbers_refresh(delay_ms=0)
        queue_line_bars_refresh(delay_ms=0)
        state["spell_version"] = int(state.get("spell_version", 0)) + 1
        queue_spellcheck(delay_ms=0)
        queue_preview_render(delay_ms=0)
        queue_preview_focus_sync(delay_ms=0)
        remember_one_level_undo(
            operation_label="synonym replacement",
            start_idx=str(start_idx),
            old_text=current_span,
            new_text=replacement,
        )
        mark_analysis_stale()
        show_transient_status_then_restore_stats(
            f"Applied synonym [{choice_idx + 1}] '{replacement}'.{analysis_stale_suffix()}",
            duration_ms=8000,
        )
        return "break"

    def update_synonym_panel_for_word(
        request_id: int,
        clicked_word: str,
        start_idx: str,
        end_idx: str,
        synonyms: List[str],
    ) -> None:
        if int(state.get("synonym_request_id", 0)) != int(request_id):
            return

        pattern = detect_inflection_pattern(clicked_word)
        cased: List[str] = []
        seen: Set[str] = set()
        for syn in synonyms:
            base = _normalize_synonym_candidate(str(syn))
            if not base:
                continue
            inflected = inflect_candidate_to_pattern(base, pattern)
            candidate = inflected if inflected else base

            # Validate transformed forms; prefer base if transformed spelling looks invalid.
            if candidate != base:
                transformed_ok = is_word_spelled_correctly(candidate, english_words)
                if not transformed_ok:
                    candidate = base

            out_word = apply_word_case_from_template(clicked_word, candidate)
            norm_out = _normalize_synonym_candidate(out_word)
            if not norm_out or norm_out in seen:
                continue
            seen.add(norm_out)
            cased.append(out_word)
            if len(cased) >= SYNONYM_OPTION_COUNT:
                break

        state["synonym_options"] = list(cased[:SYNONYM_OPTION_COUNT])
        state["synonym_target_word"] = clicked_word
        state["synonym_target_start_idx"] = start_idx
        state["synonym_target_end_idx"] = end_idx

        if not cased:
            synonym_header_var.set(f"No synonyms found for '{clicked_word}'.")
            for i in range(SYNONYM_OPTION_COUNT):
                synonym_item_vars[i].set("")
            for btn in synonym_buttons:
                btn.configure(state="disabled")
            return

        synonym_header_var.set(f"Synonyms for '{clicked_word}':")
        for i in range(SYNONYM_OPTION_COUNT):
            if i < len(cased):
                synonym_item_vars[i].set(f"[{i + 1}] {cased[i]}")
            else:
                synonym_item_vars[i].set("")
        for i, btn in enumerate(synonym_buttons):
            btn.configure(state="normal" if i < len(cased) else "disabled")

    def queue_synonym_lookup_for_click(index_clicked: str) -> None:
        # Debounce avoids accidental lookups while drag-selection is still in motion.
        cancel_pending_synonym_lookup()
        req_id = int(state.get("synonym_request_id", 0)) + 1
        state["synonym_request_id"] = req_id

        def launch() -> None:
            state["pending_synonym_job"] = None
            if state.get("internal_update"):
                return
            if bool(state.get("rewrite_busy")) or bool(state.get("analyzing")):
                return

            sel_pair = selection_index_pair()
            if sel_pair is not None:
                s_idx, e_idx = sel_pair
                if char_offset_for_index(e_idx) > char_offset_for_index(s_idx):
                    # User is selecting a region; skip synonym lookup to avoid jitter.
                    return

            text_snapshot = current_text()
            if not text_snapshot:
                clear_synonym_panel()
                return
            char_pos = char_offset_for_index(index_clicked)
            span = extract_click_word_span(text_snapshot, char_pos)
            if span is None:
                return
            start_char, end_char, word = span
            if end_char <= start_char:
                return

            norm_word = _normalize_synonym_candidate(word)
            if not norm_word:
                return

            start_idx = text_widget.index(f"1.0+{start_char}c")
            end_idx = text_widget.index(f"1.0+{end_char}c")
            synonym_header_var.set(f"Looking up synonyms for '{word}'...")
            for i in range(SYNONYM_OPTION_COUNT):
                synonym_item_vars[i].set("")
            for btn in synonym_buttons:
                btn.configure(state="disabled")

            cache_obj = state.get("synonym_cache")
            cache = cache_obj if isinstance(cache_obj, dict) else {}
            cached = cache.get(norm_word)
            if isinstance(cached, list) and cached:
                update_synonym_panel_for_word(req_id, word, start_idx, end_idx, list(cached))
                return

            def worker() -> None:
                found = lookup_synonyms_fast(norm_word, max_items=SYNONYM_OPTION_COUNT)
                if isinstance(cache_obj, dict):
                    cache_obj[norm_word] = list(found)

                def on_ready() -> None:
                    update_synonym_panel_for_word(req_id, word, start_idx, end_idx, list(found))

                root.after(0, on_ready)

            threading.Thread(target=worker, daemon=True).start()

        # Small debounce to avoid firing while drag-selection is in progress.
        state["pending_synonym_job"] = root.after(230, launch)

    def warmup_synonym_lookup_async() -> None:
        """
        Prime synonym providers (local/WordNet/Datamuse path) in the background so the
        first user-triggered synonym click feels fast.
        """
        def worker() -> None:
            try:
                warm_word = "text"
                found = lookup_synonyms_fast(warm_word, max_items=SYNONYM_OPTION_COUNT)
                cache_obj = state.get("synonym_cache")
                if isinstance(cache_obj, dict):
                    cache_obj[warm_word] = list(found)
            except Exception:
                # Warmup is opportunistic; never fail the GUI for this.
                pass

        threading.Thread(target=worker, daemon=True).start()

    def refresh_line_numbers_now() -> None:
        state["pending_line_numbers_job"] = None
        try:
            line_count = max(1, int(text_widget.index("end-1c").split(".")[0]))
        except Exception:
            line_count = 1

        try:
            line_numbers.delete("all")
        except Exception:
            return

        if int(state.get("line_count", 0)) != line_count:
            digits = max(2, len(str(line_count)))
            try:
                char_w = int(line_number_font.measure("0"))
            except Exception:
                char_w = 8
            gutter_px = max(48, (digits * char_w) + 14)
            try:
                line_numbers.configure(width=gutter_px)
            except Exception:
                pass
            state["line_count"] = line_count

        try:
            canvas_h = max(1, int(line_numbers.winfo_height()))
            canvas_w = max(24, int(line_numbers.winfo_width()))
        except Exception:
            canvas_h = 1
            canvas_w = 56
        x = canvas_w - 6

        try:
            line_idx = text_widget.index("@0,0")
        except Exception:
            queue_line_bars_refresh(delay_ms=0)
            return

        while True:
            dline = text_widget.dlineinfo(line_idx)
            if dline is None:
                break
            y = float(dline[1])
            h = float(dline[3])
            if y > canvas_h:
                break
            try:
                line_no = int(str(line_idx).split(".")[0])
            except Exception:
                line_no = 1
            line_numbers.create_text(
                x,
                y + (h / 2.0),
                text=str(line_no),
                fill="#7d7d7d",
                font=line_number_font,
                anchor="e",
            )
            line_idx = text_widget.index(f"{line_idx}+1line")

        queue_line_bars_refresh(delay_ms=0)

    def queue_line_numbers_refresh(delay_ms: int = 80) -> None:
        pending = state.get("pending_line_numbers_job")
        if pending is not None:
            try:
                root.after_cancel(pending)
            except Exception:
                pass
        state["pending_line_numbers_job"] = root.after(delay_ms, refresh_line_numbers_now)

    def line_number_for_char_end(text: str, analyzed_char_end: int) -> int:
        if not text:
            return 1
        total_lines = text.count("\n") + 1
        if analyzed_char_end >= len(text):
            return total_lines
        if analyzed_char_end <= 0:
            return 1
        last_idx = min(len(text) - 1, analyzed_char_end - 1)
        return text.count("\n", 0, last_idx + 1) + 1

    def refresh_line_bars_now() -> None:
        state["pending_line_bars_job"] = None
        try:
            line_bars.delete("all")
        except Exception:
            return

        # Build line spans for unscored intervals so the gutter can clearly
        # indicate analysis coverage boundaries.
        text_snapshot = current_text()
        unscored_line_spans: List[Tuple[int, int]] = []
        for s, e in get_unscored_intervals(text_snapshot):
            if e <= s:
                continue
            try:
                start_line = int(str(text_widget.index(f"1.0+{s}c")).split(".")[0])
                end_line = int(str(text_widget.index(f"1.0+{max(s, e - 1)}c")).split(".")[0])
            except Exception:
                continue
            if end_line < start_line:
                end_line = start_line
            unscored_line_spans.append((start_line, end_line))
        unscored_line_spans.sort(key=lambda pair: (pair[0], pair[1]))

        current_map_obj = state.get("line_contrib_map")
        current_map = current_map_obj if isinstance(current_map_obj, dict) else {}
        try:
            current_max_abs = float(state.get("line_contrib_max_abs", 0.0))
        except Exception:
            current_max_abs = 0.0
        if not np.isfinite(current_max_abs):
            current_max_abs = 0.0

        prior_entries_obj = state.get("prior_line_contrib_maps")
        prior_entries_raw = prior_entries_obj if isinstance(prior_entries_obj, list) else []
        prior_entries: List[Tuple[Dict[int, float], float]] = []
        for entry in prior_entries_raw:
            if not isinstance(entry, dict):
                continue
            map_obj = entry.get("map")
            if not isinstance(map_obj, dict) or not map_obj:
                continue
            try:
                max_abs = float(entry.get("max_abs", 0.0))
            except Exception:
                max_abs = 0.0
            if not np.isfinite(max_abs) or max_abs <= 0.0:
                continue
            prior_entries.append((map_obj, max_abs))

        if (not current_map or current_max_abs <= 0.0) and not prior_entries and not unscored_line_spans:
            return

        try:
            canvas_w = max(8, int(line_bars.winfo_width()))
            canvas_h = max(1, int(line_bars.winfo_height()))
        except Exception:
            return
        max_bar_w = max(6, canvas_w - 4)

        try:
            line_idx = text_widget.index("@0,0")
        except Exception:
            return

        unscored_span_idx = 0
        while True:
            dline = text_widget.dlineinfo(line_idx)
            if dline is None:
                break
            y = float(dline[1])
            h = float(dline[3])
            if y > canvas_h:
                break
            try:
                line_no = int(str(line_idx).split(".")[0])
            except Exception:
                line_no = 1
            pad = max(1, int(h * 0.2))
            y0 = int(y + pad)
            y1 = int(y + h - pad)
            if y1 <= y0:
                y1 = y0 + 1

            # Draw a dedicated unscored marker band first for clarity.
            while (
                unscored_span_idx < len(unscored_line_spans)
                and line_no > unscored_line_spans[unscored_span_idx][1]
            ):
                unscored_span_idx += 1
            if (
                unscored_span_idx < len(unscored_line_spans)
                and unscored_line_spans[unscored_span_idx][0] <= line_no <= unscored_line_spans[unscored_span_idx][1]
            ):
                line_bars.create_rectangle(2, y0, canvas_w - 2, y1, fill="#3f4b5b", width=0)

            # Draw prior maps first as faint backgrounds.
            for prior_map, prior_max_abs in prior_entries:
                raw_prior = prior_map.get(line_no, 0.0)
                try:
                    prior_v = float(raw_prior)
                except Exception:
                    prior_v = 0.0
                if not np.isfinite(prior_v) or abs(prior_v) <= 0.0:
                    continue
                frac = min(1.0, abs(prior_v) / prior_max_abs)
                bar_w = max(1, int(round(frac * max_bar_w)))
                x1 = canvas_w - 2
                x0 = max(2, x1 - bar_w)
                color = "#6a3a3a" if prior_v > 0 else "#2d5d3f"
                line_bars.create_rectangle(x0, y0, x1, y1, fill=color, width=0)

            top_lines_obj = state.get("line_contrib_top_lines")
            top_lines = top_lines_obj if isinstance(top_lines_obj, set) else set()

            raw_v = current_map.get(line_no, 0.0)
            try:
                val = float(raw_v)
            except Exception:
                val = 0.0
            if np.isfinite(val) and abs(val) > 0.0 and current_max_abs > 0.0:
                frac = min(1.0, abs(val) / current_max_abs)
                if line_no not in top_lines:
                    # Keep non-highlighted lines informative but visually secondary.
                    frac *= 0.45
                bar_w = max(1, int(round(frac * max_bar_w)))
                x1 = canvas_w - 2
                x0 = max(2, x1 - bar_w)
                if line_no in top_lines:
                    color = "#cf4242" if val > 0 else "#35b05f"
                else:
                    color = "#8e4848" if val > 0 else "#3a7450"
                line_bars.create_rectangle(x0, y0, x1, y1, fill=color, width=0)

            line_idx = text_widget.index(f"{line_idx}+1line")

    def queue_line_bars_refresh(delay_ms: int = 60) -> None:
        pending = state.get("pending_line_bars_job")
        if pending is not None:
            try:
                root.after_cancel(pending)
            except Exception:
                pass
        state["pending_line_bars_job"] = root.after(delay_ms, refresh_line_bars_now)

    def _char_pos_to_line(line_starts: List[int], char_pos: int) -> int:
        if not line_starts:
            return 1
        clamped = max(0, int(char_pos))
        return max(1, bisect.bisect_right(line_starts, clamped))

    def rebuild_line_contribution_map(
        rows: List[Dict[str, Any]],
        annotations: Dict[int, Dict[str, Any]],
        analyzed_char_end: int,
    ) -> None:
        text = current_text()
        if not rows or not annotations:
            state["line_contrib_map"] = {}
            state["line_contrib_max_abs"] = 0.0
            state["line_contrib_last_line"] = 0
            state["line_contrib_top_lines"] = set()
            queue_line_bars_refresh(delay_ms=0)
            return

        line_starts: List[int] = [0]
        for i, ch in enumerate(text):
            if ch == "\n":
                line_starts.append(i + 1)

        contrib: Dict[int, float] = {}
        top_lines: Set[int] = set()
        text_len = len(text)
        try:
            limit = max(0, min(text_len, int(analyzed_char_end)))
        except Exception:
            limit = text_len
        if limit <= 0:
            limit = text_len
        last_line = line_number_for_char_end(text, limit)
        for row in rows:
            para_id = int(row.get("paragraph_id", -1))
            ann = annotations.get(para_id)
            label = str(ann.get("label", "")).upper() if ann is not None else ""
            try:
                if ann is not None:
                    delta = float(
                        ann.get("delta_doc_logPPL_if_removed", row.get("delta_doc_logPPL_if_removed", 0.0))
                    )
                else:
                    delta = float(row.get("delta_doc_logPPL_if_removed", 0.0))
            except Exception:
                continue
            if not np.isfinite(delta) or delta == 0.0:
                continue
            if label not in ("LOW", "HIGH"):
                label = "LOW" if delta > 0 else "HIGH"
            magnitude = abs(delta)
            signed = magnitude if label == "LOW" else (-magnitude if label == "HIGH" else delta)
            if ann is None:
                signed *= 0.40
            try:
                c_start = int(row.get("char_start", 0))
                c_end = int(row.get("char_end", c_start))
            except Exception:
                continue
            c_start = max(0, min(text_len, c_start))
            c_end = max(c_start, min(text_len, c_end))
            if c_end <= c_start:
                continue

            start_line = _char_pos_to_line(line_starts, c_start)
            end_line = _char_pos_to_line(line_starts, max(c_start, c_end - 1))
            if ann is not None:
                for ln in range(start_line, end_line + 1):
                    top_lines.add(ln)
            n_lines = max(1, end_line - start_line + 1)
            share = signed / float(n_lines)
            for ln in range(start_line, end_line + 1):
                contrib[ln] = float(contrib.get(ln, 0.0) + share)

        state["line_contrib_map"] = contrib
        state["line_contrib_max_abs"] = max((abs(v) for v in contrib.values()), default=0.0)
        state["line_contrib_last_line"] = int(last_line)
        state["line_contrib_top_lines"] = top_lines
        queue_line_bars_refresh(delay_ms=0)

    def hide_tooltip() -> None:
        tip = state.get("tooltip")
        if tip is not None:
            try:
                tip.destroy()
            except Exception:
                pass
            state["tooltip"] = None

    inline_pattern = re.compile(
        r"`[^`\n]+`|\*\*[^*\n]+\*\*|\*[^*\n]+\*|_[^_\n]+_|\[[^\]\n]+\]\([^)]+\)"
    )

    def preview_insert_inline(raw: str, base_tags: Tuple[str, ...] = ()) -> None:
        pos = 0
        for m in inline_pattern.finditer(raw):
            if m.start() > pos:
                preview_text.insert("end", raw[pos:m.start()], base_tags)
            token = m.group(0)
            tags = list(base_tags)
            if token.startswith("`") and token.endswith("`"):
                tags.append("md_code_inline")
                preview_text.insert("end", token[1:-1], tuple(tags))
            elif token.startswith("**") and token.endswith("**"):
                tags.append("md_bold")
                preview_text.insert("end", token[2:-2], tuple(tags))
            elif token.startswith("*") and token.endswith("*"):
                tags.append("md_italic")
                preview_text.insert("end", token[1:-1], tuple(tags))
            elif token.startswith("_") and token.endswith("_"):
                tags.append("md_italic")
                preview_text.insert("end", token[1:-1], tuple(tags))
            elif token.startswith("["):
                mm = re.match(r"\[([^\]]+)\]\(([^)]+)\)", token)
                if mm:
                    tags.append("md_link")
                    preview_text.insert("end", mm.group(1), tuple(tags))
                else:
                    preview_text.insert("end", token, base_tags)
            else:
                preview_text.insert("end", token, base_tags)
            pos = m.end()
        if pos < len(raw):
            preview_text.insert("end", raw[pos:], base_tags)

    def render_markdown_preview_now() -> None:
        state["pending_preview_job"] = None
        raw = _normalize_markdown_hardbreaks(current_text())

        preview_text.configure(state="normal")
        preview_text.delete("1.0", "end")

        in_code_block = False
        line_map: Dict[int, int] = {}
        for src_line_num, line in enumerate(raw.split("\n"), start=1):
            try:
                line_map[src_line_num] = max(1, int(preview_text.index("end-1c").split(".")[0]))
            except Exception:
                line_map[src_line_num] = 1
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue

            if in_code_block:
                preview_text.insert("end", line + "\n", ("md_code_block",))
                continue

            if stripped == "":
                preview_text.insert("end", "\n")
                continue

            hm = re.match(r"^(#{1,6})\s+(.*)$", line)
            if hm:
                level = min(len(hm.group(1)), 3)
                preview_insert_inline(hm.group(2), (f"md_h{level}",))
                preview_text.insert("end", "\n\n")
                continue

            if re.match(r"^\s*([-*_])\1{2,}\s*$", line):
                preview_text.insert("end", "─" * 34 + "\n\n", ("md_hr",))
                continue

            qm = re.match(r"^\s*>\s?(.*)$", line)
            if qm:
                preview_text.insert("end", "│ ", ("md_quote",))
                preview_insert_inline(qm.group(1), ("md_quote",))
                preview_text.insert("end", "\n")
                continue

            ulm = re.match(r"^(\s*)[-*+]\s+(.*)$", line)
            if ulm:
                indent = " " * len(ulm.group(1))
                preview_text.insert("end", f"{indent}• ", ("md_list",))
                preview_insert_inline(ulm.group(2), ("md_list",))
                preview_text.insert("end", "\n")
                continue

            olm = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)
            if olm:
                indent = " " * len(olm.group(1))
                preview_text.insert("end", f"{indent}{olm.group(2)}. ", ("md_list",))
                preview_insert_inline(olm.group(3), ("md_list",))
                preview_text.insert("end", "\n")
                continue

            preview_insert_inline(line)
            preview_text.insert("end", "\n")

        state["preview_line_map"] = line_map
        preview_text.configure(state="disabled")
        apply_preview_segment_backgrounds()
        sync_preview_focus_now()

    def queue_preview_render(delay_ms: int) -> None:
        pending = state.get("pending_preview_job")
        if pending is not None:
            try:
                root.after_cancel(pending)
            except Exception:
                pass
        state["pending_preview_job"] = root.after(delay_ms, render_markdown_preview_now)

    def set_debug_text(message: str) -> None:
        if state.get("debug_enabled", False):
            debug_var.set(message)
        else:
            debug_var.set("")

    def toggle_debug_overlay(_event: Any = None) -> str:
        enabled = not bool(state.get("debug_enabled", False))
        state["debug_enabled"] = enabled
        if enabled:
            debug_label.pack(side="right", padx=(8, 0))
            set_debug_text("Debug overlay enabled")
        else:
            debug_label.pack_forget()
            set_debug_text("")
        return "break"

    def source_line_to_preview_line(src_line: int) -> int:
        map_obj = state.get("preview_line_map")
        preview_line_map = map_obj if isinstance(map_obj, dict) else {}
        if not preview_line_map:
            try:
                total_src = max(1, int(text_widget.index("end-1c").split(".")[0]))
            except Exception:
                total_src = 1
            try:
                total_preview = max(1, int(preview_text.index("end-1c").split(".")[0]))
            except Exception:
                total_preview = 1
            frac = (max(1, int(src_line)) - 1) / float(max(total_src - 1, 1))
            return int(round(frac * max(total_preview - 1, 0))) + 1

        line = max(1, int(src_line))
        mapped = preview_line_map.get(line)
        if mapped is not None:
            try:
                return max(1, int(mapped))
            except Exception:
                pass

        nearest = max((k for k in preview_line_map.keys() if k <= line), default=1)
        try:
            base = int(preview_line_map.get(nearest, 1))
        except Exception:
            base = 1
        return max(1, base)

    def preview_view_offset_lines() -> int:
        try:
            return int(state.get("preview_view_offset_lines", -3))
        except Exception:
            return -3

    def apply_preview_view_offset() -> None:
        offset = preview_view_offset_lines()
        if offset == 0:
            return
        try:
            # GUI calibration: apply the configured offset in the opposite direction.
            preview_text.yview_scroll(-offset, "units")
        except Exception:
            pass

    def set_preview_active_line_for_src_line(src_line: int) -> None:
        if selected_source_line_range() is not None:
            preview_text.tag_remove("preview_active_line", "1.0", "end")
            return
        preview_line = source_line_to_preview_line(src_line)
        try:
            preview_total = max(1, int(preview_text.index("end-1c").split(".")[0]))
        except Exception:
            preview_total = 1
        preview_line = max(1, min(preview_total, preview_line))
        preview_text.tag_remove("preview_active_line", "1.0", "end")
        label = line_heat_label(src_line)
        if label == "LOW":
            preview_text.tag_configure("preview_active_line", background="#4a3434")
        elif label == "HIGH":
            preview_text.tag_configure("preview_active_line", background="#31483a")
        else:
            preview_text.tag_configure("preview_active_line", background="#606060")
        preview_text.tag_add(
            "preview_active_line",
            f"{preview_line}.0",
            f"{preview_line}.0 lineend+1c",
        )
        preview_text.tag_raise("preview_heat_low")
        preview_text.tag_raise("preview_heat_high")
        preview_text.tag_raise("preview_sel_low")
        preview_text.tag_raise("preview_sel_high")
        preview_text.tag_raise("preview_sel_neutral")
        preview_text.tag_raise("preview_active_line")

    def clear_preview_segment_backgrounds() -> None:
        preview_text.tag_remove("preview_heat_low", "1.0", "end")
        preview_text.tag_remove("preview_heat_high", "1.0", "end")
        preview_text.tag_remove("preview_sel_low", "1.0", "end")
        preview_text.tag_remove("preview_sel_high", "1.0", "end")
        preview_text.tag_remove("preview_sel_neutral", "1.0", "end")
        preview_text.tag_remove("preview_unscored", "1.0", "end")

    def apply_preview_segment_backgrounds() -> None:
        clear_preview_segment_backgrounds()
        map_obj = state.get("preview_line_map")
        preview_line_map = map_obj if isinstance(map_obj, dict) else {}
        if not preview_line_map:
            return

        src_text = current_text()
        for unscored_start, unscored_end in get_unscored_intervals(src_text):
            if unscored_end <= unscored_start:
                continue
            try:
                start_line = int(str(text_widget.index(f"1.0+{unscored_start}c")).split(".")[0])
                end_line = int(str(text_widget.index(f"1.0+{max(unscored_start, unscored_end - 1)}c")).split(".")[0])
            except Exception:
                continue
            if end_line < start_line:
                end_line = start_line
            for src_line in range(start_line, end_line + 1):
                mapped = preview_line_map.get(src_line)
                if mapped is None:
                    mapped = source_line_to_preview_line(src_line)
                try:
                    preview_line = int(mapped)
                except Exception:
                    continue
                preview_text.tag_add(
                    "preview_unscored",
                    f"{preview_line}.0",
                    f"{preview_line}.0 lineend+1c",
                )

        sel_range = selected_source_line_range()
        if sel_range is not None:
            sel_start_line, sel_end_line = sel_range
            for src_line in range(sel_start_line, sel_end_line + 1):
                mapped = preview_line_map.get(src_line)
                if mapped is None:
                    mapped = source_line_to_preview_line(src_line)
                try:
                    preview_line = int(mapped)
                except Exception:
                    continue
                label = line_heat_label(src_line)
                if label == "LOW":
                    preview_tag = "preview_sel_low"
                elif label == "HIGH":
                    preview_tag = "preview_sel_high"
                else:
                    preview_tag = "preview_sel_neutral"
                preview_text.tag_add(
                    preview_tag,
                    f"{preview_line}.0",
                    f"{preview_line}.0 lineend+1c",
                )
            preview_text.tag_lower("preview_sel_low")
            preview_text.tag_lower("preview_sel_high")
            preview_text.tag_lower("preview_sel_neutral")
            preview_text.tag_raise("preview_unscored")
            return

        for tag in state.get("segment_tags", []):
            label = state.get("segment_labels", {}).get(tag, "")
            if label == "LOW":
                preview_tag = "preview_heat_low"
            elif label == "HIGH":
                preview_tag = "preview_heat_high"
            else:
                continue

            ranges = text_widget.tag_ranges(tag)
            for i in range(0, len(ranges), 2):
                r_start = ranges[i]
                r_end = ranges[i + 1]
                try:
                    start_line = int(str(r_start).split(".")[0])
                    end_line = int(str(text_widget.index(f"{r_end}-1c")).split(".")[0])
                except Exception:
                    continue
                if end_line < start_line:
                    continue
                for src_line in range(start_line, end_line + 1):
                    mapped = preview_line_map.get(src_line)
                    if mapped is None:
                        continue
                    try:
                        preview_line = int(mapped)
                    except Exception:
                        continue
                    preview_text.tag_add(
                        preview_tag,
                        f"{preview_line}.0",
                        f"{preview_line}.0 lineend+1c",
                    )
        preview_text.tag_lower("preview_heat_low")
        preview_text.tag_lower("preview_heat_high")
        preview_text.tag_raise("preview_unscored")

    def line_heat_label(src_line: int) -> Optional[str]:
        line_start = f"{src_line}.0"
        line_end = f"{src_line}.0 lineend+1c"
        saw_high = False
        for tag in state.get("segment_tags", []):
            label = state.get("segment_labels", {}).get(tag)
            if label not in ("LOW", "HIGH"):
                continue
            ranges = text_widget.tag_ranges(tag)
            for i in range(0, len(ranges), 2):
                r_start = ranges[i]
                r_end = ranges[i + 1]
                try:
                    overlaps = text_widget.compare(r_end, ">", line_start) and text_widget.compare(
                        r_start, "<", line_end
                    )
                except Exception:
                    overlaps = False
                if overlaps:
                    if label == "LOW":
                        return "LOW"
                    saw_high = True
        return "HIGH" if saw_high else None

    def selected_source_line_range() -> Optional[Tuple[int, int]]:
        ranges = text_widget.tag_ranges("sel")
        if len(ranges) < 2:
            return None
        start_idx = str(ranges[0])
        end_idx = str(ranges[1])
        try:
            if not text_widget.compare(end_idx, ">", start_idx):
                return None
        except Exception:
            return None
        try:
            start_line = int(start_idx.split(".")[0])
        except Exception:
            start_line = 1
        try:
            end_minus_idx = text_widget.index(f"{end_idx}-1c")
            end_line = int(str(end_minus_idx).split(".")[0])
        except Exception:
            end_line = start_line
        if end_line < start_line:
            end_line = start_line
        return (start_line, end_line)

    def sync_preview_focus_now() -> None:
        state["pending_focus_job"] = None
        apply_preview_segment_backgrounds()

        try:
            src_line = int(text_widget.index("insert").split(".")[0])
        except Exception:
            src_line = 1
        set_preview_active_line_for_src_line(src_line)
        preview_line = source_line_to_preview_line(src_line)
        preview_idx = f"{preview_line}.0"

        left_info = text_widget.dlineinfo("insert")
        if left_info is not None and text_widget.winfo_height() > 1:
            left_ratio = float(left_info[1]) / float(max(text_widget.winfo_height() - 1, 1))
        else:
            left_ratio = 0.35
        left_ratio = max(0.0, min(0.95, left_ratio))

        info = preview_text.dlineinfo(preview_idx)
        if info is None:
            try:
                preview_text.see(preview_idx)
            except Exception:
                pass
            info = preview_text.dlineinfo(preview_idx)

        top, bottom = preview_text.yview()
        visible = bottom - top
        if visible <= 0.0:
            visible = 0.25
        if info is not None and preview_text.winfo_height() > 1:
            current_ratio = float(info[1]) / float(max(preview_text.winfo_height() - 1, 1))
            delta = current_ratio - left_ratio
            if abs(delta) > 0.02:
                top_target = top + (delta * visible)
                max_top = max(0.0, 1.0 - visible)
                top_target = max(0.0, min(max_top, top_target))
                preview_text.yview_moveto(top_target)
                apply_preview_view_offset()
        else:
            try:
                preview_text.see(preview_idx)
                apply_preview_view_offset()
            except Exception:
                pass

        info2 = preview_text.dlineinfo(preview_idx)
        if info2 is not None and preview_text.winfo_height() > 1:
            preview_ratio = float(info2[1]) / float(max(preview_text.winfo_height() - 1, 1))
        else:
            preview_ratio = float("nan")
        set_debug_text(
            f"mode=focus src_insert={src_line} prev_insert={preview_line} offset={preview_view_offset_lines()} "
            f"left_ratio={left_ratio:.3f} preview_ratio={preview_ratio:.3f} "
            f"left_top={text_widget.yview()[0]:.3f} preview_top={preview_text.yview()[0]:.3f}"
        )
        if state.get("has_analysis") and not state.get("analyzing") and state.get("pending_status_restore_job") is None:
            refresh_analysis_status_line()

    def queue_preview_focus_sync(delay_ms: int = 30) -> None:
        pending = state.get("pending_focus_job")
        if pending is not None:
            try:
                root.after_cancel(pending)
            except Exception:
                pass
        state["pending_focus_job"] = root.after(delay_ms, sync_preview_focus_now)

    def sync_preview_scroll_lock(left_first: Optional[float] = None) -> None:
        if left_first is None:
            try:
                yview = text_widget.yview()
                left_first = float(yview[0]) if yview else 0.0
            except Exception:
                left_first = 0.0
        left_first = max(0.0, min(1.0, float(left_first)))
        try:
            src_top_idx = str(text_widget.index("@0,0"))
        except Exception:
            src_top_idx = "1.0"
        try:
            src_top_line, src_top_col = [int(x) for x in src_top_idx.split(".", 1)]
        except Exception:
            src_top_line, src_top_col = 1, 0

        preview_top_line = source_line_to_preview_line(src_top_line)
        preview_top_idx = f"{preview_top_line}.{max(0, src_top_col)}"
        try:
            preview_text.yview(preview_top_idx)
            apply_preview_view_offset()
        except Exception:
            preview_text.yview_moveto(left_first)
            apply_preview_view_offset()
        set_preview_active_line_for_src_line(max(1, int(text_widget.index("insert").split(".")[0])))
        set_debug_text(
            f"mode=scroll src_top={src_top_idx} prev_top={preview_top_idx} offset={preview_view_offset_lines()} "
            f"left_first={left_first:.3f} preview_first={preview_text.yview()[0]:.3f}"
        )
        if state.get("has_analysis") and not state.get("analyzing") and state.get("pending_status_restore_job") is None:
            refresh_analysis_status_line()

    def on_preview_scroll(*args: Any) -> None:
        if not args:
            return
        op = str(args[0])
        try:
            if op == "moveto" and len(args) >= 2:
                text_widget.yview_moveto(float(args[1]))
            elif op == "scroll" and len(args) >= 3:
                text_widget.yview_scroll(int(args[1]), str(args[2]))
        except Exception:
            return

    def on_preview_mousewheel(event: Any) -> str:
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return "break"
        steps = -1 if delta > 0 else 1
        try:
            text_widget.yview_scroll(steps, "units")
        except Exception:
            pass
        return "break"

    def on_preview_button4(_event: Any) -> str:
        try:
            text_widget.yview_scroll(-1, "units")
        except Exception:
            pass
        return "break"

    def on_preview_button5(_event: Any) -> str:
        try:
            text_widget.yview_scroll(1, "units")
        except Exception:
            pass
        return "break"

    def on_editor_scroll(first: str, last: str) -> None:
        scroll_y.set(first, last)
        queue_line_numbers_refresh(delay_ms=0)
        if text_widget.dlineinfo("insert") is not None:
            queue_preview_focus_sync(delay_ms=0)
        else:
            try:
                sync_preview_scroll_lock(float(first))
            except Exception:
                sync_preview_scroll_lock(None)

    def tooltip_text(info: Dict[str, Any]) -> str:
        pct = info.get("pct_contribution", float("nan"))
        pct_str = f"{pct:+.2f}%" if np.isfinite(pct) else "n/a"
        return (
            f"% contribution: {pct_str}\n"
            f"Paragraph: {info['paragraph_id']}\n"
            f"logPPL: {info['logPPL']:.6f}\n"
            f"delta_if_removed: {info['delta_doc_logPPL_if_removed']:+.6f}\n"
            f"delta_vs_doc: {info['delta_vs_doc_logPPL']:+.6f}\n"
            f"Transitions: {info['transitions']}\n"
            f"Chars: {info['char_start']}:{info['char_end']}\n"
            f"Tokens: {info['token_start']}:{info['token_end']}"
        )

    def show_tooltip(event: Any, info: Dict[str, Any]) -> None:
        hide_tooltip()
        tip = tk.Toplevel(root)
        tip.wm_overrideredirect(True)
        tip.attributes("-topmost", True)
        lbl = tk.Label(
            tip,
            text=tooltip_text(info),
            justify="left",
            bg="#202020",
            fg="#f2f2f2",
            bd=1,
            relief="solid",
            padx=8,
            pady=6,
        )
        lbl.pack()
        tip.geometry(f"+{event.x_root + 14}+{event.y_root + 12}")
        state["tooltip"] = tip

    def move_tooltip(event: Any) -> None:
        tip = state.get("tooltip")
        if tip is not None:
            tip.geometry(f"+{event.x_root + 14}+{event.y_root + 12}")

    def char_offset_for_index(index: str) -> int:
        try:
            raw = text_widget.count("1.0", index, "chars")
            if isinstance(raw, tuple) and raw:
                return int(raw[0])
        except Exception:
            pass
        return 0

    def tag_range_containing_index(tag: str, index: str) -> Optional[Tuple[str, str]]:
        ranges = text_widget.tag_ranges(tag)
        for i in range(0, len(ranges), 2):
            s = str(ranges[i])
            e = str(ranges[i + 1])
            try:
                if text_widget.compare(index, ">=", s) and text_widget.compare(index, "<", e):
                    return (s, e)
            except Exception:
                continue
        if len(ranges) >= 2:
            return (str(ranges[0]), str(ranges[1]))
        return None

    def close_rewrite_popup() -> None:
        popup = state.get("rewrite_popup")
        if popup is None:
            state["rewrite_popup"] = None
            state["rewrite_busy"] = False
            return
        try:
            popup.grab_release()
        except Exception:
            pass
        try:
            popup.destroy()
        except Exception:
            pass
        state["rewrite_popup"] = None
        state["rewrite_busy"] = False

    def selection_index_pair() -> Optional[Tuple[str, str]]:
        ranges = text_widget.tag_ranges("sel")
        if len(ranges) < 2:
            return None
        start_idx = str(ranges[0])
        end_idx = str(ranges[1])
        try:
            if text_widget.compare(end_idx, ">", start_idx):
                return (start_idx, end_idx)
        except Exception:
            return None
        return None

    def clamp_span_to_scored_range(
        start_char: int, end_char: int, text_snapshot: str
    ) -> Optional[Tuple[int, int, bool]]:
        text_len = len(text_snapshot)
        start = max(0, min(text_len, int(start_char)))
        end = max(start, min(text_len, int(end_char)))
        scored_intervals = get_scored_intervals(text_snapshot)
        if not scored_intervals:
            return None
        best: Optional[Tuple[int, int]] = None
        best_overlap = 0
        for s, e in scored_intervals:
            ov_start = max(start, s)
            ov_end = min(end, e)
            overlap = max(0, ov_end - ov_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best = (ov_start, ov_end)
        if best is None:
            return None
        clamped_start, clamped_end = best
        if clamped_end <= clamped_start:
            return None
        was_clamped = (clamped_start != start) or (clamped_end != end)
        return (clamped_start, clamped_end, was_clamped)

    def resolve_chunk_for_span(start_char: int, end_char: int, text_snapshot: str) -> Optional[Dict[str, Any]]:
        chunks = analysis_chunk_list()
        if not chunks:
            return None
        start = max(0, min(len(text_snapshot), int(start_char)))
        end = max(start, min(len(text_snapshot), int(end_char)))
        best_chunk = None
        best_overlap = 0
        for chunk in chunks:
            overlap = chunk_overlap_with_span(chunk, start, end)
            if overlap > best_overlap:
                best_chunk = chunk
                best_overlap = overlap
        if best_chunk is not None and best_overlap > 0:
            return best_chunk
        return resolve_active_chunk(text_snapshot)

    def expand_span_to_full_lines(
        start_char: int, end_char: int, text_snapshot: str
    ) -> Optional[Tuple[int, int]]:
        text_len = len(text_snapshot)
        start = max(0, min(text_len, int(start_char)))
        end = max(start, min(text_len, int(end_char)))
        if end <= start:
            return None
        # Extend to whole line boundaries. Include trailing newline for the last selected line when present.
        line_start = text_snapshot.rfind("\n", 0, start) + 1
        last_char = max(start, end - 1)
        nl_after = text_snapshot.find("\n", last_char)
        if nl_after < 0:
            line_end = text_len
        else:
            line_end = min(text_len, nl_after + 1)
        if line_end <= line_start:
            return None
        return (line_start, line_end)

    def collect_line_heat_labels_for_index_range(start_idx: str, end_idx: str) -> List[Optional[str]]:
        try:
            if not text_widget.compare(end_idx, ">", start_idx):
                return []
        except Exception:
            return []
        try:
            start_line = int(str(text_widget.index(start_idx)).split(".")[0])
        except Exception:
            start_line = 1
        try:
            end_minus_idx = text_widget.index(f"{end_idx}-1c")
            end_line = int(str(end_minus_idx).split(".")[0])
        except Exception:
            end_line = start_line
        if end_line < start_line:
            end_line = start_line
        out: List[Optional[str]] = []
        for ln in range(start_line, end_line + 1):
            out.append(line_heat_label(ln))
        return out

    def add_prior_background_from_line_labels(
        start_idx: str, end_idx: str, prior_line_labels: List[Optional[str]]
    ) -> None:
        if not prior_line_labels:
            return
        try:
            if not text_widget.compare(end_idx, ">", start_idx):
                return
        except Exception:
            return
        try:
            start_line = int(str(text_widget.index(start_idx)).split(".")[0])
        except Exception:
            start_line = 1
        try:
            end_minus_idx = text_widget.index(f"{end_idx}-1c")
            end_line = int(str(end_minus_idx).split(".")[0])
        except Exception:
            end_line = start_line
        if end_line < start_line:
            end_line = start_line

        new_line_count = max(1, end_line - start_line + 1)
        old_line_count = max(1, len(prior_line_labels))
        low_ranges: List[str] = []
        high_ranges: List[str] = []

        for i in range(new_line_count):
            src_i = min(old_line_count - 1, int((i * old_line_count) / new_line_count))
            label = prior_line_labels[src_i]
            line_no = start_line + i
            seg_start = start_idx if i == 0 else f"{line_no}.0"
            seg_end = end_idx if i == (new_line_count - 1) else f"{line_no + 1}.0"
            if label == "LOW":
                low_ranges.extend([seg_start, seg_end])
            elif label == "HIGH":
                high_ranges.extend([seg_start, seg_end])

        if low_ranges:
            add_prior_background_from_ranges(tuple(low_ranges), "#5a1f1f")
        if high_ranges:
            add_prior_background_from_ranges(tuple(high_ranges), "#203026")

    def reinject_missing_original_lines(original_text: str, rewrite_text: str) -> Tuple[str, int]:
        original = str(original_text or "")
        rewrite = str(rewrite_text or "")
        if not original:
            return rewrite, 0
        if not rewrite:
            return original, len(original.splitlines())

        original_lines = original.split("\n")
        rewrite_lines = rewrite.split("\n")
        if not original_lines:
            return rewrite, 0
        if not rewrite_lines:
            return original, len(original_lines)

        def is_accidental_short_replace(orig_line: str, rw_line: str) -> bool:
            o = str(orig_line or "").strip()
            r = str(rw_line or "").strip()
            if not o:
                return False
            if not r:
                return True
            o_words = len(re.findall(r"[A-Za-z0-9]+", o))
            r_words = len(re.findall(r"[A-Za-z0-9]+", r))
            if len(o) >= 60 and len(r) <= 14:
                return True
            if o_words >= 8 and r_words <= 2:
                return True
            if o.lower().startswith(r.lower()) and len(r) <= max(10, int(len(o) * 0.22)):
                return True
            return False

        matcher = difflib.SequenceMatcher(a=original_lines, b=rewrite_lines, autojunk=False)
        merged_lines: List[str] = []
        reinserted = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "delete":
                segment = original_lines[i1:i2]
                merged_lines.extend(segment)
                reinserted += len(segment)
            elif tag == "replace":
                orig_seg = original_lines[i1:i2]
                rw_seg = rewrite_lines[j1:j2]
                pair_n = min(len(orig_seg), len(rw_seg))
                for k in range(pair_n):
                    o_line = orig_seg[k]
                    r_line = rw_seg[k]
                    if is_accidental_short_replace(o_line, r_line):
                        merged_lines.append(o_line)
                        reinserted += 1
                    else:
                        merged_lines.append(r_line)
                # If replacement collapses multiple original lines, preserve unmapped original tail unchanged.
                if len(orig_seg) > pair_n:
                    tail = orig_seg[pair_n:]
                    merged_lines.extend(tail)
                    reinserted += len(tail)
                elif len(rw_seg) > pair_n:
                    merged_lines.extend(rw_seg[pair_n:])
            else:
                merged_lines.extend(rewrite_lines[j1:j2])

        merged = "\n".join(merged_lines)
        trailing_newlines = len(original) - len(original.rstrip("\n"))
        if trailing_newlines > 0:
            merged = merged.rstrip("\n") + ("\n" * trailing_newlines)
        return merged, reinserted

    def launch_rewrite_popup_for_span(
        start_idx: str,
        end_idx: str,
        start_char: int,
        end_char: int,
        text_snapshot: str,
        metrics: Dict[str, Any],
        span_label: str,
        range_note: Optional[str] = None,
        selection_mode: bool = False,
    ) -> str:
        close_rewrite_popup()
        hide_tooltip()
        llm_cfg_preview, _llm_cfg_preview_issue = load_optional_rewrite_llm_config()
        external_llm_expected = llm_cfg_preview is not None

        popup = None
        try:
            popup = tk.Toplevel(root)
            popup.title("Rewrite Options")
            popup.transient(root)
            popup.resizable(True, True)
            popup.configure(bg="#111111")
            width_scale = 1.25
            # Apply a reliable initial size immediately so window does not start tiny.
            base_w = int(round(900 * width_scale))
            base_h = 760 if selection_mode else 660
            min_w = int(round(780 * width_scale))
            # Keep enough vertical space so option buttons stay visible without manual resize.
            min_h = 760 if selection_mode else 620
            popup.geometry(f"{base_w}x{base_h}")
            popup.minsize(min_w, min_h)
            try:
                root.update_idletasks()
                root_w = max(900, int(root.winfo_width()))
                root_h = max(700, int(root.winfo_height()))
                popup_w = max(base_w, min(int(round(1180 * width_scale)), int(round(root_w * 0.85))))
                base_dyn_h = 600 if not selection_mode else 750
                max_dyn_h = 820 if not selection_mode else 1025
                popup_h = max(int(round(base_dyn_h)), min(int(round(max_dyn_h)), int(round(root_h * 0.88))))
                x = root.winfo_rootx() + max(0, (root_w - popup_w) // 2)
                y = root.winfo_rooty() + max(0, (root_h - popup_h) // 2)
                popup.geometry(f"{popup_w}x{popup_h}+{x}+{y}")
            except Exception:
                pass
        except Exception as exc:
            show_transient_status_then_restore_stats(
                f"Failed to open rewrite popup: {exc}{analysis_stale_suffix()}",
                duration_ms=8000,
            )
            return "break"
        state["rewrite_popup"] = popup
        state["rewrite_busy"] = True
        request_id = int(state.get("rewrite_request_id", 0)) + 1
        state["rewrite_request_id"] = request_id

        try:
            title_label = tk.Label(
                popup,
                text=f"Performing rewrite generation for {span_label}...",
                anchor="w",
                justify="left",
                bg="#111111",
                fg="#f0f0f0",
                padx=10,
                pady=8,
                font=("TkDefaultFont", 11, "bold"),
            )
            title_label.pack(side="top", fill="x")

            if not external_llm_expected:
                sub_label = tk.Label(
                    popup,
                    text=(
                        "Expected B impact is approximate. Full Analyze is required for exact B. "
                        "First run may take longer while models warm up."
                    ),
                    anchor="w",
                    justify="left",
                    bg="#111111",
                    fg="#c4c4c4",
                    padx=10,
                    pady=0,
                )
                sub_label.pack(side="top", fill="x", pady=(0, 4))

            if range_note and not external_llm_expected:
                range_note_label = tk.Label(
                    popup,
                    text=range_note,
                    anchor="w",
                    justify="left",
                    bg="#111111",
                    fg="#9cc7ff",
                    padx=10,
                    pady=0,
                )
                range_note_label.pack(side="top", fill="x", pady=(0, 6))

            status_label_var = tk.StringVar(
                value=(
                    "Using external LLM to generate rewrite options..."
                    if external_llm_expected
                    else "Working..."
                )
            )
            popup_status = tk.Label(
                popup,
                textvariable=status_label_var,
                anchor="w",
                justify="left",
                bg="#111111",
                fg="#d6d6d6",
                padx=10,
                pady=0,
            )
            popup_status.pack(side="top", fill="x", pady=(0, 8))

            wait_label_var: Optional[tk.StringVar] = None
            if not external_llm_expected:
                wait_label_var = tk.StringVar(
                    value="Performing rewrite generation and impact scoring (30-120s)."
                )
                wait_label = tk.Label(
                    popup,
                    textvariable=wait_label_var,
                    anchor="w",
                    justify="left",
                    bg="#111111",
                    fg="#9cc7ff",
                    padx=10,
                    pady=0,
                )
                wait_label.pack(side="top", fill="x", pady=(0, 10))

            button_frame = tk.Frame(popup, bg="#111111")
            button_frame.pack(side="bottom", fill="x", padx=10, pady=(0, 10))
            button_frame.configure(height=44)
            button_frame.pack_propagate(False)
            option_buttons: List[Any] = []

            options_frame = tk.Frame(popup, bg="#111111")
            options_frame.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 8))
            options_scroll = tk.Scrollbar(options_frame, orient="vertical")
            options_scroll.pack(side="right", fill="y")
            options_text = tk.Text(
                options_frame,
                wrap="word",
                height=10,
                width=100,
                yscrollcommand=options_scroll.set,
                bg="#151515",
                fg="#f0f0f0",
                relief="flat",
                borderwidth=0,
                highlightthickness=1,
                highlightbackground="#2a2a2a",
                insertbackground="#f0f0f0",
                font=("TkFixedFont", 11),
                padx=10,
                pady=8,
            )
            options_text.pack(side="left", fill="both", expand=True)
            options_scroll.configure(command=options_text.yview)
            if not external_llm_expected:
                options_text.insert(
                    "1.0",
                    "Preparing rewrite generation...\n\n"
                    "Stages:\n"
                    "1. Select rewrite model (external LLM if configured, else internal performer)\n"
                    "2. Generate 3 rewrite candidates\n"
                    "3. Score approximate B impact\n",
                )
            options_text.bind("<Key>", lambda _e: "break")

            popup_state: Dict[str, Any] = {
                "request_id": request_id,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "options": [],
            }
        except Exception as exc:
            show_transient_status_then_restore_stats(
                f"Rewrite popup render failed: {exc}{analysis_stale_suffix()}",
                duration_ms=8000,
            )
            close_rewrite_popup()
            return "break"

        def activate_popup_modal() -> None:
            if not popup_active():
                return
            try:
                popup.deiconify()
            except Exception:
                pass
            try:
                popup.lift()
            except Exception:
                pass
            try:
                popup.focus_force()
            except Exception:
                pass
            try:
                popup.grab_set()
            except Exception:
                pass

        wait_anim_job: Optional[Any] = None
        wait_anim_phase = {"n": 0}

        def queue_wait_anim() -> None:
            nonlocal wait_anim_job
            if not popup_active():
                return
            phase = int(wait_anim_phase["n"]) % 4
            dots = "." * phase
            if wait_label_var is not None:
                wait_label_var.set(
                    "Performing rewrite generation and impact scoring"
                    f"{dots} This may take 30-120s, and GPU can stay low during model load."
                )
            wait_anim_phase["n"] = phase + 1
            if wait_label_var is not None:
                wait_anim_job = root.after(500, queue_wait_anim)

        def stop_wait_anim() -> None:
            nonlocal wait_anim_job
            if wait_anim_job is not None:
                try:
                    root.after_cancel(wait_anim_job)
                except Exception:
                    pass
            wait_anim_job = None

        def popup_active() -> bool:
            return (
                state.get("rewrite_popup") is popup
                and int(state.get("rewrite_request_id", 0)) == int(popup_state["request_id"])
            )

        def on_popup_quit(_event: Any = None) -> str:
            if popup_active():
                stop_wait_anim()
                close_rewrite_popup()
                show_transient_status_then_restore_stats(
                    f"Rewrite selection canceled.{analysis_stale_suffix()}",
                    duration_ms=5000,
                )
            return "break"

        def apply_rewrite_choice(choice_idx: int, _event: Any = None) -> str:
            if not popup_active():
                return "break"
            options_obj = popup_state.get("options")
            options = options_obj if isinstance(options_obj, list) else []
            if choice_idx < 0 or choice_idx >= len(options):
                return "break"
            opt = options[choice_idx]
            rewrite_text_raw = str(opt.get("text", ""))
            if not rewrite_text_raw.strip():
                return "break"
            rewrite_text = rewrite_text_raw

            ins_start = str(popup_state["start_idx"])
            ins_end = str(popup_state["end_idx"])
            old_text = text_widget.get(ins_start, ins_end)
            prior_line_labels: List[Optional[str]] = []
            if selection_mode:
                prior_line_labels = collect_line_heat_labels_for_index_range(ins_start, ins_end)
            stop_wait_anim()
            close_rewrite_popup()

            text_widget.focus_set()
            text_widget.mark_set("insert", ins_start)
            text_widget.delete(ins_start, ins_end)
            text_widget.insert(ins_start, rewrite_text)
            new_end = text_widget.index(f"{ins_start}+{len(rewrite_text)}c")
            text_widget.tag_add("edited", ins_start, new_end)
            if selection_mode:
                add_prior_background_from_line_labels(ins_start, new_end, prior_line_labels)
            else:
                add_prior_background_from_ranges((ins_start, new_end), "#5a1f1f")
            text_widget.tag_raise("edited")
            text_widget.tag_raise("misspelled")
            queue_line_numbers_refresh(delay_ms=0)
            queue_line_bars_refresh(delay_ms=0)
            state["spell_version"] = int(state.get("spell_version", 0)) + 1
            queue_spellcheck(delay_ms=0)
            queue_preview_render(delay_ms=0)
            queue_preview_focus_sync(delay_ms=0)
            remember_one_level_undo(
                operation_label=("rewrite replacement (selection)" if selection_mode else "rewrite replacement"),
                start_idx=ins_start,
                old_text=old_text,
                new_text=rewrite_text,
            )
            mark_analysis_stale()

            approx_b = float(opt.get("approx_B", float("nan")))
            delta_b = float(opt.get("delta_B", float("nan")))
            if np.isfinite(approx_b) and np.isfinite(delta_b):
                show_transient_status_then_restore_stats(
                    f"Applied rewrite {choice_idx + 1}. Approx B: {approx_b:.6f} ({delta_b:+.6f}). "
                    "Full Analyze required for exact B.",
                    duration_ms=8000,
                )
            else:
                show_transient_status_then_restore_stats(
                    "Applied rewrite option. Full Analyze required for exact B.",
                    duration_ms=8000,
                )
            return "break"

        for idx in range(3):
            btn = tk.Button(
                button_frame,
                text=str(idx + 1),
                width=8,
                state="disabled",
                command=lambda i=idx: apply_rewrite_choice(i),
            )
            btn.pack(side="left", padx=(0, 8))
            option_buttons.append(btn)
        quit_btn_popup = tk.Button(button_frame, text="Quit", width=12, command=on_popup_quit)
        quit_btn_popup.pack(side="right")

        popup.bind("<KeyPress-1>", lambda e: apply_rewrite_choice(0, e))
        popup.bind("<KeyPress-2>", lambda e: apply_rewrite_choice(1, e))
        popup.bind("<KeyPress-3>", lambda e: apply_rewrite_choice(2, e))
        popup.bind("<KeyPress-q>", on_popup_quit)
        popup.bind("<KeyPress-Q>", on_popup_quit)
        popup.bind("<Escape>", on_popup_quit)
        popup.protocol("WM_DELETE_WINDOW", on_popup_quit)
        root.after_idle(activate_popup_modal)
        try:
            popup.update_idletasks()
        except Exception:
            pass

        status_var.set("Performing rewrite generation and impact scoring...")
        if external_llm_expected:
            status_label_var.set("Using external LLM to generate rewrite options...")
        else:
            status_label_var.set("Preparing rewrite pipeline...")
        queue_wait_anim()

        def post_popup_status(message: str) -> None:
            def apply() -> None:
                if not popup_active():
                    return
                status_label_var.set(message)

            root.after(0, apply)

        def post_popup_log(message: str) -> None:
            def apply() -> None:
                if not popup_active():
                    return
                options_text.insert("end", f"- {message}\n")
                options_text.see("end")

            root.after(0, apply)

        def worker() -> None:
            try:
                if not external_llm_expected:
                    post_popup_status("Selecting rewrite model...")
                    post_popup_log("Selecting rewrite model...")
                    if range_note:
                        post_popup_log(range_note)

                def rewrite_progress(message: str) -> None:
                    if external_llm_expected:
                        msg_low = str(message or "").lower()
                        if "fall" in msg_low or "unavailable" in msg_low or "failed" in msg_low:
                            post_popup_status(message)
                        return
                    post_popup_status(message)
                    post_popup_log(message)

                rewrites, rewrite_source, rewrite_note = generate_rewrite_candidates_for_span(
                    cfg_path=cfg_path,
                    full_text=text_snapshot,
                    span_start=start_char,
                    span_end=end_char,
                    option_count=3,
                    status_callback=rewrite_progress,
                )
                if selection_mode:
                    original_span = text_snapshot[start_char:end_char]
                    repaired: List[str] = []
                    repaired_lines = 0
                    for cand in rewrites:
                        fixed, count = reinject_missing_original_lines(original_span, cand)
                        repaired.append(fixed)
                        repaired_lines += max(0, int(count))
                    deduped: List[str] = []
                    seen_keys: Set[str] = set()
                    for cand in repaired:
                        key = _semantic_text_key(cand)
                        if key and key not in seen_keys:
                            seen_keys.add(key)
                            deduped.append(cand)
                    if deduped:
                        rewrites = deduped
                    while len(rewrites) < 3:
                        rewrites.append(rewrites[-1] if rewrites else original_span)
                    rewrites = rewrites[:3]
                    if repaired_lines > 0:
                        if not external_llm_expected:
                            post_popup_log(
                                f"Preserved {repaired_lines} unchanged source line(s) omitted by model output."
                            )
                if rewrite_source == "external-llm":
                    post_popup_status("Using external LLM for rewrite generation...")
                    if not external_llm_expected:
                        post_popup_log("Using external LLM for rewrite generation.")
                elif rewrite_source == "internal-fallback":
                    post_popup_status("External LLM unavailable; using internal performer model...")
                    if external_llm_expected:
                        post_popup_log("External LLM unavailable; switched to internal performer model.")
                    else:
                        post_popup_log("External LLM unavailable; used internal performer model.")
                else:
                    post_popup_status("Using internal performer model for rewrite generation...")
                    if not external_llm_expected:
                        post_popup_log("Using internal performer model for rewrite generation.")
                if rewrite_note and not external_llm_expected:
                    post_popup_log(rewrite_note)
                post_popup_status("Scoring expected impact for each rewrite option...")
                if not external_llm_expected:
                    post_popup_log("Scoring approximate B impact...")
                approx_full_text = text_snapshot
                approx_start_char = start_char
                approx_end_char = end_char
                try:
                    chunk_base_start = int(metrics.get("chunk_char_start", start_char))
                    chunk_base_end = int(metrics.get("chunk_char_end", end_char))
                except Exception:
                    chunk_base_start = start_char
                    chunk_base_end = end_char
                if chunk_base_end > chunk_base_start:
                    bounded_start = max(0, min(len(text_snapshot), chunk_base_start))
                    bounded_end = max(bounded_start, min(len(text_snapshot), chunk_base_end))
                    if bounded_end > bounded_start:
                        approx_full_text = text_snapshot[bounded_start:bounded_end]
                        approx_start_char = max(0, min(len(approx_full_text), start_char - bounded_start))
                        approx_end_char = max(
                            approx_start_char,
                            min(len(approx_full_text), end_char - bounded_start),
                        )
                options = estimate_rewrite_b_impact_options(
                    cfg_path=cfg_path,
                    full_text=approx_full_text,
                    span_start=approx_start_char,
                    span_end=approx_end_char,
                    rewrites=rewrites,
                    base_doc_b=float(metrics.get("binoculars_score", float("nan"))),
                    base_doc_observer_logppl=float(metrics.get("observer_logPPL", float("nan"))),
                    base_doc_cross_logxppl=float(metrics.get("cross_logXPPL", float("nan"))),
                    base_doc_transitions=int(metrics.get("transitions", 0)),
                    text_max_tokens_override=text_max_tokens_override,
                )
                if not external_llm_expected:
                    post_popup_log("Sorted options by expected increase in B (more human-like first).")
            except Exception as exc:
                def on_error(msg: str = str(exc)) -> None:
                    if not popup_active():
                        return
                    stop_wait_anim()
                    show_transient_status_then_restore_stats(
                        f"Rewrite generation failed: {msg}{analysis_stale_suffix()}",
                        duration_ms=8000,
                    )
                    close_rewrite_popup()

                root.after(0, on_error)
                return

            def on_ready() -> None:
                if not popup_active():
                    return
                stop_wait_anim()
                state["rewrite_busy"] = False
                popup_state["options"] = list(options)

                options_text.delete("1.0", "end")
                for i, opt in enumerate(options, start=1):
                    approx_b = float(opt.get("approx_B", float("nan")))
                    delta_b = float(opt.get("delta_B", float("nan")))
                    if np.isfinite(approx_b) and np.isfinite(delta_b):
                        score_line = f"approx B: {approx_b:.6f} ({delta_b:+.6f})"
                    else:
                        score_line = "approx B: n/a"
                    options_text.insert("end", f"[{i}] {score_line}\n")
                    options_text.insert("end", f"{opt.get('text', '')}\n\n")

                for idx, btn in enumerate(option_buttons):
                    btn.configure(state="normal" if idx < len(options) else "disabled")
                src_label = (
                    "external LLM"
                    if rewrite_source == "external-llm"
                    else ("internal fallback" if rewrite_source == "internal-fallback" else "internal model")
                )
                status_label_var.set(f"Select rewrite 1, 2, or 3. Press Quit to cancel. Source: {src_label}.")
                if wait_label_var is not None:
                    wait_label_var.set("Generation complete.")
                refresh_analysis_status_line()

            root.after(0, on_ready)

        threading.Thread(target=worker, daemon=True).start()
        return "break"

    def maybe_start_selection_rewrite(_event: Optional[Any] = None) -> bool:
        sel_pair = selection_index_pair()
        if sel_pair is None:
            return False
        if state.get("analyzing") or state.get("rewrite_busy"):
            return True
        if not state.get("has_analysis"):
            show_transient_status_then_restore_stats(
                "Rewrite menu is available after Analyze has been run at least once.",
                duration_ms=5000,
            )
            return True

        text_snapshot = current_text()
        raw_start_idx, raw_end_idx = sel_pair
        raw_start_char = char_offset_for_index(raw_start_idx)
        raw_end_char = char_offset_for_index(raw_end_idx)
        if raw_end_char <= raw_start_char:
            return True

        expanded = expand_span_to_full_lines(raw_start_char, raw_end_char, text_snapshot)
        if expanded is None:
            show_transient_status_then_restore_stats(
                "Selected text could not be expanded to full lines.",
                duration_ms=5000,
            )
            return True
        expanded_start_char, expanded_end_char = expanded

        clamped = clamp_span_to_scored_range(expanded_start_char, expanded_end_char, text_snapshot)
        if clamped is None:
            show_transient_status_then_restore_stats(
                "Selected text is outside analyzed/scored range. Adjust selection or run Analyze again.",
                duration_ms=6000,
            )
            return True
        start_char, end_char, was_clamped = clamped
        selected = text_snapshot[start_char:end_char]
        if not selected.strip():
            show_transient_status_then_restore_stats(
                "Selected text is empty after clamping to scored range.",
                duration_ms=5000,
            )
            return True
        chunk_for_span = resolve_chunk_for_span(start_char, end_char, text_snapshot)
        metrics = chunk_metrics_from_descriptor(chunk_for_span)
        if metrics is None:
            show_transient_status_then_restore_stats(
                "Rewrite menu is unavailable until Analyze completes successfully.",
                duration_ms=5000,
            )
            return True

        start_idx = text_widget.index(f"1.0+{start_char}c")
        end_idx = text_widget.index(f"1.0+{end_char}c")
        range_note = (
            "Selection was rounded to whole lines, then clamped to scored text only where needed."
            if was_clamped
            else "Selection was rounded to whole lines for rewrite generation."
        )
        launch_rewrite_popup_for_span(
            start_idx=start_idx,
            end_idx=end_idx,
            start_char=start_char,
            end_char=end_char,
            text_snapshot=text_snapshot,
            metrics=metrics,
            span_label="selected highlighted section",
            range_note=range_note,
            selection_mode=True,
        )
        return True

    def on_editor_right_click(event: Any) -> Optional[str]:
        if maybe_start_selection_rewrite(event):
            return "break"
        return None

    def on_editor_left_click_release(event: Any) -> None:
        queue_preview_focus_sync(delay_ms=0)
        if state.get("internal_update"):
            return
        try:
            click_idx = text_widget.index(f"@{event.x},{event.y}")
        except Exception:
            return
        queue_synonym_lookup_for_click(click_idx)

    def on_delete_key(event: Any) -> Optional[str]:
        """
        Delete selected text as a tracked operation so Undo can restore it.
        If no selection exists, return None to preserve normal single-char delete behavior.
        """
        del event
        if state.get("internal_update"):
            return None
        if state.get("analyzing") or state.get("rewrite_busy"):
            return "break"
        sel_pair = selection_index_pair()
        if sel_pair is None:
            return None
        start_idx, end_idx = sel_pair
        if not text_widget.compare(end_idx, ">", start_idx):
            return None
        old_text = text_widget.get(start_idx, end_idx)
        if not old_text:
            return "break"
        text_widget.focus_set()
        text_widget.mark_set("insert", start_idx)
        text_widget.delete(start_idx, end_idx)
        text_widget.tag_raise("edited")
        text_widget.tag_raise("misspelled")
        queue_line_numbers_refresh(delay_ms=0)
        queue_line_bars_refresh(delay_ms=0)
        state["spell_version"] = int(state.get("spell_version", 0)) + 1
        queue_spellcheck(delay_ms=0)
        queue_preview_render(delay_ms=0)
        queue_preview_focus_sync(delay_ms=0)
        remember_one_level_undo(
            operation_label="delete selection",
            start_idx=start_idx,
            old_text=old_text,
            new_text="",
        )
        mark_analysis_stale()
        show_transient_status_then_restore_stats(
            f"Deleted selected block.{analysis_stale_suffix()}",
            duration_ms=8000,
        )
        return "break"

    def on_backspace_key(event: Any) -> Optional[str]:
        # Match Delete behavior for selected blocks so one-level Undo is available
        # for both Del and Backspace-based deletions.
        return on_delete_key(event)

    def on_low_segment_right_click(event: Any, tag: str) -> str:
        # Selection-based rewrite mode takes precedence if a span is highlighted.
        if maybe_start_selection_rewrite(event):
            return "break"
        if state.get("analyzing") or state.get("rewrite_busy"):
            return "break"
        if not state.get("has_analysis"):
            show_transient_status_then_restore_stats(
                "Rewrite menu is available after Analyze has been run at least once.",
                duration_ms=5000,
            )
            return "break"

        click_idx = text_widget.index(f"@{event.x},{event.y}")
        range_pair = tag_range_containing_index(tag, click_idx)
        if range_pair is None:
            return "break"
        start_idx, end_idx = range_pair
        if not text_widget.compare(end_idx, ">", start_idx):
            return "break"

        text_snapshot = current_text()
        raw_start_char = char_offset_for_index(start_idx)
        raw_end_char = char_offset_for_index(end_idx)
        clamped = clamp_span_to_scored_range(raw_start_char, raw_end_char, text_snapshot)
        if clamped is None:
            show_transient_status_then_restore_stats(
                "Selected LOW section is outside analyzed/scored text.",
                duration_ms=6000,
            )
            return "break"
        start_char, end_char, was_clamped = clamped
        if end_char <= start_char:
            return "break"

        selected = text_snapshot[start_char:end_char]
        if not selected.strip():
            return "break"
        chunk_for_span = resolve_chunk_for_span(start_char, end_char, text_snapshot)
        metrics = chunk_metrics_from_descriptor(chunk_for_span)
        if metrics is None:
            show_transient_status_then_restore_stats(
                "Rewrite menu is unavailable until Analyze completes successfully.",
                duration_ms=5000,
            )
            return "break"

        clamped_start_idx = text_widget.index(f"1.0+{start_char}c")
        clamped_end_idx = text_widget.index(f"1.0+{end_char}c")
        range_note = (
            "LOW section overlapped unscored text. Rewrite options are limited to scored/analyzed text only."
            if was_clamped
            else None
        )
        return launch_rewrite_popup_for_span(
            start_idx=clamped_start_idx,
            end_idx=clamped_end_idx,
            start_char=start_char,
            end_char=end_char,
            text_snapshot=text_snapshot,
            metrics=metrics,
            span_label="selected LOW section",
            range_note=range_note,
            selection_mode=False,
        )

    def clear_current_segment_tags() -> None:
        hide_tooltip()
        for tag in state["segment_tags"]:
            try:
                text_widget.tag_remove(tag, "1.0", "end")
                text_widget.tag_delete(tag)
            except Exception:
                pass
        state["segment_tags"] = []
        state["segment_labels"] = {}
        state["segment_infos"] = {}
        state["line_contrib_map"] = {}
        state["line_contrib_max_abs"] = 0.0
        state["line_contrib_last_line"] = 0
        state["line_contrib_top_lines"] = set()
        text_widget.tag_remove("unscored", "1.0", "end")
        queue_line_bars_refresh(delay_ms=0)
        clear_preview_segment_backgrounds()

    def clear_prior_backgrounds() -> None:
        for tag in state["prior_bg_tags"]:
            try:
                text_widget.tag_remove(tag, "1.0", "end")
                text_widget.tag_delete(tag)
            except Exception:
                pass
        state["prior_bg_tags"] = []
        state["prior_line_contrib_maps"] = []
        queue_line_bars_refresh(delay_ms=0)

    def add_prior_background_from_ranges(ranges: Tuple[Any, ...], color: str) -> None:
        if not ranges:
            return
        state["prior_counter"] += 1
        tag = f"prior_bg_{state['prior_counter']}"
        text_widget.tag_configure(tag, background=color)
        for i in range(0, len(ranges), 2):
            text_widget.tag_add(tag, ranges[i], ranges[i + 1])
        text_widget.tag_lower(tag)
        state["prior_bg_tags"].append(tag)

    def snapshot_current_to_priors() -> None:
        # Convert current foreground analysis and edit markers into faint background "prior" tags.
        current_map_obj = state.get("line_contrib_map")
        if isinstance(current_map_obj, dict) and current_map_obj:
            try:
                current_max = float(state.get("line_contrib_max_abs", 0.0))
            except Exception:
                current_max = 0.0
            if np.isfinite(current_max) and current_max > 0.0:
                prior_entries = state.get("prior_line_contrib_maps")
                if not isinstance(prior_entries, list):
                    prior_entries = []
                prior_entries.append({"map": dict(current_map_obj), "max_abs": float(current_max)})
                state["prior_line_contrib_maps"] = prior_entries[-8:]

        for tag in list(state["segment_tags"]):
            ranges = text_widget.tag_ranges(tag)
            label = state["segment_labels"].get(tag, "")
            color = "#5a1f1f" if label == "LOW" else "#203026"
            add_prior_background_from_ranges(ranges, color)
        clear_current_segment_tags()

        edited_ranges = text_widget.tag_ranges("edited")
        add_prior_background_from_ranges(edited_ranges, "#3a3420")
        text_widget.tag_remove("edited", "1.0", "end")

    def apply_edited_diff() -> None:
        state["pending_edit_job"] = None
        if state["internal_update"]:
            return
        prev_text = state["prev_text"]
        curr = current_text()
        if prev_text == curr:
            return
        j1, j2 = changed_span_in_new_text(prev_text, curr)
        if j2 > j1:
            text_widget.tag_add("edited", f"1.0+{j1}c", f"1.0+{j2}c")
        state["prev_text"] = curr
        text_widget.tag_raise("edited")
        text_widget.tag_raise("misspelled")

    def apply_spellcheck_spans(spans: List[Tuple[int, int]], version: int) -> None:
        if version != state.get("spell_version"):
            return
        if state["internal_update"]:
            return

        if state.get("last_spell_spans") == spans:
            return
        state["last_spell_spans"] = list(spans)

        text_widget.tag_remove("misspelled", "1.0", "end")
        for s, e in spans:
            text_widget.tag_add("misspelled", f"1.0+{s}c", f"1.0+{e}c")
        text_widget.tag_raise("misspelled")
        text_widget.tag_raise("edited")

    def run_spellcheck_worker(version: int, text_snapshot: str) -> None:
        spans = find_misspelled_spans(text_snapshot, english_words)
        try:
            root.after(0, lambda: apply_spellcheck_spans(spans, version))
        except Exception:
            pass

    def queue_spellcheck(delay_ms: int) -> None:
        pending_spell = state.get("pending_spell_job")
        if pending_spell is not None:
            try:
                root.after_cancel(pending_spell)
            except Exception:
                pass

        def launch() -> None:
            state["pending_spell_job"] = None
            if state["internal_update"]:
                return
            version = int(state.get("spell_version", 0))
            text_snapshot = current_text()
            threading.Thread(
                target=run_spellcheck_worker,
                args=(version, text_snapshot),
                daemon=True,
            ).start()

        state["pending_spell_job"] = root.after(delay_ms, launch)

    def on_modified(_event: Any) -> None:
        if state["internal_update"]:
            text_widget.edit_modified(False)
            return
        if not text_widget.edit_modified():
            return
        text_widget.edit_modified(False)
        sync_save_button_state(enabled_base=(not bool(state.get("analyzing"))))
        cancel_pending_synonym_lookup()
        state["synonym_request_id"] = int(state.get("synonym_request_id", 0)) + 1
        mark_analysis_stale()
        pending = state.get("pending_edit_job")
        if pending is not None:
            try:
                root.after_cancel(pending)
            except Exception:
                pass
        state["pending_edit_job"] = root.after(90, apply_edited_diff)
        queue_line_numbers_refresh(delay_ms=80)
        state["spell_version"] = int(state.get("spell_version", 0)) + 1
        queue_spellcheck(delay_ms=280)
        queue_preview_render(delay_ms=120)
        queue_preview_focus_sync(delay_ms=0)

    def register_analysis_chunk(
        chunk_start_char: int,
        analyzed_text_snapshot: str,
        result: Dict[str, Any],
        shifted_profile: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        doc_len = len(analyzed_text_snapshot)
        start = max(0, min(doc_len, int(chunk_start_char)))
        local_analyzed = len(analyzed_text_snapshot) - start
        if isinstance(shifted_profile, dict):
            raw_end = shifted_profile.get("analyzed_char_end")
            if raw_end is not None:
                try:
                    local_analyzed = int(raw_end) - start
                except Exception:
                    pass
        analyzed_end = max(start, min(doc_len, start + max(0, int(local_analyzed))))

        state["analysis_chunk_id_seq"] = int(state.get("analysis_chunk_id_seq", 0)) + 1
        chunk_id = int(state.get("analysis_chunk_id_seq", 1))

        chunk_desc: Dict[str, Any] = {
            "id": chunk_id,
            "char_start": int(start),
            "char_end": int(analyzed_end),
            "analyzed_char_end": int(analyzed_end),
            "metrics": {
                "binoculars_score": float(result["binoculars"]["score"]),
                "observer_logPPL": float(result["observer"]["logPPL"]),
                "performer_logPPL": float(result["performer"]["logPPL"]),
                "cross_logXPPL": float(result["cross"]["logXPPL"]),
                "transitions": int(result["input"]["transitions"]),
            },
            "profile": shifted_profile if isinstance(shifted_profile, dict) else None,
            "rows": list((shifted_profile or {}).get("rows", [])),
        }
        merge_chunk_descriptor(chunk_desc)
        recompute_chunk_coverage_state(analyzed_text_snapshot)
        return chunk_desc

    def render_analysis_chunks() -> None:
        clear_current_segment_tags()
        doc_text = current_text()
        doc_len = len(doc_text)
        chunks = analysis_chunk_list()
        combined_rows: List[Dict[str, Any]] = []
        for chunk in chunks:
            rows_obj = chunk.get("rows")
            rows = rows_obj if isinstance(rows_obj, list) else []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                merged_row = dict(row)
                combined_rows.append(merged_row)

        combined_rows.sort(key=lambda r: int(r.get("char_start", 0)))
        for idx, row in enumerate(combined_rows, start=1):
            row["paragraph_id"] = idx

        active_chunk = resolve_active_chunk(doc_text)
        active_metrics = chunk_metrics_from_descriptor(active_chunk)
        observer_for_annotations = (
            float(active_metrics.get("observer_logPPL", float("nan")))
            if isinstance(active_metrics, dict)
            else float("nan")
        )
        if not np.isfinite(observer_for_annotations):
            observer_for_annotations = 0.0

        annotations: Dict[int, Dict[str, Any]] = {}
        if combined_rows:
            _, _, annotations, _endnotes = _prepare_heatmap_annotations(
                rows=combined_rows,
                top_k=top_k,
                observer_logppl=observer_for_annotations,
            )

        rebuild_line_contribution_map(combined_rows, annotations, max(0, int(state.get("analysis_covered_until", 0))))

        if combined_rows and annotations:
            for row in combined_rows:
                ann = annotations.get(int(row.get("paragraph_id", -1)))
                if ann is None:
                    continue
                tag = f"heat_seg_{ann['note_num']}"
                try:
                    c_start = int(row.get("char_start", 0))
                    c_end = int(row.get("char_end", c_start))
                except Exception:
                    continue
                c_start = max(0, min(doc_len, c_start))
                c_end = max(c_start, min(doc_len, c_end))
                if c_end <= c_start:
                    continue
                start_idx = f"1.0+{c_start}c"
                end_idx = f"1.0+{c_end}c"
                color = "#ff4d4d" if ann["label"] == "LOW" else "#39d97f"
                text_widget.tag_configure(tag, foreground=color)
                text_widget.tag_add(tag, start_idx, end_idx)
                text_widget.tag_bind(tag, "<Enter>", lambda e, info=dict(ann): show_tooltip(e, info))
                text_widget.tag_bind(tag, "<Motion>", move_tooltip)
                text_widget.tag_bind(tag, "<Leave>", lambda _e: hide_tooltip())
                if ann["label"] == "LOW":
                    text_widget.tag_bind(tag, "<Button-3>", lambda e, t=tag: on_low_segment_right_click(e, t))
                state["segment_tags"].append(tag)
                state["segment_labels"][tag] = ann["label"]
                state["segment_infos"][tag] = dict(ann)

        text_widget.tag_remove("unscored", "1.0", "end")
        for s, e in get_unscored_intervals(doc_text):
            if e <= s:
                continue
            text_widget.tag_add("unscored", f"1.0+{s}c", f"1.0+{e}c")
        text_widget.tag_raise("unscored")
        text_widget.tag_raise("misspelled")
        text_widget.tag_raise("edited")
        apply_preview_segment_backgrounds()

    def set_controls(enabled: bool) -> None:
        btn_state = "normal" if enabled else "disabled"
        analyze_btn.configure(state=btn_state)
        try:
            analyze_next_btn.configure(state=btn_state)
        except Exception:
            pass
        open_btn.configure(state=btn_state)
        sync_save_button_state(enabled_base=enabled)
        clear_priors_btn.configure(state=btn_state)
        quit_btn.configure(state=btn_state)
        sync_undo_button_state(enabled_base=enabled)
        text_widget.configure(state="normal" if enabled else "disabled")

    def show_progress_popup(message: str = "Performing analysis on current text...") -> None:
        popup = tk.Toplevel(root)
        popup.title("Analyzing")
        popup.transient(root)
        popup.resizable(False, False)
        popup.grab_set()
        msg = tk.Label(
            popup,
            text=message,
            padx=18,
            pady=14,
        )
        msg.pack()
        popup.update_idletasks()
        x = root.winfo_rootx() + (root.winfo_width() // 2) - (popup.winfo_width() // 2)
        y = root.winfo_rooty() + (root.winfo_height() // 2) - (popup.winfo_height() // 2)
        popup.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        state["progress_popup"] = popup

    def close_progress_popup() -> None:
        popup = state.get("progress_popup")
        if popup is None:
            return
        try:
            popup.grab_release()
        except Exception:
            pass
        try:
            popup.destroy()
        except Exception:
            pass
        state["progress_popup"] = None

    def show_save_popup(out_path: str) -> None:
        popup = tk.Toplevel(root)
        popup.title("Saving")
        popup.transient(root)
        popup.resizable(False, False)
        popup.grab_set()
        popup.configure(bg="#111111")
        msg = tk.Label(
            popup,
            text=(
                "One moment please, performing Save...\n\n"
                f"Saving to:\n{out_path}"
            ),
            justify="left",
            anchor="w",
            bg="#111111",
            fg="#f0f0f0",
            padx=18,
            pady=14,
            wraplength=700,
        )
        msg.pack(fill="both", expand=True)
        popup.update_idletasks()
        x = root.winfo_rootx() + (root.winfo_width() // 2) - (popup.winfo_width() // 2)
        y = root.winfo_rooty() + (root.winfo_height() // 2) - (popup.winfo_height() // 2)
        popup.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        state["save_popup"] = popup

    def close_save_popup() -> None:
        popup = state.get("save_popup")
        if popup is None:
            return
        try:
            popup.grab_release()
        except Exception:
            pass
        try:
            popup.destroy()
        except Exception:
            pass
        state["save_popup"] = None

    def finish_analysis_error(message: str) -> None:
        cancel_pending_status_restore()
        close_progress_popup()
        state["analyzing"] = False
        set_controls(True)
        status_var.set(f"Analyze failed: {message}")
        messagebox.showerror("Analyze Error", message)

    def finish_analysis_success(
        analyzed_text: str,
        result: Dict[str, Any],
        profile: Optional[Dict[str, Any]],
        chunk_start_char: int,
    ) -> None:
        cancel_pending_status_restore()
        close_progress_popup()
        state["internal_update"] = True
        try:
            text_widget.tag_remove("edited", "1.0", "end")
            shifted_profile = shift_profile_to_global_char_offsets(profile, int(chunk_start_char))
            register_analysis_chunk(
                chunk_start_char=int(chunk_start_char),
                analyzed_text_snapshot=analyzed_text,
                result=result,
                shifted_profile=shifted_profile,
            )
            render_analysis_chunks()
            state["baseline_text"] = analyzed_text
            state["prev_text"] = analyzed_text
            text_widget.edit_modified(False)
        finally:
            state["internal_update"] = False

        state["has_analysis"] = True
        set_clear_priors_visible(True)
        set_analyze_next_visible(bool(state.get("analysis_next_available")))
        state["b_score_stale"] = False
        state["last_b_score"] = float(result["binoculars"]["score"])
        state["analyzing"] = False
        set_controls(True)
        restore_idx = str(state.get("analyze_cursor_idx", "1.0"))
        restore_top = state.get("analyze_yview_top", 0.0)
        try:
            top = float(restore_top)
        except Exception:
            top = 0.0
        top = max(0.0, min(1.0, top))
        try:
            text_widget.yview_moveto(top)
        except Exception:
            pass
        try:
            text_widget.mark_set("insert", restore_idx)
        except Exception:
            pass
        refresh_analysis_status_line()
        queue_preview_focus_sync(delay_ms=0)

    def prepare_for_analysis() -> None:
        pending = state.get("pending_edit_job")
        if pending is not None:
            try:
                root.after_cancel(pending)
            except Exception:
                pass
            state["pending_edit_job"] = None
            apply_edited_diff()

        pending_spell = state.get("pending_spell_job")
        if pending_spell is not None:
            try:
                root.after_cancel(pending_spell)
            except Exception:
                pass
            state["pending_spell_job"] = None

    def begin_chunk_analysis(start_char: int, status_message: str) -> None:
        if state["analyzing"]:
            return
        cancel_pending_status_restore()
        prepare_for_analysis()
        snapshot_current_to_priors()
        analyzed_text = current_text()
        start = max(0, min(len(analyzed_text), int(start_char)))
        if start >= len(analyzed_text):
            recompute_chunk_coverage_state(analyzed_text)
            set_analyze_next_visible(bool(state.get("analysis_next_available")))
            show_transient_status_then_restore_stats(
                f"No remaining text to analyze.{analysis_stale_suffix()}",
                duration_ms=5000,
            )
            return

        cursor_idx = text_widget.index("insert")
        yview = text_widget.yview()
        yview_top = float(yview[0]) if yview else 0.0
        state["analyze_cursor_idx"] = cursor_idx
        state["analyze_yview_top"] = yview_top
        state["analyzing"] = True
        status_var.set(status_message)
        set_controls(False)
        show_progress_popup(status_message)

        def worker() -> None:
            try:
                result, profile = analyze_text_document(
                    cfg_path=cfg_path,
                    text=analyzed_text[start:],
                    input_label=src_path,
                    diagnose_paragraphs=False,
                    diagnose_top_k=top_k,
                    need_paragraph_profile=True,
                    text_max_tokens_override=text_max_tokens_override,
                )
            except Exception as exc:
                root.after(0, lambda: finish_analysis_error(str(exc)))
                return
            root.after(0, lambda: finish_analysis_success(analyzed_text, result, profile, start))

        threading.Thread(target=worker, daemon=True).start()

    def on_analyze() -> None:
        if state["analyzing"]:
            return
        text_snapshot = current_text()
        chunks = analysis_chunk_list()
        if not state.get("has_analysis") or not chunks:
            begin_chunk_analysis(0, "Performing analysis on current text...")
            return
        active_chunk = resolve_active_chunk(text_snapshot)
        status_message = "Performing analysis on active chunk..."
        if isinstance(active_chunk, dict):
            start_char = int(active_chunk.get("char_start", 0))
            try:
                chunk_end_char = int(active_chunk.get("analyzed_char_end", start_char))
            except Exception:
                chunk_end_char = start_char
            try:
                chunk_start_line = int(str(text_widget.index(f"1.0+{max(0, start_char)}c")).split(".")[0])
            except Exception:
                chunk_start_line = 1
            chunk_end_line = line_number_for_char_end(text_snapshot, max(start_char + 1, chunk_end_char))
            if chunk_end_line < chunk_start_line:
                chunk_end_line = chunk_start_line
            status_message = (
                f"Performing analysis on active chunk (lines {chunk_start_line}-{chunk_end_line})..."
            )
        else:
            start_char = 0
        begin_chunk_analysis(start_char, status_message)

    def on_analyze_next() -> None:
        if state["analyzing"]:
            return
        if not state.get("has_analysis"):
            on_analyze()
            return
        text_snapshot = current_text()
        start_char = int(state.get("analysis_covered_until", 0))
        if start_char >= len(text_snapshot):
            set_analyze_next_visible(False)
            show_transient_status_then_restore_stats(
                f"All document text is already analyzed.{analysis_stale_suffix()}",
                duration_ms=5000,
            )
            return
        begin_chunk_analysis(start_char, "Performing analysis on next unscored chunk...")

    def sidecar_state_path_for_document(doc_path: str) -> str:
        stem, _ext = os.path.splitext(os.path.abspath(doc_path))
        return stem + ".json"

    def text_sha256(text_value: str) -> str:
        raw = str(text_value or "")
        return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()

    def to_json_compatible(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            out: Dict[str, Any] = {}
            for k, v in value.items():
                out[str(k)] = to_json_compatible(v)
            return out
        if isinstance(value, (list, tuple, set)):
            return [to_json_compatible(v) for v in value]
        return str(value)

    def collect_char_ranges_for_tag(tag: str) -> List[List[int]]:
        spans: List[List[int]] = []
        ranges = text_widget.tag_ranges(tag)
        for i in range(0, len(ranges), 2):
            s_idx = str(ranges[i])
            e_idx = str(ranges[i + 1])
            s = char_offset_for_index(s_idx)
            e = char_offset_for_index(e_idx)
            if e <= s:
                continue
            spans.append([int(s), int(e)])
        return spans

    def collect_prior_background_payload() -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for tag in state.get("prior_bg_tags", []):
            try:
                color = str(text_widget.tag_cget(tag, "background") or "")
            except Exception:
                color = ""
            spans = collect_char_ranges_for_tag(tag)
            if not spans:
                continue
            payload.append({"color": color, "ranges": spans})
        return payload

    def collect_gui_state_payload(doc_path: str, text_snapshot: str) -> Dict[str, Any]:
        yview = text_widget.yview()
        yview_top = float(yview[0]) if yview else 0.0
        cursor_char = char_offset_for_index(text_widget.index("insert"))
        return {
            "binoculars_gui_state": True,
            "version": 1,
            "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "document_path": os.path.abspath(doc_path),
            "document_basename": os.path.basename(doc_path),
            "text_sha256": text_sha256(text_snapshot),
            "state": to_json_compatible(
                {
                    "baseline_text": state.get("baseline_text", text_snapshot),
                    "prev_text": state.get("prev_text", text_snapshot),
                    "has_analysis": bool(state.get("has_analysis")),
                    "b_score_stale": bool(state.get("b_score_stale")),
                    "last_b_score": state.get("last_b_score"),
                    "last_analysis_status_core": state.get("last_analysis_status_core", ""),
                    "last_analysis_metrics": state.get("last_analysis_metrics"),
                    "analyzed_char_end": int(state.get("analyzed_char_end", 0)),
                    "analysis_chunks": state.get("analysis_chunks", []),
                    "analysis_chunk_id_seq": int(state.get("analysis_chunk_id_seq", 0)),
                    "analysis_covered_until": int(state.get("analysis_covered_until", 0)),
                    "analysis_next_available": bool(state.get("analysis_next_available", False)),
                    "prior_counter": int(state.get("prior_counter", 0)),
                    "prior_line_contrib_maps": state.get("prior_line_contrib_maps", []),
                    "prior_backgrounds": collect_prior_background_payload(),
                    "edited_ranges": collect_char_ranges_for_tag("edited"),
                    "cursor_char": int(cursor_char),
                    "yview_top": float(yview_top),
                }
            ),
        }

    def write_sidecar_state(doc_path: str, text_snapshot: str) -> Tuple[bool, str]:
        sidecar_path = sidecar_state_path_for_document(doc_path)
        payload = collect_gui_state_payload(doc_path=doc_path, text_snapshot=text_snapshot)
        try:
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            return (False, f"{exc}")
        return (True, sidecar_path)

    def load_sidecar_state(doc_path: str, text_snapshot: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], str]:
        doc_abs = os.path.abspath(doc_path)
        if os.path.splitext(doc_abs)[1].lower() != ".md":
            return (None, None, "")
        sidecar_path = sidecar_state_path_for_document(doc_abs)
        if not os.path.isfile(sidecar_path):
            return (None, None, sidecar_path)
        try:
            with open(sidecar_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            return (None, f"State sidecar found but could not be read: {exc}", sidecar_path)
        if not isinstance(payload, dict) or not bool(payload.get("binoculars_gui_state", False)):
            return (None, f"State sidecar ignored (unrecognized format): {sidecar_path}", sidecar_path)
        expected_hash = str(payload.get("text_sha256", "") or "")
        actual_hash = text_sha256(text_snapshot)
        if expected_hash and expected_hash != actual_hash:
            return (
                None,
                "State sidecar ignored: markdown content does not match saved state hash.",
                sidecar_path,
            )
        raw_state = payload.get("state")
        if not isinstance(raw_state, dict):
            return (None, "State sidecar ignored: missing state object.", sidecar_path)
        return (raw_state, None, sidecar_path)

    def clamp_span_to_document(s_raw: Any, e_raw: Any, text_len: int) -> Optional[Tuple[int, int]]:
        try:
            s = int(s_raw)
            e = int(e_raw)
        except Exception:
            return None
        s = max(0, min(text_len, s))
        e = max(s, min(text_len, e))
        if e <= s:
            return None
        return (s, e)

    def load_persisted_state_into_gui(raw_state: Dict[str, Any], text_snapshot: str) -> Tuple[bool, str]:
        text_len = len(text_snapshot)

        raw_chunks = raw_state.get("analysis_chunks")
        chunks_in = raw_chunks if isinstance(raw_chunks, list) else []
        cleaned_chunks: List[Dict[str, Any]] = []
        max_chunk_id = 0
        for idx, raw_chunk in enumerate(chunks_in, start=1):
            if not isinstance(raw_chunk, dict):
                continue
            span = clamp_span_to_document(
                raw_chunk.get("char_start", 0),
                raw_chunk.get("analyzed_char_end", raw_chunk.get("char_end", raw_chunk.get("char_start", 0))),
                text_len,
            )
            if span is None:
                continue
            cs, ce = span
            try:
                chunk_id = int(raw_chunk.get("id", idx))
            except Exception:
                chunk_id = idx
            max_chunk_id = max(max_chunk_id, chunk_id)

            metrics_raw = raw_chunk.get("metrics")
            metrics = metrics_raw if isinstance(metrics_raw, dict) else {}
            cleaned_metrics: Dict[str, Any] = {}
            for key in ("binoculars_score", "observer_logPPL", "performer_logPPL", "cross_logXPPL"):
                try:
                    val = float(metrics.get(key, float("nan")))
                except Exception:
                    val = float("nan")
                cleaned_metrics[key] = val
            try:
                cleaned_metrics["transitions"] = int(metrics.get("transitions", 0))
            except Exception:
                cleaned_metrics["transitions"] = 0

            rows_raw = raw_chunk.get("rows")
            rows_in = rows_raw if isinstance(rows_raw, list) else []
            cleaned_rows: List[Dict[str, Any]] = []
            for raw_row in rows_in:
                if not isinstance(raw_row, dict):
                    continue
                row_span = clamp_span_to_document(
                    raw_row.get("char_start", 0),
                    raw_row.get("char_end", raw_row.get("char_start", 0)),
                    text_len,
                )
                if row_span is None:
                    continue
                rs, re_ = row_span
                row_copy = dict(raw_row)
                row_copy["char_start"] = int(rs)
                row_copy["char_end"] = int(re_)
                cleaned_rows.append(row_copy)

            cleaned_chunks.append(
                {
                    "id": int(chunk_id),
                    "char_start": int(cs),
                    "char_end": int(ce),
                    "analyzed_char_end": int(ce),
                    "metrics": cleaned_metrics,
                    "profile": None,
                    "rows": cleaned_rows,
                }
            )

        cleaned_chunks.sort(key=lambda c: int(c.get("char_start", 0)))
        state["analysis_chunks"] = cleaned_chunks
        try:
            seq_saved = int(raw_state.get("analysis_chunk_id_seq", 0))
        except Exception:
            seq_saved = 0
        state["analysis_chunk_id_seq"] = max(seq_saved, max_chunk_id)
        recompute_chunk_coverage_state(text_snapshot)

        state["has_analysis"] = bool(raw_state.get("has_analysis")) and bool(cleaned_chunks)
        state["analysis_next_available"] = bool(state.get("analysis_next_available", False))
        try:
            state["last_b_score"] = (
                None if raw_state.get("last_b_score") is None else float(raw_state.get("last_b_score"))
            )
        except Exception:
            state["last_b_score"] = None
        state["b_score_stale"] = bool(raw_state.get("b_score_stale", False))
        state["last_analysis_status_core"] = str(raw_state.get("last_analysis_status_core", "") or "")
        lam_raw = raw_state.get("last_analysis_metrics")
        state["last_analysis_metrics"] = lam_raw if isinstance(lam_raw, dict) else None
        try:
            state["analyzed_char_end"] = int(raw_state.get("analyzed_char_end", state.get("analysis_covered_until", 0)))
        except Exception:
            state["analyzed_char_end"] = int(state.get("analysis_covered_until", 0))

        try:
            state["prior_counter"] = max(0, int(raw_state.get("prior_counter", 0)))
        except Exception:
            state["prior_counter"] = 0
        state["prior_line_contrib_maps"] = []
        raw_prior_maps = raw_state.get("prior_line_contrib_maps")
        prior_maps_in = raw_prior_maps if isinstance(raw_prior_maps, list) else []
        for entry in prior_maps_in:
            if not isinstance(entry, dict):
                continue
            map_obj = entry.get("map")
            if not isinstance(map_obj, dict):
                continue
            clean_map: Dict[int, float] = {}
            for k, v in map_obj.items():
                try:
                    line_no = int(k)
                    line_val = float(v)
                except Exception:
                    continue
                if line_no <= 0 or (not np.isfinite(line_val)):
                    continue
                clean_map[line_no] = line_val
            if not clean_map:
                continue
            try:
                max_abs = float(entry.get("max_abs", 0.0))
            except Exception:
                max_abs = 0.0
            if not np.isfinite(max_abs) or max_abs <= 0.0:
                max_abs = max(abs(v) for v in clean_map.values())
            state["prior_line_contrib_maps"].append({"map": clean_map, "max_abs": float(max_abs)})
        state["prior_line_contrib_maps"] = state.get("prior_line_contrib_maps", [])[-8:]

        raw_prior_bgs = raw_state.get("prior_backgrounds")
        prior_bgs_in = raw_prior_bgs if isinstance(raw_prior_bgs, list) else []
        for entry in prior_bgs_in:
            if not isinstance(entry, dict):
                continue
            color = str(entry.get("color", "") or "")
            ranges_obj = entry.get("ranges")
            spans = ranges_obj if isinstance(ranges_obj, list) else []
            rebuilt: List[str] = []
            for pair in spans:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                span = clamp_span_to_document(pair[0], pair[1], text_len)
                if span is None:
                    continue
                s, e = span
                rebuilt.append(f"1.0+{s}c")
                rebuilt.append(f"1.0+{e}c")
            if rebuilt:
                add_prior_background_from_ranges(tuple(rebuilt), color if color else "#303030")

        text_widget.tag_remove("edited", "1.0", "end")
        raw_edited = raw_state.get("edited_ranges")
        edited_in = raw_edited if isinstance(raw_edited, list) else []
        for pair in edited_in:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            span = clamp_span_to_document(pair[0], pair[1], text_len)
            if span is None:
                continue
            s, e = span
            text_widget.tag_add("edited", f"1.0+{s}c", f"1.0+{e}c")

        if state["has_analysis"]:
            render_analysis_chunks()
        else:
            clear_current_segment_tags()
            text_widget.tag_remove("unscored", "1.0", "end")
            apply_preview_segment_backgrounds()

        text_widget.tag_raise("edited")
        text_widget.tag_raise("misspelled")
        set_clear_priors_visible(bool(state["has_analysis"]))
        set_analyze_next_visible(bool(state.get("analysis_next_available")))

        try:
            cursor_char = int(raw_state.get("cursor_char", 0))
        except Exception:
            cursor_char = 0
        cursor_char = max(0, min(text_len, cursor_char))
        try:
            yview_top = float(raw_state.get("yview_top", 0.0))
        except Exception:
            yview_top = 0.0
        yview_top = max(0.0, min(1.0, yview_top))

        try:
            text_widget.yview_moveto(yview_top)
        except Exception:
            pass
        try:
            text_widget.mark_set("insert", f"1.0+{cursor_char}c")
            text_widget.see(f"1.0+{cursor_char}c")
        except Exception:
            pass
        text_widget.edit_modified(False)
        return (True, "Loaded analysis state sidecar.")

    def save_current_document(show_status: bool = True) -> bool:
        content = current_text()
        # Saves follow the currently loaded document location.
        src_dir = os.path.dirname(src_path) or "."
        stem, _ext = os.path.splitext(os.path.basename(src_path))
        stamp = datetime.now().strftime("%Y%m%d%H%M")
        out_path = os.path.join(src_dir, f"{stem}_edited_{stamp}.md")
        idx = 1
        while os.path.exists(out_path):
            idx += 1
            out_path = os.path.join(src_dir, f"{stem}_edited_{stamp}_{idx}.md")
        show_save_popup(out_path)
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as exc:
            close_save_popup()
            status_var.set(f"Save failed: {exc}{analysis_stale_suffix()}")
            messagebox.showerror("Save Error", str(exc))
            return False
        sidecar_ok, sidecar_result = write_sidecar_state(doc_path=out_path, text_snapshot=content)
        close_save_popup()
        state["last_saved_text"] = content
        sync_save_button_state(enabled_base=(not bool(state.get("analyzing"))))
        if show_status:
            if sidecar_ok:
                show_transient_status_then_restore_stats(
                    f"Saved edited file: {out_path} | Saved state: {sidecar_result}{analysis_stale_suffix()}",
                    duration_ms=8000,
                )
            else:
                show_transient_status_then_restore_stats(
                    f"Saved edited file: {out_path} | State save failed: {sidecar_result}{analysis_stale_suffix()}",
                    duration_ms=10000,
                )
                messagebox.showwarning(
                    "State Save Warning",
                    (
                        "Markdown save succeeded, but saving GUI state sidecar failed.\n\n"
                        f"File: {out_path}\n"
                        f"Reason: {sidecar_result}"
                    ),
                )
        return True

    def load_document_into_editor(new_src_path: str) -> bool:
        nonlocal src_path
        if state.get("analyzing") or state.get("rewrite_busy"):
            return False

        try:
            with open(new_src_path, "r", encoding="utf-8") as f:
                loaded_text = _normalize_markdown_hardbreaks(f.read())
        except Exception as exc:
            messagebox.showerror("Open Error", str(exc))
            status_var.set(f"Open failed: {exc}")
            return False
        persisted_state, state_load_warning, sidecar_path = load_sidecar_state(new_src_path, loaded_text)

        pending_keys = (
            "pending_edit_job",
            "pending_spell_job",
            "pending_line_numbers_job",
            "pending_line_bars_job",
            "pending_preview_job",
            "pending_focus_job",
        )
        for key in pending_keys:
            pending = state.get(key)
            if pending is None:
                continue
            try:
                root.after_cancel(pending)
            except Exception:
                pass
            state[key] = None

        cancel_pending_synonym_lookup()
        state["synonym_request_id"] = int(state.get("synonym_request_id", 0)) + 1
        cancel_pending_status_restore()
        hide_tooltip()
        close_rewrite_popup()
        close_progress_popup()
        close_save_popup()

        clear_current_segment_tags()
        clear_prior_backgrounds()
        text_widget.tag_remove("edited", "1.0", "end")
        text_widget.tag_remove("misspelled", "1.0", "end")
        text_widget.tag_remove("unscored", "1.0", "end")
        clear_synonym_panel()
        clear_one_level_undo()

        state["internal_update"] = True
        try:
            text_widget.delete("1.0", "end")
            text_widget.insert("1.0", loaded_text)
            text_widget.edit_modified(False)
        finally:
            state["internal_update"] = False

        src_path = os.path.abspath(new_src_path)
        opened_dir = os.path.dirname(src_path) or "."
        if os.path.isdir(opened_dir):
            state["open_dialog_dir"] = opened_dir
        root.title(f"Binoculars - {os.path.basename(src_path)}")
        state["baseline_text"] = loaded_text
        state["prev_text"] = loaded_text
        state["last_saved_text"] = loaded_text
        state["last_b_score"] = None
        state["has_analysis"] = False
        state["b_score_stale"] = False
        state["last_analysis_status_core"] = ""
        state["last_analysis_metrics"] = None
        state["analyzed_char_end"] = 0
        state["analysis_chunks"] = []
        state["analysis_chunk_id_seq"] = 0
        state["analysis_covered_until"] = 0
        state["analysis_next_available"] = False
        state["line_count"] = 0
        state["spell_version"] = int(state.get("spell_version", 0)) + 1
        set_clear_priors_visible(False)
        set_analyze_next_visible(False)
        loaded_state_message = ""
        if isinstance(persisted_state, dict):
            try:
                ok, loaded_state_message = load_persisted_state_into_gui(persisted_state, loaded_text)
            except Exception as exc:
                ok = False
                loaded_state_message = f"State sidecar found but could not be applied: {exc}"
            if ok:
                state["baseline_text"] = str(persisted_state.get("baseline_text", loaded_text))
                state["prev_text"] = str(persisted_state.get("prev_text", loaded_text))
                state["last_saved_text"] = loaded_text
                sync_save_button_state(enabled_base=(not bool(state.get("analyzing"))))
                state["spell_version"] = int(state.get("spell_version", 0)) + 1
                queue_spellcheck(delay_ms=0)
                queue_preview_render(delay_ms=0)
                queue_preview_focus_sync(delay_ms=0)
                refresh_line_numbers_now()
                refresh_line_bars_now()
                if state.get("has_analysis"):
                    refresh_analysis_status_line()
                    status_var.set(
                        f"Opened: {src_path}. {loaded_state_message} ({sidecar_path}){analysis_stale_suffix()}"
                    )
                else:
                    status_var.set(f"Opened: {src_path}. {loaded_state_message} ({sidecar_path})")
                if state_load_warning:
                    show_transient_status_then_restore_stats(
                        f"{state_load_warning}{analysis_stale_suffix()}",
                        duration_ms=8000,
                    )
                return True
            state_load_warning = loaded_state_message

        sync_save_button_state(enabled_base=(not bool(state.get("analyzing"))))

        text_widget.mark_set("insert", "1.0")
        text_widget.see("1.0")
        refresh_line_numbers_now()
        refresh_line_bars_now()
        queue_spellcheck(delay_ms=0)
        queue_preview_render(delay_ms=0)
        queue_preview_focus_sync(delay_ms=0)
        base_status = f"Opened: {src_path}. Press Analyze to score and highlight this document."
        if state_load_warning:
            base_status += f" State load note: {state_load_warning}"
        status_var.set(base_status)
        if state_load_warning and ("does not match saved state hash" in state_load_warning):
            messagebox.showwarning(
                "State Not Loaded",
                (
                    "This markdown file was modified outside of Binoculars.\n\n"
                    "The saved analysis state sidecar was not loaded."
                ),
            )
        return True

    def choose_open_file_with_dark_preference(initial_dir: str) -> str:
        """
        Best-effort dark file dialog on Linux.
        Native toolkit support varies by distro/desktop; fallback remains functional.
        """
        def normalize_dialog_choice(raw_choice: Any) -> str:
            # Some dialog APIs can return tuple-like or sentinel text values on cancel.
            if isinstance(raw_choice, (tuple, list)):
                if not raw_choice:
                    return ""
                return str(raw_choice[0] or "").strip()
            choice = str(raw_choice or "").strip()
            if choice in {"()", "[]", "''", '""'}:
                return ""
            return choice

        def try_zenity_open(dialog_dir: str) -> Tuple[bool, str]:
            if shutil.which("zenity") is None:
                return (False, "")
            cmd = [
                "zenity",
                "--file-selection",
                "--title=Open Markdown/Text File",
                f"--filename={os.path.join(dialog_dir, '')}",
                "--file-filter=Markdown/Text files | *.md *.markdown *.txt",
                "--file-filter=All files | *",
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            except Exception:
                return (False, "")
            if proc.returncode == 0:
                return (True, normalize_dialog_choice(proc.stdout))
            if proc.returncode in (1, 5):
                # User canceled/closed the dialog.
                return (True, "")
            return (False, "")

        def try_yad_open(dialog_dir: str) -> Tuple[bool, str]:
            if shutil.which("yad") is None:
                return (False, "")
            cmd = [
                "yad",
                "--file-selection",
                "--title=Open Markdown/Text File",
                f"--filename={os.path.join(dialog_dir, '')}",
                "--file-filter=Markdown/Text files (*.md *.markdown *.txt) | *.md *.markdown *.txt",
                "--file-filter=All files (*) | *",
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            except Exception:
                return (False, "")
            if proc.returncode == 0:
                return (True, normalize_dialog_choice(proc.stdout))
            if proc.returncode in (1, 252):
                return (True, "")
            return (False, "")

        if not bool(state.get("open_dialog_resize_bindings_installed", False)):
            resize_script = (
                "if {[winfo exists %W]} {"
                "namespace eval ::tk::dialog::file {"
                "variable showHiddenVar; set showHiddenVar 0; "
                "variable showHiddenBtn; set showHiddenBtn 0"
                "}; "
                "update idletasks; "
                "set sw [winfo screenwidth %W]; "
                "set sh [winfo screenheight %W]; "
                "set ww [expr {int($sw * 0.56)}]; "
                "set wh [expr {int($sh * 0.58)}]; "
                "if {$ww < 900} {set ww 900}; "
                "if {$wh < 560} {set wh 560}; "
                "if {$ww > 1380} {set ww 1380}; "
                "if {$wh > 900} {set wh 900}; "
                "wm minsize %W 900 560; "
                "wm geometry %W ${ww}x${wh}"
                "}"
            )
            for dlg_class in ("TkFDialog", "TkMotifFDialog"):
                try:
                    root.tk.call("bind", dlg_class, "<Map>", resize_script)
                except Exception:
                    pass
            state["open_dialog_resize_bindings_installed"] = True

        dialog_dir = str(initial_dir or "")
        if not os.path.isdir(dialog_dir):
            dialog_dir = os.path.expanduser("~")
        if not os.path.isdir(dialog_dir):
            dialog_dir = "."

        try:
            # Hide hidden files by default for readability.
            root.tk.eval(
                "namespace eval ::tk::dialog::file {"
                "variable showHiddenVar; set showHiddenVar 0; "
                "variable showHiddenBtn; set showHiddenBtn 0"
                "}"
            )
        except Exception:
            pass

        prior_gtk_theme = os.environ.get("GTK_THEME")
        applied_temp_dark_theme = False
        if sys.platform.startswith("linux") and not prior_gtk_theme:
            os.environ["GTK_THEME"] = "Adwaita:dark"
            applied_temp_dark_theme = True

        try:
            handled, native_choice = try_zenity_open(dialog_dir)
            if handled:
                return native_choice
            handled, native_choice = try_yad_open(dialog_dir)
            if handled:
                return native_choice

            raw_choice = filedialog.askopenfilename(
                parent=root,
                title="Open Markdown/Text File",
                initialdir=dialog_dir,
                filetypes=(
                    ("Markdown/Text files", "*.md *.markdown *.txt"),
                    ("All files", "*.*"),
                ),
            )
            return normalize_dialog_choice(raw_choice)
        finally:
            if applied_temp_dark_theme:
                os.environ.pop("GTK_THEME", None)

    def on_open() -> None:
        if state.get("analyzing") or state.get("rewrite_busy"):
            return

        if has_unsaved_changes():
            ask = messagebox.askyesnocancel(
                "Unsaved Changes",
                "This document has unsaved changes.\n\nSave before opening a new file?",
            )
            if ask is None:
                return
            if ask:
                if not save_current_document(show_status=False):
                    return

        initial_dir_obj = state.get("open_dialog_dir")
        initial_dir = str(initial_dir_obj or "")
        chosen = choose_open_file_with_dark_preference(initial_dir)
        if not chosen:
            return
        load_document_into_editor(chosen)

    def on_save() -> None:
        save_current_document(show_status=True)

    def on_clear_priors() -> None:
        clear_prior_backgrounds()
        show_transient_status_then_restore_stats(
            f"Cleared prior background highlights.{analysis_stale_suffix()}",
            duration_ms=8000,
        )

    def on_quit() -> None:
        if state.get("pending_edit_job") is not None:
            try:
                root.after_cancel(state["pending_edit_job"])
            except Exception:
                pass
        if state.get("pending_spell_job") is not None:
            try:
                root.after_cancel(state["pending_spell_job"])
            except Exception:
                pass
        if state.get("pending_line_numbers_job") is not None:
            try:
                root.after_cancel(state["pending_line_numbers_job"])
            except Exception:
                pass
        if state.get("pending_line_bars_job") is not None:
            try:
                root.after_cancel(state["pending_line_bars_job"])
            except Exception:
                pass
        if state.get("pending_preview_job") is not None:
            try:
                root.after_cancel(state["pending_preview_job"])
            except Exception:
                pass
        if state.get("pending_focus_job") is not None:
            try:
                root.after_cancel(state["pending_focus_job"])
            except Exception:
                pass
        cancel_pending_synonym_lookup()
        state["synonym_request_id"] = int(state.get("synonym_request_id", 0)) + 1
        cancel_pending_status_restore()
        hide_tooltip()
        close_rewrite_popup()
        close_progress_popup()
        close_save_popup()
        root.destroy()

    def place_initial_sash(retries: int = 24) -> None:
        try:
            width = int(root.winfo_width())
        except Exception:
            width = 0
        if width < 900 and retries > 0:
            root.after(80, lambda: place_initial_sash(retries - 1))
            return
        if width <= 0:
            width = 1200
        target = int(width * 0.67)
        min_left = 460
        max_left = max(min_left, width - 320)
        target = max(min_left, min(max_left, target))
        try:
            split_pane.sash_place(0, target, 0)
            state["sash_initialized"] = True
        except Exception:
            pass

    def on_root_configure(_event: Any) -> None:
        if not state.get("sash_initialized", False):
            place_initial_sash(retries=0)
        queue_line_numbers_refresh(delay_ms=0)

    analyze_btn = tk.Button(toolbar, text="Analyze", command=on_analyze, width=12)
    analyze_next_btn = tk.Button(toolbar, text="Analyze Next", command=on_analyze_next, width=12)
    open_btn = tk.Button(toolbar, text="Open", command=on_open, width=12)
    save_btn = tk.Button(toolbar, text="Save", command=on_save, width=12)
    undo_btn = tk.Button(toolbar, text="Undo", command=on_undo, width=12)
    clear_priors_btn = tk.Button(toolbar, text="Clear Priors", command=on_clear_priors, width=12)
    quit_btn = tk.Button(toolbar, text="Quit", command=on_quit, width=12)

    def set_clear_priors_visible(visible: bool) -> None:
        if visible == bool(state.get("clear_priors_visible")):
            return
        if visible:
            clear_priors_btn.pack(side="left", padx=(0, 8), before=quit_btn)
        else:
            clear_priors_btn.pack_forget()
        state["clear_priors_visible"] = bool(visible)

    def set_analyze_next_visible(visible: bool) -> None:
        if visible == bool(state.get("analyze_next_visible")):
            return
        if visible:
            analyze_next_btn.pack(side="left", padx=(0, 8), after=analyze_btn)
        else:
            analyze_next_btn.pack_forget()
        state["analyze_next_visible"] = bool(visible)

    open_btn.pack(side="left", padx=(0, 8))
    analyze_btn.pack(side="left", padx=(0, 8))
    set_analyze_next_visible(False)
    save_btn.pack(side="left", padx=(0, 8))
    undo_btn.pack(side="left", padx=(0, 8))
    quit_btn.pack(side="left")
    set_clear_priors_visible(False)
    for idx in range(SYNONYM_OPTION_COUNT):
        btn = tk.Button(
            synonym_btn_frame,
            text=str(idx + 1),
            width=5,
            state="disabled",
            command=lambda i=idx: apply_synonym_choice(i),
        )
        btn.pack(side="left", padx=(0, 6))
        synonym_buttons.append(btn)
    clear_synonym_panel()
    sync_undo_button_state(enabled_base=True)
    sync_save_button_state(enabled_base=True)

    state["internal_update"] = True
    text_widget.insert("1.0", initial_text)
    text_widget.edit_modified(False)
    state["internal_update"] = False
    refresh_line_numbers_now()
    refresh_line_bars_now()
    text_widget.configure(yscrollcommand=on_editor_scroll)
    preview_scroll.configure(command=on_preview_scroll)
    preview_text.bind("<MouseWheel>", on_preview_mousewheel, add="+")
    preview_text.bind("<Button-4>", on_preview_button4, add="+")
    preview_text.bind("<Button-5>", on_preview_button5, add="+")
    text_widget.bind("<<Modified>>", on_modified)
    text_widget.bind("<Button-3>", on_editor_right_click, add="+")
    text_widget.bind("<<Selection>>", lambda _e: queue_preview_focus_sync(delay_ms=0), add="+")
    text_widget.bind("<KeyRelease>", lambda _e: queue_preview_focus_sync(delay_ms=0), add="+")
    text_widget.bind("<ButtonRelease-1>", on_editor_left_click_release, add="+")
    text_widget.bind("<Delete>", on_delete_key, add="+")
    text_widget.bind("<BackSpace>", on_backspace_key, add="+")
    root.bind("<F9>", toggle_debug_overlay, add="+")
    text_widget.mark_set("insert", "1.0")
    text_widget.see("1.0")
    state["spell_version"] = int(state.get("spell_version", 0)) + 1
    queue_spellcheck(delay_ms=0)
    queue_preview_render(delay_ms=0)
    queue_preview_focus_sync(delay_ms=0)
    root.after(250, warmup_synonym_lookup_async)
    text_widget.focus_set()
    root.bind("<Configure>", on_root_configure, add="+")
    root.after_idle(lambda: place_initial_sash(24))
    root.protocol("WM_DELETE_WINDOW", on_quit)
    root.mainloop()
    return 0


def run(
    cfg_path: str,
    input_path: Optional[str],
    output_path: Optional[str],
    as_json: bool,
    diagnose_paragraphs: bool,
    diagnose_top_k: int,
    diagnose_print_text: bool,
    heatmap: bool,
    text_max_tokens_override: Optional[int] = None,
) -> int:
    text = read_text(input_path)
    result, paragraph_profile = analyze_text_document(
        cfg_path=cfg_path,
        text=text,
        input_label=input_path if input_path else "<stdin>",
        diagnose_paragraphs=diagnose_paragraphs,
        diagnose_top_k=diagnose_top_k,
        need_paragraph_profile=heatmap,
        text_max_tokens_override=text_max_tokens_override,
    )

    if heatmap:
        if as_json:
            raise ValueError("--heatmap cannot be combined with --json.")
        if paragraph_profile is None:
            raise ValueError("Internal error: paragraph profile missing for heatmap generation.")
        heatmap_md = build_heatmap_markdown(
            text=text,
            source_label=input_path if input_path else "<stdin>",
            paragraph_profile=paragraph_profile,
            top_k=diagnose_top_k,
            observer_logppl=result["observer"]["logPPL"],
            observer_ppl=result["observer"]["PPL"],
            performer_logppl=result["performer"]["logPPL"],
            performer_ppl=result["performer"]["PPL"],
            logxppl=result["cross"]["logXPPL"],
            xppl=result["cross"]["XPPL"],
            binoculars_score=result["binoculars"]["score"],
        )
        heatmap_console = build_heatmap_output_console(
            text=text,
            source_label=input_path if input_path else "<stdin>",
            paragraph_profile=paragraph_profile,
            top_k=diagnose_top_k,
            observer_logppl=result["observer"]["logPPL"],
            observer_ppl=result["observer"]["PPL"],
            performer_logppl=result["performer"]["logPPL"],
            performer_ppl=result["performer"]["PPL"],
            logxppl=result["cross"]["logXPPL"],
            xppl=result["cross"]["XPPL"],
            binoculars_score=result["binoculars"]["score"],
        )
        heatmap_path = infer_heatmap_output_path(input_path)
        backup_path = backup_existing_file(heatmap_path)
        if backup_path is not None:
            print(f"[heatmap] backed up {heatmap_path} -> {backup_path}", file=sys.stderr)
        with open(heatmap_path, "w", encoding="utf-8") as f:
            f.write(heatmap_md)
        print(heatmap_console)
        print(f"[heatmap] wrote {heatmap_path}", file=sys.stderr)
        return 0

    out_str = json_dump(result) if as_json else (
        f"Tokens: {result['input']['tokens']} (transitions={result['input']['transitions']})\n"
        f"Observer logPPL: {result['observer']['logPPL']:.6f}  PPL: {result['observer']['PPL']:.3f}\n"
        f"Performer logPPL: {result['performer']['logPPL']:.6f}  PPL: {result['performer']['PPL']:.3f}\n"
        f"Cross logXPPL: {result['cross']['logXPPL']:.6f}  XPPL: {result['cross']['XPPL']:.3f}\n"
        f"Binoculars score B (high is more human-like): {result['binoculars']['score']:.6f}\n"
    )
    if (not as_json) and ("diagnostics" in result):
        hotspots = result["diagnostics"]["low_perplexity_spans"]["top_low_perplexity_hotspots"]
        out_str += "Top low-perplexity hotspots (paragraph-level):\n"
        for row in hotspots:
            out_str += (
                f"  - p{row['paragraph_id']}: logPPL={row['logPPL']:.6f}, "
                f"delta_if_removed={row['delta_doc_logPPL_if_removed']:+.6f}, "
                f"transitions={row['transitions']}, excerpt={row['excerpt']}\n"
            )
        if diagnose_print_text:
            out_str += "\nHotspot full text segments:\n"
            for row in hotspots:
                seg = text[row["char_start"] : row["char_end"]]
                out_str += (
                    f"\n--- p{row['paragraph_id']} "
                    f"(chars {row['char_start']}:{row['char_end']}) ---\n"
                    f"{seg}\n"
                )

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(out_str)
    else:
        print(out_str)

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Binoculars-style scoring for markdown text using two llama.cpp models loaded sequentially.",
        usage=(
            "%(prog)s [--master-config FILE] [--config PROFILE] [--gui INPUT] [--input INPUT | INPUT] "
            "[--output OUTPUT] [--json] "
            "[--diagnose-paragraphs] [--diagnose-top-k N] [--diagnose-print-text] [--heatmap]"
        ),
        epilog=(
            "Examples:\n"
            "  %(prog)s --config fast samples/Athens.md\n"
            "  %(prog)s --config=long --input samples/Athens.md --heatmap\n"
            "  %(prog)s --config fast --gui samples/Athens.md\n"
            "  %(prog)s samples/Athens.md   # uses default profile from config.binoculars.json"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument(
        "--master-config",
        default=default_master_config_path(),
        help="Path to master config mapping profile labels to concrete config files.",
    )
    ap.add_argument(
        "--config",
        default=None,
        help="Config profile label from master config (for example: fast or long).",
    )
    ap.add_argument(
        "--gui",
        default=None,
        help="Open interactive GUI editor/analyzer for the provided markdown file.",
    )
    ap.add_argument("--input", default=None, help="Markdown file to score, or '-' for stdin.")
    ap.add_argument(
        "input_positional",
        nargs="*",
        default=[],
        help="Positional input file(s); if provided, the first path is used as input.",
    )
    ap.add_argument("--output", default=None, help="Optional path to write output.")
    ap.add_argument("--json", action="store_true", help="Output full results as JSON.")
    ap.add_argument(
        "--diagnose-paragraphs",
        action="store_true",
        help="Emit paragraph-level hotspots that disproportionately lower observer logPPL.",
    )
    ap.add_argument(
        "--diagnose-top-k",
        type=int,
        default=10,
        help="Number of hotspot paragraphs to include when --diagnose-paragraphs is set.",
    )
    ap.add_argument(
        "--diagnose-print-text",
        action="store_true",
        help="When diagnosing paragraphs, print full hotspot segment text to console (non-JSON mode).",
    )
    ap.add_argument(
        "--heatmap",
        action="store_true",
        help=(
            "Output markdown heatmap with lowest perplexity sections in red and highest in green; "
            "prints to console and writes '<input_stem>_heatmap.md'."
        ),
    )
    args = ap.parse_args()

    if args.gui is not None:
        print("Launching GUI...", file=sys.stderr)
    else:
        print("One moment please, starting...", file=sys.stderr)

    try:
        _, cfg_path, text_max_tokens_override = resolve_profile_config(args.master_config, args.config)

        if args.gui is not None:
            if args.input is not None or args.input_positional:
                raise ValueError("When using --gui, provide the input file only via --gui.")
            if args.output is not None or args.json or args.heatmap:
                raise ValueError("--gui cannot be combined with --output, --json, or --heatmap.")
            return launch_gui(
                cfg_path=cfg_path,
                gui_input_path=args.gui,
                top_k=args.diagnose_top_k,
                text_max_tokens_override=text_max_tokens_override,
            )

        if args.input is not None and args.input_positional:
            raise ValueError("Provide input either as --input or positional INPUT, not both.")

        if args.input is not None:
            input_path = args.input
        elif args.input_positional:
            input_path = args.input_positional[0]
            if len(args.input_positional) > 1:
                print(
                    f"WARNING: multiple positional inputs provided; using first: {input_path}",
                    file=sys.stderr,
                )
        else:
            input_path = "-"

        return run(
            cfg_path=cfg_path,
            input_path=input_path,
            output_path=args.output,
            as_json=args.json,
            diagnose_paragraphs=args.diagnose_paragraphs,
            diagnose_top_k=args.diagnose_top_k,
            diagnose_print_text=args.diagnose_print_text,
            heatmap=args.heatmap,
            text_max_tokens_override=text_max_tokens_override,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
