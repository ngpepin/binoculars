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
from datetime import datetime
import gc
import json
import os
import re
import shutil
import sys
import tempfile
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Set

import numpy as np
from llama_cpp import Llama


# ----------------------------
# Utilities
# ----------------------------

SPELL_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*(?:[-/][A-Za-z]+(?:['’][A-Za-z]+)*)*")
_ENGLISH_WORDLIST_PATHS = [
    "/usr/share/dict/american-english",
    "/usr/share/dict/british-english",
    "/usr/share/dict/words",
    "/usr/dict/words",
]
_ENGLISH_WORDS_CACHE: Optional[Set[str]] = None


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


def default_master_config_path() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "config.binoculars.json")


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


def _normalize_console_text(text: str) -> str:
    # Normalize markdown hard-break markers for cleaner terminal output.
    # Handles both backslash+newline and backslash+spaces patterns seen in prose/dialogue.
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\\[ \t]*\n", "\n", t)
    t = re.sub(r"(?<=\S)\\(?=\s)", "", t)
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


def launch_gui(
    cfg_path: str,
    gui_input_path: str,
    top_k: int,
    text_max_tokens_override: Optional[int] = None,
) -> int:
    try:
        import tkinter as tk
        import tkinter.font as tkfont
        from tkinter import messagebox
    except Exception as exc:
        raise ValueError("Tkinter is required for --gui mode but could not be imported.") from exc

    src_path = os.path.abspath(gui_input_path)
    if not os.path.isfile(src_path):
        raise ValueError(f"--gui file not found: {src_path}")

    with open(src_path, "r", encoding="utf-8") as f:
        initial_text = f.read()

    try:
        root = tk.Tk()
    except Exception as exc:
        raise ValueError("Unable to start GUI window. Check your display environment.") from exc

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

    root.title(f"Binoculars Editor - {os.path.basename(src_path)}")
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
        bg="#101010",
        fg="#d9d9d9",
        padx=8,
        pady=6,
    )
    status_label.pack(side="bottom", fill="x")

    line_numbers = tk.Text(
        editor_frame,
        width=6,
        wrap="none",
        state="disabled",
        bg="#0a0a0a",
        fg="#7d7d7d",
        relief="flat",
        padx=6,
        pady=12,
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

    preview_text = tk.Text(
        preview_frame,
        wrap="word",
        state="disabled",
        bg="#2a2a2a",
        fg="#e4e4e4",
        insertbackground="#e4e4e4",
        font=(preview_font_family, preview_font_size),
        relief="flat",
        padx=12,
        pady=12,
    )
    preview_scroll = tk.Scrollbar(preview_frame, orient="vertical", command=preview_text.yview)
    preview_text.configure(yscrollcommand=preview_scroll.set)
    preview_scroll.pack(side="right", fill="y")
    preview_text.pack(side="left", fill="both", expand=True)

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
    preview_text.tag_configure("preview_active_line", background="#3d3d3d")

    text_widget.tag_configure("edited", foreground="#ffd54f")
    text_widget.tag_configure("misspelled", underline=1, underlinefg="#ff4d4d")
    text_widget.tag_configure("unscored", foreground="#c3c3c3")
    english_words = load_english_words()

    state: Dict[str, Any] = {
        "baseline_text": initial_text,
        "prev_text": initial_text,
        "segment_tags": [],
        "segment_labels": {},
        "prior_bg_tags": [],
        "prior_counter": 0,
        "last_b_score": None,
        "spell_version": 0,
        "last_spell_spans": None,
        "tooltip": None,
        "pending_edit_job": None,
        "pending_line_numbers_job": None,
        "pending_spell_job": None,
        "pending_preview_job": None,
        "pending_focus_job": None,
        "preview_line_map": {},
        "analyze_cursor_idx": "1.0",
        "analyze_yview_top": 0.0,
        "line_count": 0,
        "internal_update": False,
        "analyzing": False,
        "progress_popup": None,
    }

    def current_text() -> str:
        return text_widget.get("1.0", "end-1c")

    def refresh_line_numbers_now() -> None:
        state["pending_line_numbers_job"] = None
        try:
            line_count = max(1, int(text_widget.index("end-1c").split(".")[0]))
        except Exception:
            line_count = 1

        if int(state.get("line_count", 0)) != line_count:
            gutter_width = max(4, len(str(line_count)) + 1)
            line_numbers.configure(state="normal", width=gutter_width)
            line_numbers.delete("1.0", "end")
            line_numbers.insert("1.0", "\n".join(str(i) for i in range(1, line_count + 1)))
            line_numbers.configure(state="disabled")
            state["line_count"] = line_count

        try:
            first = text_widget.yview()[0]
            line_numbers.yview_moveto(first)
        except Exception:
            pass

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
        raw = current_text().replace("\r\n", "\n").replace("\r", "\n")

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

    def clear_preview_segment_backgrounds() -> None:
        preview_text.tag_remove("preview_heat_low", "1.0", "end")
        preview_text.tag_remove("preview_heat_high", "1.0", "end")

    def apply_preview_segment_backgrounds() -> None:
        # Keep the preview unadorned except for the currently active line.
        clear_preview_segment_backgrounds()

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

    def sync_preview_focus_now() -> None:
        state["pending_focus_job"] = None

        try:
            src_line = int(text_widget.index("insert").split(".")[0])
        except Exception:
            src_line = 1

        map_obj = state.get("preview_line_map")
        preview_line_map = map_obj if isinstance(map_obj, dict) else {}

        try:
            preview_total = max(1, int(preview_text.index("end-1c").split(".")[0]))
        except Exception:
            preview_total = 1

        mapped = preview_line_map.get(src_line, 0)
        try:
            preview_line = int(mapped)
        except Exception:
            preview_line = 0

        if preview_line <= 0:
            try:
                src_total = max(1, int(text_widget.index("end-1c").split(".")[0]))
            except Exception:
                src_total = 1
            frac = (max(src_line, 1) - 1) / max(src_total - 1, 1)
            preview_line = int(round(frac * max(preview_total - 1, 0))) + 1

        preview_line = max(1, min(preview_total, preview_line))
        preview_text.tag_remove("preview_active_line", "1.0", "end")
        clear_preview_segment_backgrounds()
        label = line_heat_label(src_line)
        if label == "LOW":
            preview_text.tag_add("preview_heat_low", f"{preview_line}.0", f"{preview_line}.0 lineend+1c")
            preview_text.tag_configure("preview_active_line", background="#4a3434")
        elif label == "HIGH":
            preview_text.tag_add("preview_heat_high", f"{preview_line}.0", f"{preview_line}.0 lineend+1c")
            preview_text.tag_configure("preview_active_line", background="#31483a")
        else:
            preview_text.tag_configure("preview_active_line", background="#3d3d3d")
        preview_text.tag_add(
            "preview_active_line",
            f"{preview_line}.0",
            f"{preview_line}.0 lineend+1c",
        )
        preview_text.tag_raise("preview_heat_low")
        preview_text.tag_raise("preview_heat_high")
        preview_text.tag_raise("preview_active_line")

        left_info = text_widget.dlineinfo("insert")
        if left_info is not None and text_widget.winfo_height() > 1:
            left_ratio = float(left_info[1]) / float(max(text_widget.winfo_height() - 1, 1))
        else:
            left_ratio = 0.35
        left_ratio = max(0.0, min(0.95, left_ratio))

        top, bottom = preview_text.yview()
        visible = bottom - top
        if visible <= 0.0:
            visible = min(1.0, 30.0 / float(max(preview_total, 1)))
        line_frac = (preview_line - 1) / float(max(preview_total - 1, 1))
        top_target = line_frac - (left_ratio * visible)
        max_top = max(0.0, 1.0 - visible)
        top_target = max(0.0, min(max_top, top_target))
        preview_text.yview_moveto(top_target)

    def queue_preview_focus_sync(delay_ms: int = 30) -> None:
        pending = state.get("pending_focus_job")
        if pending is not None:
            try:
                root.after_cancel(pending)
            except Exception:
                pass
        state["pending_focus_job"] = root.after(delay_ms, sync_preview_focus_now)

    def on_editor_scroll(first: str, last: str) -> None:
        scroll_y.set(first, last)
        try:
            line_numbers.yview_moveto(float(first))
        except Exception:
            pass
        queue_preview_focus_sync(delay_ms=0)

    def tooltip_text(info: Dict[str, Any]) -> str:
        pct = info.get("pct_contribution", float("nan"))
        pct_str = f"{pct:+.2f}%" if np.isfinite(pct) else "n/a"
        return (
            f"Index: {info['note_num']}\n"
            f"Label: {info['label']}\n"
            f"% contribution: {pct_str}\n"
            f"Paragraph: {info['paragraph_id']}\n"
            f"logPPL: {info['logPPL']:.6f}\n"
            f"delta_vs_doc: {info['delta_vs_doc_logPPL']:+.6f}\n"
            f"delta_if_removed: {info['delta_doc_logPPL_if_removed']:+.6f}\n"
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
        text_widget.tag_remove("unscored", "1.0", "end")
        clear_preview_segment_backgrounds()

    def clear_prior_backgrounds() -> None:
        for tag in state["prior_bg_tags"]:
            try:
                text_widget.tag_remove(tag, "1.0", "end")
                text_widget.tag_delete(tag)
            except Exception:
                pass
        state["prior_bg_tags"] = []

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

    def apply_heatmap_profile(profile: Dict[str, Any], observer_logppl: float) -> None:
        clear_current_segment_tags()
        rows = list(profile.get("rows", []))
        if rows:
            _, _, annotations, _endnotes = _prepare_heatmap_annotations(
                rows=rows,
                top_k=top_k,
                observer_logppl=observer_logppl,
            )
            for row in sorted(rows, key=lambda r: r["char_start"]):
                ann = annotations.get(row["paragraph_id"])
                if ann is None:
                    continue
                tag = f"heat_seg_{ann['note_num']}"
                start = f"1.0+{row['char_start']}c"
                end = f"1.0+{row['char_end']}c"
                color = "#ff4d4d" if ann["label"] == "LOW" else "#39d97f"
                text_widget.tag_configure(tag, foreground=color)
                text_widget.tag_add(tag, start, end)
                text_widget.tag_bind(tag, "<Enter>", lambda e, info=dict(ann): show_tooltip(e, info))
                text_widget.tag_bind(tag, "<Motion>", move_tooltip)
                text_widget.tag_bind(tag, "<Leave>", lambda _e: hide_tooltip())
                state["segment_tags"].append(tag)
                state["segment_labels"][tag] = ann["label"]

        analyzed_char_end_raw = profile.get("analyzed_char_end")
        if analyzed_char_end_raw is not None:
            try:
                analyzed_char_end = int(analyzed_char_end_raw)
            except Exception:
                analyzed_char_end = len(current_text())
            doc_len = len(current_text())
            analyzed_char_end = max(0, min(doc_len, analyzed_char_end))
            if analyzed_char_end < doc_len:
                text_widget.tag_add("unscored", f"1.0+{analyzed_char_end}c", "end-1c")
                text_widget.tag_raise("unscored")

        text_widget.tag_raise("misspelled")
        text_widget.tag_raise("edited")
        apply_preview_segment_backgrounds()

    def set_controls(enabled: bool) -> None:
        btn_state = "normal" if enabled else "disabled"
        analyze_btn.configure(state=btn_state)
        save_btn.configure(state=btn_state)
        clear_priors_btn.configure(state=btn_state)
        quit_btn.configure(state=btn_state)
        text_widget.configure(state="normal" if enabled else "disabled")

    def show_progress_popup() -> None:
        popup = tk.Toplevel(root)
        popup.title("Analyzing")
        popup.transient(root)
        popup.resizable(False, False)
        popup.grab_set()
        msg = tk.Label(
            popup,
            text="The text is being analyzed by the binoculars...",
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

    def finish_analysis_error(message: str) -> None:
        close_progress_popup()
        state["analyzing"] = False
        set_controls(True)
        status_var.set(f"Analyze failed: {message}")
        messagebox.showerror("Analyze Error", message)

    def finish_analysis_success(
        analyzed_text: str,
        result: Dict[str, Any],
        profile: Optional[Dict[str, Any]],
    ) -> None:
        close_progress_popup()
        state["internal_update"] = True
        try:
            text_widget.tag_remove("edited", "1.0", "end")
            if profile is not None:
                apply_heatmap_profile(profile, result["observer"]["logPPL"])
            state["baseline_text"] = analyzed_text
            state["prev_text"] = analyzed_text
            text_widget.edit_modified(False)
        finally:
            state["internal_update"] = False

        prior_b = state.get("last_b_score")
        prior_b_text = f"{prior_b:.6f}" if isinstance(prior_b, (float, int)) else "n/a"
        analyzed_char_end = len(analyzed_text)
        if isinstance(profile, dict):
            raw_end = profile.get("analyzed_char_end")
            if raw_end is not None:
                try:
                    analyzed_char_end = int(raw_end)
                except Exception:
                    analyzed_char_end = len(analyzed_text)
        analyzed_char_end = max(0, min(len(analyzed_text), analyzed_char_end))
        last_line = line_number_for_char_end(analyzed_text, analyzed_char_end)
        status_var.set(
            "Binocular score B (high is more human-like): "
            f"{result['binoculars']['score']:.6f} [prior: {prior_b_text}] | "
            f"Last Line: {last_line} | "
            f"Observer logPPL: {result['observer']['logPPL']:.6f} | "
            f"Performer logPPL: {result['performer']['logPPL']:.6f} | "
            f"Cross logXPPL: {result['cross']['logXPPL']:.6f}"
        )
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
        queue_preview_focus_sync(delay_ms=0)

    def on_analyze() -> None:
        if state["analyzing"]:
            return

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

        snapshot_current_to_priors()
        analyzed_text = current_text()
        cursor_idx = text_widget.index("insert")
        yview = text_widget.yview()
        yview_top = float(yview[0]) if yview else 0.0
        state["analyze_cursor_idx"] = cursor_idx
        state["analyze_yview_top"] = yview_top
        state["analyzing"] = True
        status_var.set("Analyzing current text...")
        set_controls(False)
        show_progress_popup()

        def worker() -> None:
            try:
                result, profile = analyze_text_document(
                    cfg_path=cfg_path,
                    text=analyzed_text,
                    input_label=src_path,
                    diagnose_paragraphs=False,
                    diagnose_top_k=top_k,
                    need_paragraph_profile=True,
                    text_max_tokens_override=text_max_tokens_override,
                )
            except Exception as exc:
                root.after(0, lambda: finish_analysis_error(str(exc)))
                return
            root.after(0, lambda: finish_analysis_success(analyzed_text, result, profile))

        threading.Thread(target=worker, daemon=True).start()

    def on_save() -> None:
        content = current_text()
        src_dir = os.path.dirname(src_path) or "."
        stem, _ext = os.path.splitext(os.path.basename(src_path))
        stamp = datetime.now().strftime("%Y%m%d%H%M")
        out_path = os.path.join(src_dir, f"{stem}_edited_{stamp}.md")
        idx = 1
        while os.path.exists(out_path):
            idx += 1
            out_path = os.path.join(src_dir, f"{stem}_edited_{stamp}_{idx}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        status_var.set(f"Saved edited file: {out_path}")

    def on_clear_priors() -> None:
        clear_prior_backgrounds()
        status_var.set("Cleared prior background highlights.")

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
        hide_tooltip()
        close_progress_popup()
        root.destroy()

    analyze_btn = tk.Button(toolbar, text="Analyze", command=on_analyze, width=12)
    save_btn = tk.Button(toolbar, text="Save", command=on_save, width=12)
    clear_priors_btn = tk.Button(toolbar, text="Clear Priors", command=on_clear_priors, width=12)
    quit_btn = tk.Button(toolbar, text="Quit", command=on_quit, width=12)
    analyze_btn.pack(side="left", padx=(0, 8))
    save_btn.pack(side="left", padx=(0, 8))
    clear_priors_btn.pack(side="left", padx=(0, 8))
    quit_btn.pack(side="left")

    state["internal_update"] = True
    text_widget.insert("1.0", initial_text)
    text_widget.edit_modified(False)
    state["internal_update"] = False
    refresh_line_numbers_now()
    text_widget.configure(yscrollcommand=on_editor_scroll)
    text_widget.bind("<<Modified>>", on_modified)
    text_widget.bind("<KeyRelease>", lambda _e: queue_preview_focus_sync(delay_ms=0), add="+")
    text_widget.bind("<ButtonRelease-1>", lambda _e: queue_preview_focus_sync(delay_ms=0), add="+")
    text_widget.bind("<MouseWheel>", lambda _e: queue_preview_focus_sync(delay_ms=0), add="+")
    text_widget.bind("<Button-4>", lambda _e: queue_preview_focus_sync(delay_ms=0), add="+")
    text_widget.bind("<Button-5>", lambda _e: queue_preview_focus_sync(delay_ms=0), add="+")
    text_widget.mark_set("insert", "1.0")
    text_widget.see("1.0")
    state["spell_version"] = int(state.get("spell_version", 0)) + 1
    queue_spellcheck(delay_ms=0)
    queue_preview_render(delay_ms=0)
    queue_preview_focus_sync(delay_ms=0)
    text_widget.focus_set()
    root.after(
        50,
        lambda: split_pane.sash_place(0, int(root.winfo_width() * 0.67), 0)
        if root.winfo_width() > 0
        else None,
    )
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
