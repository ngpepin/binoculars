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
import difflib
import gc
import json
import os
import re
import shutil
import sys
import tempfile
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
from llama_cpp import Llama


# ----------------------------
# Utilities
# ----------------------------

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


def load_master_config(path: str) -> Tuple[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Master config must be a JSON object.")

    default_label = str(cfg.get("default", "")).strip()
    profiles_raw = cfg.get("profiles")
    if not isinstance(profiles_raw, dict) or not profiles_raw:
        raise ValueError("Master config must contain a non-empty 'profiles' object.")

    profiles: Dict[str, str] = {}
    for label, p in profiles_raw.items():
        key = str(label).strip()
        val = str(p).strip()
        if key and val:
            profiles[key] = val

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


def resolve_profile_config_path(master_cfg_path: str, profile_label: Optional[str]) -> Tuple[str, str]:
    if not os.path.isfile(master_cfg_path):
        raise ValueError(f"Master config file not found: {master_cfg_path}")

    default_label, profiles = load_master_config(master_cfg_path)
    label = (profile_label or "").strip() or default_label
    if label not in profiles:
        raise ValueError(
            f"Unknown --config profile '{label}'. Available profiles: {', '.join(sorted(profiles.keys()))}"
        )

    cfg_path = profiles[label]
    if not os.path.isfile(cfg_path):
        raise ValueError(f"Config profile '{label}' points to missing file: {cfg_path}")

    return label, cfg_path


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
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    cfg, tcfg, ccfg = load_config(cfg_path)

    observer_section = cfg["observer"]
    performer_section = cfg["performer"]

    obs_path = observer_section.get("model_path")
    perf_path = performer_section.get("model_path")
    if not obs_path or not perf_path:
        raise ValueError("observer.model_path and performer.model_path are required.")

    text_bytes = text.encode("utf-8", errors="replace")

    # Tokenize with vocab_only model(s) to decide n_ctx and ensure tokenizer alignment.
    tokens_obs = tokenize_with_vocab_only(obs_path, text_bytes, tcfg)
    tokens_obs = maybe_truncate_tokens(tokens_obs, tcfg.max_tokens)

    tokens_perf = tokenize_with_vocab_only(perf_path, text_bytes, tcfg)
    tokens_perf = maybe_truncate_tokens(tokens_perf, tcfg.max_tokens)

    if tokens_obs != tokens_perf:
        raise ValueError(
            "Tokenizer mismatch: the two models do not tokenize the input identically. "
            "Use two models from the same family/tokenizer (e.g., base + instruct sibling)."
        )

    tokens = tokens_obs
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
                "transitions": len(tokens) - 1,
                "markdown_preserved": True,
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


def launch_gui(cfg_path: str, gui_input_path: str, top_k: int) -> int:
    try:
        import tkinter as tk
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

    root.title(f"Binoculars Editor - {os.path.basename(src_path)}")
    root.geometry("1200x800")
    root.configure(bg="#141414")

    toolbar = tk.Frame(root, bg="#141414")
    toolbar.pack(side="top", fill="x", padx=10, pady=8)

    editor_frame = tk.Frame(root, bg="#141414")
    editor_frame.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 8))

    status_var = tk.StringVar(value="Ready. Press Analyze to score and highlight this document.")
    status_label = tk.Label(
        root,
        textvariable=status_var,
        anchor="w",
        bg="#1f1f1f",
        fg="#d9d9d9",
        padx=8,
        pady=6,
    )
    status_label.pack(side="bottom", fill="x")

    text_widget = tk.Text(
        editor_frame,
        wrap="word",
        undo=True,
        bg="#101010",
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

    text_widget.tag_configure("edited", foreground="#ffd54f")

    state: Dict[str, Any] = {
        "baseline_text": initial_text,
        "segment_tags": [],
        "tooltip": None,
        "pending_edit_job": None,
        "internal_update": False,
        "analyzing": False,
        "progress_popup": None,
    }

    def current_text() -> str:
        return text_widget.get("1.0", "end-1c")

    def hide_tooltip() -> None:
        tip = state.get("tooltip")
        if tip is not None:
            try:
                tip.destroy()
            except Exception:
                pass
            state["tooltip"] = None

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

    def clear_segment_tags() -> None:
        hide_tooltip()
        for tag in state["segment_tags"]:
            try:
                text_widget.tag_remove(tag, "1.0", "end")
                text_widget.tag_delete(tag)
            except Exception:
                pass
        state["segment_tags"] = []

    def apply_edited_diff() -> None:
        state["pending_edit_job"] = None
        if state["internal_update"]:
            return
        baseline = state["baseline_text"]
        curr = current_text()
        text_widget.tag_remove("edited", "1.0", "end")
        if baseline == curr:
            return
        matcher = difflib.SequenceMatcher(a=baseline, b=curr, autojunk=False)
        for op, _i1, _i2, j1, j2 in matcher.get_opcodes():
            if op in {"replace", "insert"} and j2 > j1:
                text_widget.tag_add("edited", f"1.0+{j1}c", f"1.0+{j2}c")
        text_widget.tag_raise("edited")

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
        state["pending_edit_job"] = root.after(80, apply_edited_diff)

    def apply_heatmap_profile(profile: Dict[str, Any], observer_logppl: float) -> None:
        clear_segment_tags()
        rows = list(profile.get("rows", []))
        if not rows:
            return
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
            color = "#ff6b6b" if ann["label"] == "LOW" else "#39d97f"
            text_widget.tag_configure(tag, foreground=color)
            text_widget.tag_add(tag, start, end)
            text_widget.tag_bind(tag, "<Enter>", lambda e, info=dict(ann): show_tooltip(e, info))
            text_widget.tag_bind(tag, "<Motion>", move_tooltip)
            text_widget.tag_bind(tag, "<Leave>", lambda _e: hide_tooltip())
            state["segment_tags"].append(tag)
        text_widget.tag_raise("edited")

    def set_controls(enabled: bool) -> None:
        btn_state = "normal" if enabled else "disabled"
        analyze_btn.configure(state=btn_state)
        save_btn.configure(state=btn_state)
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
        cursor_idx: str,
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
            try:
                text_widget.mark_set("insert", cursor_idx)
                text_widget.see("insert")
            except Exception:
                pass
            text_widget.edit_modified(False)
        finally:
            state["internal_update"] = False

        status_var.set(
            "Binocular score B (high is more human-like): "
            f"{result['binoculars']['score']:.6f} | "
            f"Observer logPPL: {result['observer']['logPPL']:.6f} | "
            f"Performer logPPL: {result['performer']['logPPL']:.6f} | "
            f"Cross logXPPL: {result['cross']['logXPPL']:.6f}"
        )
        state["analyzing"] = False
        set_controls(True)

    def on_analyze() -> None:
        if state["analyzing"]:
            return
        analyzed_text = current_text()
        cursor_idx = text_widget.index("insert")
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
                )
            except Exception as exc:
                root.after(0, lambda: finish_analysis_error(str(exc)))
                return
            root.after(0, lambda: finish_analysis_success(analyzed_text, cursor_idx, result, profile))

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

    def on_quit() -> None:
        if state.get("pending_edit_job") is not None:
            try:
                root.after_cancel(state["pending_edit_job"])
            except Exception:
                pass
        hide_tooltip()
        close_progress_popup()
        root.destroy()

    analyze_btn = tk.Button(toolbar, text="Analyze", command=on_analyze, width=12)
    save_btn = tk.Button(toolbar, text="Save", command=on_save, width=12)
    quit_btn = tk.Button(toolbar, text="Quit", command=on_quit, width=12)
    analyze_btn.pack(side="left", padx=(0, 8))
    save_btn.pack(side="left", padx=(0, 8))
    quit_btn.pack(side="left")

    state["internal_update"] = True
    text_widget.insert("1.0", initial_text)
    text_widget.edit_modified(False)
    state["internal_update"] = False
    text_widget.bind("<<Modified>>", on_modified)
    text_widget.focus_set()
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
) -> int:
    text = read_text(input_path)
    result, paragraph_profile = analyze_text_document(
        cfg_path=cfg_path,
        text=text,
        input_label=input_path if input_path else "<stdin>",
        diagnose_paragraphs=diagnose_paragraphs,
        diagnose_top_k=diagnose_top_k,
        need_paragraph_profile=heatmap,
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
        _, cfg_path = resolve_profile_config_path(args.master_config, args.config)

        if args.gui is not None:
            if args.input is not None or args.input_positional:
                raise ValueError("When using --gui, provide the input file only via --gui.")
            if args.output is not None or args.json or args.heatmap:
                raise ValueError("--gui cannot be combined with --output, --json, or --heatmap.")
            return launch_gui(cfg_path=cfg_path, gui_input_path=args.gui, top_k=args.diagnose_top_k)

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
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
