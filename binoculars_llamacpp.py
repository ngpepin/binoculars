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
import gc
import json
import os
import sys
import tempfile
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


def compute_logppl_from_scores(scores: np.ndarray, tokens: List[int]) -> Tuple[float, float]:
    """
    Compute observer logPPL:
      logPPL = - mean_{i=0..N-2} log p(token[i+1] | prefix up to i)
    Returns: (logPPL, ppl)
    """
    if len(tokens) < 2:
        raise ValueError("Need at least 2 tokens to compute perplexity (a transition).")

    n = len(tokens)
    total_logp = 0.0
    count = 0

    # scores[i] predicts token[i+1]
    for i in range(n - 1):
        logits = scores[i].astype(np.float64, copy=False)
        logZ = logsumexp_1d(logits)
        nxt = tokens[i + 1]
        logp = float(logits[nxt]) - logZ
        total_logp += logp
        count += 1

    logppl = -total_logp / max(count, 1)
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
        l1 = np.array(logits1_mm[i], dtype=np.float64, copy=False)
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


# ----------------------------
# Main pipeline
# ----------------------------

def run(cfg_path: str, input_path: Optional[str], output_path: Optional[str], as_json: bool) -> int:
    cfg, tcfg, ccfg = load_config(cfg_path)

    observer_section = cfg["observer"]
    performer_section = cfg["performer"]

    obs_path = observer_section.get("model_path")
    perf_path = performer_section.get("model_path")
    if not obs_path or not perf_path:
        raise ValueError("observer.model_path and performer.model_path are required.")

    text = read_text(input_path)
    text_bytes = text.encode("utf-8", errors="replace")

    # Tokenize with vocab_only model(s) to decide n_ctx and ensure tokenizer alignment
    tokens_obs = tokenize_with_vocab_only(obs_path, text_bytes, tcfg)
    tokens_obs = maybe_truncate_tokens(tokens_obs, tcfg.max_tokens)

    tokens_perf = tokenize_with_vocab_only(perf_path, text_bytes, tcfg)
    tokens_perf = maybe_truncate_tokens(tokens_perf, tcfg.max_tokens)

    if tokens_obs != tokens_perf:
        # Strict behavior: refuse, because logits are position/token dependent.
        # You can relax this by truncating to the common prefix length, but it's safer to fail.
        # If you want prefix mode, implement common-prefix truncation here.
        raise ValueError(
            "Tokenizer mismatch: the two models do not tokenize the input identically. "
            "Use two models from the same family/tokenizer (e.g., base + instruct sibling)."
        )

    tokens = tokens_obs
    if len(tokens) < 2:
        raise ValueError("Text is too short after tokenization (need at least 2 tokens).")

    needed_ctx = len(tokens)

    # Cache directory
    if ccfg.dir:
        cache_dir = ccfg.dir
        ensure_dir(cache_dir)
        temp_dir_ctx = None
    else:
        temp_dir_ctx = tempfile.TemporaryDirectory(prefix="binoculars_cache_")
        cache_dir = temp_dir_ctx.name

    cache_dtype = np.float16 if ccfg.dtype == "float16" else np.float32
    logits1_path = os.path.join(cache_dir, "observer_logits.dat")

    # ----------------------------
    # 1) Load observer (M1), eval, compute logPPL, save logits to disk, unload
    # ----------------------------
    obs = None
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
        scores1 = obs.scores[:needed_ctx, :n_vocab_obs]  # view
        # Compute logPPL from scores1 and tokens
        logppl_obs, ppl_obs = compute_logppl_from_scores(scores1, tokens)

        # Save logits for positions 0..needed_ctx-2 (we only use up to N-2)
        # (scores1 row i predicts token[i+1], last row unused for scoring)
        save_arr = scores1[:needed_ctx - 1, :n_vocab_obs]
        save_logits_memmap(logits1_path, save_arr, cache_dtype)

    finally:
        close_llama(obs)
        del obs
        gc.collect()

    # ----------------------------
    # 2) Load performer (M2), eval, compute logXPPL using saved M1 logits, unload
    # ----------------------------
    perf = None
    try:
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

        scores2 = perf.scores[:needed_ctx, :n_vocab_perf]  # view

        # Optional: performer perplexity (informational)
        logppl_perf, ppl_perf = compute_logppl_from_scores(scores2, tokens)

        # Load observer logits memmap and compute cross logXPPL
        logits1_mm = load_logits_memmap(
            logits1_path,
            shape=(needed_ctx - 1, n_vocab_obs),
            dtype=cache_dtype,
        )
        logxppl, xppl = compute_cross_logxppl(logits1_mm, scores2, tokens)
        del logits1_mm

        # Binoculars ratio
        B = float(logppl_obs / logxppl) if logxppl != 0.0 else float("inf")

        result = {
            "input": {
                "path": input_path if input_path else "<stdin>",
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

    finally:
        close_llama(perf)
        del perf
        gc.collect()

        # Clean cache unless told to keep
        if not ccfg.keep:
            try:
                if os.path.exists(logits1_path):
                    os.remove(logits1_path)
            except Exception:
                pass
            if temp_dir_ctx is not None:
                temp_dir_ctx.cleanup()

    out_str = json_dump(result) if as_json else (
        f"Tokens: {result['input']['tokens']} (transitions={result['input']['transitions']})\n"
        f"Observer logPPL: {result['observer']['logPPL']:.6f}  PPL: {result['observer']['PPL']:.3f}\n"
        f"Performer logPPL: {result['performer']['logPPL']:.6f}  PPL: {result['performer']['PPL']:.3f}\n"
        f"Cross logXPPL: {result['cross']['logXPPL']:.6f}  XPPL: {result['cross']['XPPL']:.3f}\n"
        f"Binoculars score B: {result['binoculars']['score']:.6f}\n"
    )

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(out_str)
    else:
        print(out_str)

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Binoculars-style scoring for markdown text using two llama.cpp models loaded sequentially."
    )
    ap.add_argument("--config", required=True, help="Path to JSON configuration file.")
    ap.add_argument("--input", default="-", help="Markdown file to score, or '-' for stdin.")
    ap.add_argument("--output", default=None, help="Optional path to write output.")
    ap.add_argument("--json", action="store_true", help="Output full results as JSON.")
    args = ap.parse_args()

    try:
        return run(
            cfg_path=args.config,
            input_path=args.input,
            output_path=args.output,
            as_json=args.json,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
