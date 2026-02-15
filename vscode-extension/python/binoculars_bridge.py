#!/usr/bin/env python3
"""Persistent JSON bridge for Binoculars analysis/rewrite workflows.

Protocol: line-delimited JSON request/response over stdin/stdout.
Each request must include: {"id": ..., "method": "...", "params": {...}}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import socketserver
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is importable when this script runs from subdirectory.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from binoculars import (  # type: ignore
    analyze_text_document,
    build_llama_instance,
    close_llama,
    compute_logppl_from_scores,
    estimate_char_end_for_token_limit,
    estimate_rewrite_b_impact_options,
    filter_llama_kwargs,
    generate_rewrite_candidates_for_span,
    infer_n_ctx,
    load_config,
    maybe_truncate_tokens,
    tokenize_with_vocab_only,
)


@dataclass
class BridgeState:
    cfg_path: Optional[str] = None
    top_k: int = 5
    text_max_tokens_override: Optional[int] = None
    last_text: str = ""
    last_input_label: str = "<unsaved>"
    next_chunk_start: int = 0
    observer_model_path_override: str = ""
    performer_model_path_override: str = ""
    rewrite_llm_config_path: str = ""
    _runtime_cfg_path: Optional[str] = None


OBSERVER_IDLE_TIMEOUT_SEC = 300.0


class ObserverWarmCache:
    def __init__(self, idle_timeout_sec: float = OBSERVER_IDLE_TIMEOUT_SEC) -> None:
        self.idle_timeout_sec = max(5.0, float(idle_timeout_sec))
        self._lock = threading.Lock()
        self._model: Any = None
        self._signature: str = ""
        self._n_ctx: int = 0
        self._last_used: float = 0.0
        self._in_use: int = 0

    def close_if_idle(self) -> None:
        now = time.monotonic()
        with self._lock:
            if self._model is None:
                return
            if self._in_use > 0:
                return
            if (now - self._last_used) < self.idle_timeout_sec:
                return
            model = self._model
            self._model = None
            self._signature = ""
            self._n_ctx = 0
        try:
            close_llama(model)
        except Exception:
            pass

    def close(self) -> None:
        with self._lock:
            model = self._model
            self._model = None
            self._signature = ""
            self._n_ctx = 0
        if model is None:
            return
        try:
            close_llama(model)
        except Exception:
            pass

    def score_slice(self, state: "BridgeState", full_text: str, start_char: int, end_char: Optional[int] = None) -> Dict[str, Any]:
        cfg_path = _runtime_cfg_path(state)
        cfg, tcfg, _ccfg = load_config(cfg_path)
        if state.text_max_tokens_override is not None:
            tcfg.max_tokens = int(state.text_max_tokens_override)

        observer_section = cfg["observer"]
        obs_path = observer_section.get("model_path")
        if not obs_path:
            raise ValueError("observer.model_path is required.")

        text_len = len(full_text)
        start = max(0, min(text_len, int(start_char)))
        end = text_len if end_char is None else max(start, min(text_len, int(end_char)))
        text = full_text[start:end]
        if not text:
            return {
                "observer_logPPL": float("nan"),
                "transitions": 0,
                "analyzed_char_end": start,
                "truncated_by_limit": False,
            }

        tok_full = tokenize_with_vocab_only(obs_path, text.encode("utf-8", errors="replace"), tcfg)
        tok = maybe_truncate_tokens(tok_full, tcfg.max_tokens)
        if len(tok) < 2:
            raise ValueError("Live estimate text is too short after tokenization.")
        needed_ctx = int(len(tok))
        n_ctx_obs = infer_n_ctx(observer_section, needed_ctx)
        cfg_obj = dict(observer_section)
        cfg_obj["model_path"] = obs_path
        cfg_obj["logits_all"] = True
        cfg_obj.setdefault("verbose", False)
        sig_obj = filter_llama_kwargs(dict(cfg_obj))
        sig_obj.pop("n_ctx", None)
        signature = json.dumps(sig_obj, sort_keys=True, ensure_ascii=True, default=str)

        with self._lock:
            if self._model is None or self._signature != signature or self._n_ctx < n_ctx_obs:
                old = self._model
                self._model = None
                self._signature = ""
                self._n_ctx = 0
                if old is not None:
                    try:
                        close_llama(old)
                    except Exception:
                        pass
                build_cfg = dict(cfg_obj)
                build_cfg["n_ctx"] = n_ctx_obs
                build_cfg = filter_llama_kwargs(build_cfg)
                self._model = build_llama_instance(build_cfg)
                self._signature = signature
                self._n_ctx = int(n_ctx_obs)
            model = self._model
            self._last_used = time.monotonic()
            self._in_use += 1

        assert model is not None
        try:
            model.reset()
            model.eval(tok)
            n_vocab_obs = model.n_vocab()
            scores = model.scores[: len(tok), :n_vocab_obs]
            logppl, _ = compute_logppl_from_scores(scores, tok)
            truncated = len(tok_full) > len(tok)
            analyzed_end_local = len(text)
            if truncated:
                analyzed_end_local = estimate_char_end_for_token_limit(
                    text=text,
                    model=model,
                    tcfg=tcfg,
                    token_limit=len(tok),
                )
            analyzed_end_abs = start + max(0, min(len(text), int(analyzed_end_local)))
            return {
                "observer_logPPL": float(logppl),
                "transitions": int(len(tok) - 1),
                "analyzed_char_end": int(analyzed_end_abs),
                "truncated_by_limit": bool(truncated),
            }
        finally:
            with self._lock:
                self._in_use = max(0, self._in_use - 1)
                self._last_used = time.monotonic()


_ORDERED_LIST_RE = re.compile(r"^([ \t]{0,8})(\d{1,4})([.)])(?:[ \t]+)(.*)$")
_UNORDERED_LIST_RE = re.compile(r"^[ \t]{0,8}(?:[-+*]|(?:\[[ xX]\]))(?:[ \t]+)")


def _preserve_ordered_list_markers(original_text: str, rewrite_text: str) -> Tuple[str, int]:
    original = str(original_text or "")
    rewrite = str(rewrite_text or "")
    if not original or not rewrite:
        return rewrite, 0

    original_lines = original.split("\n")
    rewrite_lines = rewrite.split("\n")
    if not rewrite_lines:
        return rewrite, 0

    # Single-line rewrite case: keep ordered-list marker when model drops it.
    if len(original_lines) == 1:
        m = _ORDERED_LIST_RE.match(original_lines[0])
        if not m:
            return rewrite, 0
        first_rw_idx = None
        for i, ln in enumerate(rewrite_lines):
            if ln.strip():
                first_rw_idx = i
                break
        if first_rw_idx is None:
            return rewrite, 0
        rw_line = rewrite_lines[first_rw_idx]
        if _ORDERED_LIST_RE.match(rw_line) or _UNORDERED_LIST_RE.match(rw_line):
            return rewrite, 0
        indent, num, delim, _tail = m.groups()
        rewrite_lines[first_rw_idx] = f"{indent}{num}{delim} {rw_line.lstrip()}"
        fixed = "\n".join(rewrite_lines)
        trailing_newlines = len(rewrite) - len(rewrite.rstrip("\n"))
        if trailing_newlines > 0:
            fixed = fixed.rstrip("\n") + ("\n" * trailing_newlines)
        return fixed, 1

    if len(original_lines) < 2:
        return rewrite, 0

    original_marker_idx = [i for i, ln in enumerate(original_lines) if _ORDERED_LIST_RE.match(ln)]
    if len(original_marker_idx) < 2:
        return rewrite, 0

    original_nonempty = [ln for ln in original_lines if ln.strip()]
    if not original_nonempty:
        return rewrite, 0
    list_density = len(original_marker_idx) / float(max(1, len(original_nonempty)))
    if list_density < 0.5:
        return rewrite, 0

    rewrite_ordered_count = sum(1 for ln in rewrite_lines if _ORDERED_LIST_RE.match(ln))
    if rewrite_ordered_count == 0 and list_density < 0.8:
        # Do not force list formatting when rewrite intentionally converted structure.
        return rewrite, 0

    out = list(rewrite_lines)
    changed = 0
    pair_n = min(len(original_lines), len(out))
    for i in range(pair_n):
        orig_line = original_lines[i]
        rw_line = out[i]
        m = _ORDERED_LIST_RE.match(orig_line)
        if not m:
            continue
        if not rw_line.strip():
            continue
        if _ORDERED_LIST_RE.match(rw_line):
            continue
        if _UNORDERED_LIST_RE.match(rw_line):
            continue
        indent, num, delim, _tail = m.groups()
        out[i] = f"{indent}{num}{delim} {rw_line.lstrip()}"
        changed += 1

    # Common model failure: first ordered item loses its marker while later items keep theirs.
    if changed == 0 and rewrite_ordered_count > 0:
        first_orig_marker = None
        for ln in original_lines:
            mo = _ORDERED_LIST_RE.match(ln)
            if mo:
                first_orig_marker = mo
                break
        if first_orig_marker is not None:
            first_rw_idx = None
            for i, ln in enumerate(out):
                if ln.strip():
                    first_rw_idx = i
                    break
            if first_rw_idx is not None:
                rw_line = out[first_rw_idx]
                if not _ORDERED_LIST_RE.match(rw_line) and not _UNORDERED_LIST_RE.match(rw_line):
                    indent, num, delim, _tail = first_orig_marker.groups()
                    out[first_rw_idx] = f"{indent}{num}{delim} {rw_line.lstrip()}"
                    changed += 1

    if changed <= 0:
        return rewrite, 0

    fixed = "\n".join(out)
    trailing_newlines = len(rewrite) - len(rewrite.rstrip("\n"))
    if trailing_newlines > 0:
        fixed = fixed.rstrip("\n") + ("\n" * trailing_newlines)
    return fixed, changed


def _cleanup_runtime_cfg(state: BridgeState) -> None:
    runtime = state._runtime_cfg_path
    if not runtime:
        return
    try:
        if os.path.isfile(runtime):
            os.unlink(runtime)
    except Exception:
        pass
    state._runtime_cfg_path = None


def _runtime_cfg_path(state: BridgeState) -> str:
    base_cfg = str(state.cfg_path or "").strip()
    if not base_cfg:
        raise ValueError("cfg_path is required.")

    observer_override = str(state.observer_model_path_override or "").strip()
    performer_override = str(state.performer_model_path_override or "").strip()
    if not observer_override and not performer_override:
        _cleanup_runtime_cfg(state)
        return base_cfg

    with open(base_cfg, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be JSON object.")

    if observer_override:
        observer = cfg.get("observer")
        if not isinstance(observer, dict):
            raise ValueError("Config missing observer section.")
        observer["model_path"] = observer_override
    if performer_override:
        performer = cfg.get("performer")
        if not isinstance(performer, dict):
            raise ValueError("Config missing performer section.")
        performer["model_path"] = performer_override

    fd, temp_path = tempfile.mkstemp(prefix="binoculars-vscode-cfg-", suffix=".json")
    os.close(fd)
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    _cleanup_runtime_cfg(state)
    state._runtime_cfg_path = temp_path
    return temp_path


def _send(msg: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(_json_safe(msg), ensure_ascii=True) + "\n")
    sys.stdout.flush()


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _err(request_id: Any, message: str, code: str = "bridge_error", details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "id": request_id,
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
        },
    }


def _shift_profile(profile: Optional[Dict[str, Any]], base_offset: int) -> Optional[Dict[str, Any]]:
    if profile is None:
        return None
    out = dict(profile)
    rows = []
    for row in profile.get("rows", []):
        new_row = dict(row)
        new_row["char_start"] = int(base_offset + int(row.get("char_start", 0)))
        new_row["char_end"] = int(base_offset + int(row.get("char_end", 0)))
        rows.append(new_row)
    out["rows"] = rows
    if "analyzed_char_end" in out:
        out["analyzed_char_end"] = int(base_offset + int(out.get("analyzed_char_end", 0)))
    return out


def _extract_metrics(analysis: Dict[str, Any]) -> Dict[str, float]:
    return {
        "binoculars_score": float(analysis.get("binoculars", {}).get("score", float("nan"))),
        "observer_logPPL": float(analysis.get("observer", {}).get("logPPL", float("nan"))),
        "performer_logPPL": float(analysis.get("performer", {}).get("logPPL", float("nan"))),
        "cross_logXPPL": float(analysis.get("cross", {}).get("logXPPL", float("nan"))),
        "transitions": float(analysis.get("input", {}).get("transitions", 0)),
    }


def _analyze_slice(
    state: BridgeState,
    full_text: str,
    input_label: str,
    slice_start: int,
    slice_end: int,
) -> Dict[str, Any]:
    bounded_start = max(0, min(len(full_text), int(slice_start)))
    bounded_end = max(bounded_start, min(len(full_text), int(slice_end)))
    slice_text = full_text[bounded_start:bounded_end]
    if not slice_text:
        return {
            "ok": True,
            "analysis": None,
            "paragraph_profile": None,
            "chunk": {
                "char_start": bounded_start,
                "char_end": bounded_end,
                "analyzed_char_end": bounded_start,
                "metrics": {},
            },
        }

    analysis, profile = analyze_text_document(
        cfg_path=_runtime_cfg_path(state),
        text=slice_text,
        input_label=input_label,
        diagnose_paragraphs=False,
        diagnose_top_k=max(1, int(state.top_k)),
        need_paragraph_profile=True,
        text_max_tokens_override=state.text_max_tokens_override,
    )

    shifted_profile = _shift_profile(profile, bounded_start)
    local_analyzed_end = len(slice_text)
    if profile is not None:
        local_analyzed_end = int(profile.get("analyzed_char_end", local_analyzed_end))
    analyzed_char_end = bounded_start + max(0, min(local_analyzed_end, len(slice_text)))

    return {
        "ok": True,
        "analysis": analysis,
        "paragraph_profile": shifted_profile,
        "chunk": {
            "char_start": bounded_start,
            "char_end": bounded_end,
            "analyzed_char_end": analyzed_char_end,
            "metrics": _extract_metrics(analysis),
        },
    }


def _handle_request(
    state: BridgeState,
    request: Dict[str, Any],
    observer_cache: Optional[ObserverWarmCache] = None,
) -> Tuple[Dict[str, Any], bool]:
    request_id = request.get("id")
    method = str(request.get("method", "")).strip()
    params = request.get("params", {})
    if not isinstance(params, dict):
        return _err(request_id, "params must be a JSON object", "invalid_params"), False

    if method == "initialize":
        cfg_path = params.get("cfg_path")
        if cfg_path:
            state.cfg_path = str(cfg_path)
        state.observer_model_path_override = str(params.get("observer_model_path") or "").strip()
        state.performer_model_path_override = str(params.get("performer_model_path") or "").strip()
        state.rewrite_llm_config_path = str(params.get("rewrite_llm_config_path") or "").strip()
        if state.rewrite_llm_config_path:
            os.environ["BINOCULARS_REWRITE_LLM_CONFIG_PATH"] = state.rewrite_llm_config_path
        else:
            os.environ.pop("BINOCULARS_REWRITE_LLM_CONFIG_PATH", None)
        top_k = params.get("top_k")
        if top_k is not None:
            state.top_k = max(1, int(top_k))
        tmax = params.get("text_max_tokens_override")
        state.text_max_tokens_override = None if tmax is None else int(tmax)
        runtime_cfg = _runtime_cfg_path(state)
        return {
            "id": request_id,
            "result": {
                "ok": True,
                "state": {
                    "cfg_path": state.cfg_path,
                    "runtime_cfg_path": runtime_cfg,
                    "top_k": state.top_k,
                    "text_max_tokens_override": state.text_max_tokens_override,
                    "observer_model_path": state.observer_model_path_override,
                    "performer_model_path": state.performer_model_path_override,
                    "rewrite_llm_config_path": state.rewrite_llm_config_path,
                },
            },
        }, False

    if method == "health":
        return {
            "id": request_id,
            "result": {
                "ok": True,
                "state": {
                    "cfg_path": state.cfg_path,
                    "top_k": state.top_k,
                    "text_max_tokens_override": state.text_max_tokens_override,
                    "next_chunk_start": state.next_chunk_start,
                },
            },
        }, False

    if method == "shutdown":
        _cleanup_runtime_cfg(state)
        return {"id": request_id, "result": {"ok": True}}, True

    if state.cfg_path is None:
        return _err(request_id, "Bridge not initialized. Call initialize first.", "not_initialized"), False

    if method == "analyze_document":
        text = str(params.get("text", ""))
        input_label = str(params.get("input_label") or "<unsaved>")
        state.last_text = text
        state.last_input_label = input_label
        state.next_chunk_start = 0
        result = _analyze_slice(state, text, input_label, 0, len(text))
        state.next_chunk_start = int(result["chunk"]["analyzed_char_end"])
        result["next_chunk_start"] = state.next_chunk_start
        return {"id": request_id, "result": result}, False

    if method == "analyze_chunk":
        text = str(params.get("text", state.last_text))
        input_label = str(params.get("input_label") or state.last_input_label or "<unsaved>")
        start = int(params.get("start_char", 0))
        end = int(params.get("end_char", len(text)))
        state.last_text = text
        state.last_input_label = input_label
        result = _analyze_slice(state, text, input_label, start, end)
        state.next_chunk_start = int(result["chunk"]["analyzed_char_end"])
        result["next_chunk_start"] = state.next_chunk_start
        return {"id": request_id, "result": result}, False

    if method == "analyze_next_chunk":
        text = str(params.get("text", state.last_text))
        input_label = str(params.get("input_label") or state.last_input_label or "<unsaved>")
        if params.get("start_char") is not None:
            start = int(params.get("start_char"))
        else:
            start = int(state.next_chunk_start)
        state.last_text = text
        state.last_input_label = input_label
        result = _analyze_slice(state, text, input_label, start, len(text))
        state.next_chunk_start = int(result["chunk"]["analyzed_char_end"])
        result["next_chunk_start"] = state.next_chunk_start
        return {"id": request_id, "result": result}, False

    if method == "estimate_live_b":
        text = str(params.get("text", state.last_text))
        input_label = str(params.get("input_label") or state.last_input_label or "<unsaved>")
        start = int(params.get("start_char", 0))
        end = int(params.get("end_char", len(text)))
        base_cross = float(params.get("base_cross_logxppl", float("nan")))
        state.last_text = text
        state.last_input_label = input_label

        if observer_cache is not None:
            local = observer_cache.score_slice(state, text, start, end)
        else:
            temp_cache = ObserverWarmCache(idle_timeout_sec=5.0)
            try:
                local = temp_cache.score_slice(state, text, start, end)
            finally:
                temp_cache.close()

        observer_logppl = float(local.get("observer_logPPL", float("nan")))
        if math.isfinite(base_cross) and base_cross != 0.0 and math.isfinite(observer_logppl):
            approx_b = float(observer_logppl / base_cross)
        else:
            approx_b = float("nan")
        return {
            "id": request_id,
            "result": {
                "ok": True,
                "approx_b": approx_b,
                "observer_logPPL": observer_logppl,
                "transitions": int(local.get("transitions", 0)),
                "analyzed_char_end": int(local.get("analyzed_char_end", max(0, min(len(text), int(start))))),
                "truncated_by_limit": bool(local.get("truncated_by_limit", False)),
                "observer_cache_idle_timeout_s": float(OBSERVER_IDLE_TIMEOUT_SEC),
            },
        }, False

    if method == "rewrite_span":
        text = str(params.get("text", ""))
        span_start = int(params.get("span_start", 0))
        span_end = int(params.get("span_end", span_start))
        option_count = max(1, int(params.get("option_count", 3)))
        status_messages: List[str] = []

        def _status(msg: str) -> None:
            status_messages.append(str(msg))

        rewrites, source, fallback_reason = generate_rewrite_candidates_for_span(
            cfg_path=_runtime_cfg_path(state),
            full_text=text,
            span_start=span_start,
            span_end=span_end,
            option_count=option_count,
            status_callback=_status,
        )

        start = max(0, min(len(text), min(span_start, span_end)))
        end = max(0, min(len(text), max(span_start, span_end)))
        original_span = text[start:end]
        if rewrites:
            repaired: List[str] = []
            repaired_markers = 0
            for cand in rewrites:
                fixed, changed = _preserve_ordered_list_markers(original_span, str(cand))
                repaired.append(fixed)
                repaired_markers += max(0, int(changed))
            rewrites = repaired
            if repaired_markers > 0:
                status_messages.append(
                    f"Preserved {repaired_markers} ordered-list marker(s) from source span."
                )

        scored: List[Dict[str, Any]] = []
        base_metrics = params.get("base_metrics")
        if isinstance(base_metrics, dict):
            scored = estimate_rewrite_b_impact_options(
                cfg_path=_runtime_cfg_path(state),
                full_text=text,
                span_start=span_start,
                span_end=span_end,
                rewrites=rewrites,
                base_doc_b=float(base_metrics.get("binoculars_score", 0.0)),
                base_doc_observer_logppl=float(base_metrics.get("observer_logPPL", 0.0)),
                base_doc_cross_logxppl=float(base_metrics.get("cross_logXPPL", 0.0)),
                base_doc_transitions=int(base_metrics.get("transitions", 1)),
                text_max_tokens_override=state.text_max_tokens_override,
            )

        return {
            "id": request_id,
            "result": {
                "ok": True,
                "source": source,
                "fallback_reason": fallback_reason,
                "status_messages": status_messages,
                "rewrites": scored if scored else [{"text": t} for t in rewrites],
            },
        }, False

    return _err(request_id, f"Unknown method: {method}", "unknown_method"), False


class _ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: str, request_handler_cls: type[socketserver.BaseRequestHandler]):
        super().__init__(server_address, request_handler_cls)
        self.request_lock = threading.Lock()
        self.observer_cache = ObserverWarmCache()
        self._janitor_stop = threading.Event()
        self._janitor_thread = threading.Thread(target=self._janitor_loop, daemon=True)
        self._janitor_thread.start()

    def _janitor_loop(self) -> None:
        while not self._janitor_stop.wait(1.0):
            try:
                self.observer_cache.close_if_idle()
            except Exception:
                pass

    def server_close(self) -> None:
        self._janitor_stop.set()
        try:
            self._janitor_thread.join(timeout=1.5)
        except Exception:
            pass
        try:
            self.observer_cache.close()
        except Exception:
            pass
        super().server_close()


class _DaemonRequestHandler(socketserver.StreamRequestHandler):
    def setup(self) -> None:
        super().setup()
        self.state = BridgeState()
        self._send(
            {
                "event": "ready",
                "payload": {
                    "root_dir": ROOT_DIR,
                },
            }
        )

    def _send(self, msg: Dict[str, Any]) -> None:
        payload = (json.dumps(_json_safe(msg), ensure_ascii=True) + "\n").encode("utf-8")
        self.wfile.write(payload)
        self.wfile.flush()

    def handle(self) -> None:
        while True:
            raw = self.rfile.readline()
            if not raw:
                return
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                if not isinstance(req, dict):
                    self._send(_err(None, "Request must be a JSON object", "invalid_request"))
                    continue
            except Exception as exc:
                self._send(_err(None, f"Invalid JSON: {exc}", "invalid_json"))
                continue

            method = str(req.get("method", "")).strip()
            if method == "shutdown_daemon":
                self._send({"id": req.get("id"), "result": {"ok": True}})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return

            try:
                with self.server.request_lock:  # type: ignore[attr-defined]
                    response, should_exit = _handle_request(
                        self.state,
                        req,
                        observer_cache=getattr(self.server, "observer_cache", None),
                    )
                self._send(response)
                if should_exit:
                    return
            except Exception as exc:
                self._send(
                    _err(
                        req.get("id"),
                        f"Unhandled bridge error: {exc}",
                        "unhandled_exception",
                        {
                            "traceback": traceback.format_exc(),
                        },
                    )
                )

    def finish(self) -> None:
        try:
            _cleanup_runtime_cfg(self.state)
        except Exception:
            pass
        super().finish()


def daemon_main(socket_path: str) -> int:
    if not socket_path:
        raise ValueError("socket_path is required in daemon mode.")

    socket_path = os.path.abspath(socket_path)
    socket_dir = os.path.dirname(socket_path) or "."
    os.makedirs(socket_dir, exist_ok=True)

    if os.path.exists(socket_path):
        try:
            os.unlink(socket_path)
        except Exception as exc:
            raise RuntimeError(f"Unable to remove stale socket path: {socket_path}: {exc}") from exc

    server = _ThreadingUnixServer(socket_path, _DaemonRequestHandler)
    try:
        try:
            os.chmod(socket_path, 0o600)
        except Exception:
            pass
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        try:
            if os.path.exists(socket_path):
                os.unlink(socket_path)
        except Exception:
            pass
    return 0


def main() -> int:
    state = BridgeState()
    _send({"event": "ready", "payload": {"root_dir": ROOT_DIR}})

    try:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                if not isinstance(req, dict):
                    _send(_err(None, "Request must be a JSON object", "invalid_request"))
                    continue
            except Exception as exc:
                _send(_err(None, f"Invalid JSON: {exc}", "invalid_json"))
                continue

            try:
                response, should_exit = _handle_request(state, req)
                _send(response)
                if should_exit:
                    break
            except Exception as exc:
                _send(
                    _err(
                        req.get("id"),
                        f"Unhandled bridge error: {exc}",
                        "unhandled_exception",
                        {
                            "traceback": traceback.format_exc(),
                        },
                    )
                )
    finally:
        _cleanup_runtime_cfg(state)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--daemon", action="store_true", help="Run as shared Unix-socket daemon.")
    parser.add_argument("--socket-path", default="", help="Unix socket path for daemon mode.")
    args = parser.parse_args()
    if args.daemon:
        raise SystemExit(daemon_main(str(args.socket_path or "").strip()))
    raise SystemExit(main())
