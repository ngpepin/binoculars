#!/usr/bin/env python3
"""Persistent JSON bridge for Binoculars analysis/rewrite workflows.

Protocol: line-delimited JSON request/response over stdin/stdout.
Each request must include: {"id": ..., "method": "...", "params": {...}}
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is importable when this script runs from subdirectory.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from binoculars import (  # type: ignore
    analyze_text_document,
    estimate_rewrite_b_impact_options,
    generate_rewrite_candidates_for_span,
)


@dataclass
class BridgeState:
    cfg_path: Optional[str] = None
    top_k: int = 5
    text_max_tokens_override: Optional[int] = None
    last_text: str = ""
    last_input_label: str = "<unsaved>"
    next_chunk_start: int = 0


def _send(msg: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(msg, ensure_ascii=True) + "\n")
    sys.stdout.flush()


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
        cfg_path=state.cfg_path or "",
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


def _handle_request(state: BridgeState, request: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    request_id = request.get("id")
    method = str(request.get("method", "")).strip()
    params = request.get("params", {})
    if not isinstance(params, dict):
        return _err(request_id, "params must be a JSON object", "invalid_params"), False

    if method == "initialize":
        cfg_path = params.get("cfg_path")
        if cfg_path:
            state.cfg_path = str(cfg_path)
        top_k = params.get("top_k")
        if top_k is not None:
            state.top_k = max(1, int(top_k))
        tmax = params.get("text_max_tokens_override")
        state.text_max_tokens_override = None if tmax is None else int(tmax)
        return {
            "id": request_id,
            "result": {
                "ok": True,
                "state": {
                    "cfg_path": state.cfg_path,
                    "top_k": state.top_k,
                    "text_max_tokens_override": state.text_max_tokens_override,
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

    if method == "rewrite_span":
        text = str(params.get("text", ""))
        span_start = int(params.get("span_start", 0))
        span_end = int(params.get("span_end", span_start))
        option_count = max(1, int(params.get("option_count", 3)))
        status_messages: List[str] = []

        def _status(msg: str) -> None:
            status_messages.append(str(msg))

        rewrites, source, fallback_reason = generate_rewrite_candidates_for_span(
            cfg_path=state.cfg_path,
            full_text=text,
            span_start=span_start,
            span_end=span_end,
            option_count=option_count,
            status_callback=_status,
        )

        scored: List[Dict[str, Any]] = []
        base_metrics = params.get("base_metrics")
        if isinstance(base_metrics, dict):
            scored = estimate_rewrite_b_impact_options(
                cfg_path=state.cfg_path,
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


def main() -> int:
    state = BridgeState()
    _send({"event": "ready", "payload": {"root_dir": ROOT_DIR}})

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
