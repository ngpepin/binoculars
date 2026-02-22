#!/usr/bin/env python3
"""
Simple GUI test harness for the Binoculars local HTTP API.

This utility is intentionally lightweight so users can quickly:
1) Verify API health.
2) Submit text segments for scoring.
3) Inspect both a compact metric summary and full raw JSON response.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
    from tkinter.scrolledtext import ScrolledText
except Exception as exc:  # pragma: no cover
    raise SystemExit("Tkinter is required to run this demo harness.") from exc


API_PORT = 8765
DEFAULT_BASE_URL = os.environ.get("BINOCULARS_API_BASE_URL", f"http://127.0.0.1:{API_PORT}")
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_INPUT_LABEL = "demo-snippet"
DEFAULT_SAMPLE_TEXT = (
    "Paste or type text here, then press 'Score Segment'.\n\n"
    "This GUI sends POST /score to the Binoculars local API and displays the returned metrics."
)


@dataclass
class ScoreRequestOptions:
    """Options used for /score requests."""

    input_label: str
    diagnose_paragraphs: bool
    diagnose_top_k: int
    need_paragraph_profile: bool


def _http_get_json(url: str, timeout_s: float) -> Dict[str, Any]:
    """Issue a GET request and parse JSON response."""

    req = urllib.request.Request(url=url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=max(0.25, float(timeout_s))) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Connection error: {exc.reason}") from exc
    try:
        parsed = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception as exc:
        raise RuntimeError("Response is not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Expected JSON object response.")
    return parsed


def _http_post_json(url: str, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    """Issue a POST request with JSON body and parse JSON response."""

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=max(0.25, float(timeout_s))) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body_text}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Connection error: {exc.reason}") from exc
    try:
        parsed = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception as exc:
        raise RuntimeError("Response is not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Expected JSON object response.")
    return parsed


def _nested_get(obj: Dict[str, Any], path: list[str], default: Any = None) -> Any:
    """Safely fetch nested dictionary values."""

    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _format_float(value: Any, decimals: int = 6) -> str:
    """Format numeric display values for summary view."""

    try:
        return f"{float(value):.{int(decimals)}f}"
    except Exception:
        return "n/a"


class ApiDemoHarness:
    """Tkinter app for interactive API testing."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Binoculars API Demo Harness")
        self.root.geometry("1050x760")
        self.root.minsize(900, 650)

        self.base_url_var = tk.StringVar(value=DEFAULT_BASE_URL)
        self.timeout_var = tk.StringVar(value=str(DEFAULT_TIMEOUT_S))
        self.input_label_var = tk.StringVar(value=DEFAULT_INPUT_LABEL)
        self.diagnose_var = tk.BooleanVar(value=False)
        self.top_k_var = tk.StringVar(value="10")
        self.profile_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready.")

        self.health_button: ttk.Button
        self.score_button: ttk.Button
        self.clear_button: ttk.Button
        self.summary_text: ScrolledText
        self.raw_text: ScrolledText
        self.input_text: ScrolledText
        self.busy = False

        self._build_ui()

    def _build_ui(self) -> None:
        """Create all widgets and layout."""

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        connection = ttk.LabelFrame(self.root, text="Connection")
        connection.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        connection.columnconfigure(1, weight=1)

        ttk.Label(connection, text="Base URL").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(connection, textvariable=self.base_url_var).grid(row=0, column=1, sticky="ew", padx=8, pady=6)

        ttk.Label(connection, text="Timeout (s)").grid(row=0, column=2, sticky="w", padx=(12, 8), pady=6)
        ttk.Entry(connection, textvariable=self.timeout_var, width=10).grid(row=0, column=3, sticky="w", padx=(0, 8), pady=6)

        self.health_button = ttk.Button(connection, text="Health Check", command=self.on_health_check)
        self.health_button.grid(row=0, column=4, sticky="e", padx=(10, 8), pady=6)

        self.score_button = ttk.Button(connection, text="Score Segment", command=self.on_score_segment)
        self.score_button.grid(row=0, column=5, sticky="e", padx=(0, 8), pady=6)

        self.clear_button = ttk.Button(connection, text="Clear Response", command=self.on_clear_response)
        self.clear_button.grid(row=0, column=6, sticky="e", padx=(0, 8), pady=6)

        options = ttk.LabelFrame(self.root, text="Request Options")
        options.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))
        options.columnconfigure(7, weight=1)

        ttk.Label(options, text="input_label").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(options, textvariable=self.input_label_var, width=20).grid(row=0, column=1, sticky="w", padx=(0, 12), pady=6)

        ttk.Checkbutton(options, text="diagnose_paragraphs", variable=self.diagnose_var).grid(
            row=0, column=2, sticky="w", padx=(0, 10), pady=6
        )

        ttk.Label(options, text="diagnose_top_k").grid(row=0, column=3, sticky="w", padx=(0, 8), pady=6)
        ttk.Spinbox(options, from_=1, to=500, textvariable=self.top_k_var, width=8).grid(
            row=0, column=4, sticky="w", padx=(0, 12), pady=6
        )

        ttk.Checkbutton(options, text="need_paragraph_profile", variable=self.profile_var).grid(
            row=0, column=5, sticky="w", padx=(0, 8), pady=6
        )

        body = ttk.Panedwindow(self.root, orient=tk.VERTICAL)
        body.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))

        input_frame = ttk.LabelFrame(body, text="Text Segment")
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(0, weight=1)
        self.input_text = ScrolledText(input_frame, wrap="word", height=10)
        self.input_text.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.input_text.insert("1.0", DEFAULT_SAMPLE_TEXT)
        body.add(input_frame, weight=3)

        output_frame = ttk.LabelFrame(body, text="Response")
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(2, weight=1)
        output_frame.rowconfigure(4, weight=2)

        ttk.Label(output_frame, text="Summary").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 2))
        self.summary_text = ScrolledText(output_frame, wrap="word", height=8, state="disabled")
        self.summary_text.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        ttk.Label(output_frame, text="Raw JSON").grid(row=2, column=0, sticky="w", padx=8, pady=(0, 2))
        self.raw_text = ScrolledText(output_frame, wrap="none", height=14, state="disabled")
        self.raw_text.grid(row=3, column=0, sticky="nsew", padx=8, pady=(0, 8))

        status = ttk.Label(output_frame, textvariable=self.status_var, anchor="w")
        status.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 8))

        body.add(output_frame, weight=5)

    def _set_busy(self, busy: bool) -> None:
        """Enable/disable action buttons while a request is running."""

        self.busy = bool(busy)
        state = "disabled" if self.busy else "normal"
        self.health_button.configure(state=state)
        self.score_button.configure(state=state)

    def _set_text_widget(self, widget: ScrolledText, value: str) -> None:
        """Replace text content in a read-only ScrolledText widget."""

        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def _normalized_base_url(self) -> str:
        """Normalize and validate base URL input."""

        raw = str(self.base_url_var.get() or "").strip().rstrip("/")
        if not raw:
            raise ValueError("Base URL is required.")
        parsed = urllib.parse.urlparse(raw)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Base URL must include scheme and host, for example http://127.0.0.1:8765.")
        return raw

    def _timeout_seconds(self) -> float:
        """Parse timeout from UI."""

        try:
            value = float(self.timeout_var.get())
        except Exception as exc:
            raise ValueError("Timeout must be numeric.") from exc
        if value <= 0:
            raise ValueError("Timeout must be greater than 0.")
        return value

    def _collect_score_options(self) -> ScoreRequestOptions:
        """Collect /score option flags from UI controls."""

        input_label = str(self.input_label_var.get() or "").strip() or DEFAULT_INPUT_LABEL
        try:
            top_k = int(str(self.top_k_var.get() or "10").strip())
        except Exception as exc:
            raise ValueError("diagnose_top_k must be an integer.") from exc
        if top_k < 1:
            raise ValueError("diagnose_top_k must be >= 1.")
        return ScoreRequestOptions(
            input_label=input_label,
            diagnose_paragraphs=bool(self.diagnose_var.get()),
            diagnose_top_k=top_k,
            need_paragraph_profile=bool(self.profile_var.get()),
        )

    def _run_async(
        self,
        work: Callable[[], Dict[str, Any]],
        on_success: Callable[[Dict[str, Any]], None],
        action_label: str,
    ) -> None:
        """Run request work on a background thread and marshal results to UI thread."""

        if self.busy:
            return
        self._set_busy(True)
        self.status_var.set(f"{action_label}...")

        def worker() -> None:
            try:
                result = work()
                err: Optional[Exception] = None
            except Exception as exc:
                result = {}
                err = exc
            self.root.after(0, lambda: self._finish_async(result, err, on_success))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_async(
        self,
        result: Dict[str, Any],
        err: Optional[Exception],
        on_success: Callable[[Dict[str, Any]], None],
    ) -> None:
        """Finalize asynchronous request completion on UI thread."""

        self._set_busy(False)
        if err is not None:
            self.status_var.set(f"Request failed: {err}")
            messagebox.showerror("Request Failed", str(err))
            return
        on_success(result)

    def on_health_check(self) -> None:
        """Handle health check button."""

        try:
            base = self._normalized_base_url()
            timeout_s = self._timeout_seconds()
        except Exception as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return
        url = f"{base}/health"
        self._run_async(lambda: _http_get_json(url, timeout_s), self._handle_health_response, "Checking health")

    def on_score_segment(self) -> None:
        """Handle score button."""

        try:
            base = self._normalized_base_url()
            timeout_s = self._timeout_seconds()
            options = self._collect_score_options()
        except Exception as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return

        segment = self.input_text.get("1.0", "end-1c")
        if not segment.strip():
            messagebox.showwarning("Empty Text", "Enter a text segment before scoring.")
            return

        payload: Dict[str, Any] = {
            "text": segment,
            "input_label": options.input_label,
            "diagnose_paragraphs": options.diagnose_paragraphs,
            "diagnose_top_k": options.diagnose_top_k,
            "need_paragraph_profile": options.need_paragraph_profile,
        }
        url = f"{base}/score"
        self._run_async(lambda: _http_post_json(url, payload, timeout_s), self._handle_score_response, "Scoring segment")

    def _handle_health_response(self, data: Dict[str, Any]) -> None:
        """Render health response in summary/raw panes."""

        summary_lines = [
            "Health check completed.",
            "",
            f"ok: {data.get('ok', False)}",
            f"service: {data.get('service', 'n/a')}",
            f"route: {data.get('route', 'n/a')}",
            f"ts_utc: {data.get('ts_utc', 'n/a')}",
        ]
        self._set_text_widget(self.summary_text, "\n".join(summary_lines))
        self._set_text_widget(self.raw_text, json.dumps(data, indent=2, ensure_ascii=False))
        self.status_var.set("Health check succeeded.")

    def _handle_score_response(self, data: Dict[str, Any]) -> None:
        """Render scoring response in summary/raw panes."""

        result_obj = data.get("result", {})
        if not isinstance(result_obj, dict):
            result_obj = {}

        paragraph_profile = data.get("paragraph_profile")
        row_count = 0
        if isinstance(paragraph_profile, dict):
            rows = paragraph_profile.get("rows", [])
            if isinstance(rows, list):
                row_count = len(rows)

        summary_lines = [
            "Score request completed.",
            "",
            f"ok: {data.get('ok', False)}",
            f"tokens: {_nested_get(result_obj, ['input', 'tokens'], 'n/a')}",
            f"transitions: {_nested_get(result_obj, ['input', 'transitions'], 'n/a')}",
            f"observer.logPPL: {_format_float(_nested_get(result_obj, ['observer', 'logPPL']))}",
            f"performer.logPPL: {_format_float(_nested_get(result_obj, ['performer', 'logPPL']))}",
            f"cross.logXPPL: {_format_float(_nested_get(result_obj, ['cross', 'logXPPL']))}",
            f"binoculars.score: {_format_float(_nested_get(result_obj, ['binoculars', 'score']))}",
            f"paragraph_profile_rows: {row_count}",
        ]
        self._set_text_widget(self.summary_text, "\n".join(summary_lines))
        self._set_text_widget(self.raw_text, json.dumps(data, indent=2, ensure_ascii=False))
        self.status_var.set("Score request succeeded.")

    def on_clear_response(self) -> None:
        """Clear response panes and status line."""

        self._set_text_widget(self.summary_text, "")
        self._set_text_widget(self.raw_text, "")
        self.status_var.set("Ready.")


def main() -> int:
    """Program entry point."""

    root = tk.Tk()
    ApiDemoHarness(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
