# Binoculars GUI User Guide

This guide explains how to use the GUI for iterative editing and rescoring, with emphasis on the intended workflow:
- fast rewrite iterations using approximate impact feedback,
- occasional full `Analyze` runs for exact metrics,
- periodic `Clear Priors` to keep the view readable.

## 1) Start the GUI

From repo root:

```bash
./binoculars.sh --config fast --gui path/to/doc.md
```

or:

```bash
venv/bin/python binoculars.py --config fast --gui path/to/doc.md
```

## 2) UI Overview

Main areas:
- Left pane: editable source text.
- Left gutter: line numbers and per-line contribution bars.
- Right pane: markdown preview with score-based backgrounds.
- Toolbar: `Analyze`, `Save`, `Clear Priors`, `Quit`.
- Status bar: current score/metrics and workflow state.

Color conventions (after Analyze):
- Red foreground lines/segments: lower-perplexity (`LOW`) contributors.
- Green foreground lines/segments: higher-perplexity (`HIGH`) contributors.
- Yellow foreground: your edits since last Analyze.
- Faint background colors: prior analyses/edits preserved for comparison.

## 3) Core Workflow Concept

The GUI is intentionally split into two phases:

1. Approximate iteration phase:
- edit text,
- request rewrites,
- review approximate B impact in rewrite popup,
- apply options quickly.

2. Exact scoring phase:
- run `Analyze` periodically,
- refresh exact `B`, exact observer/cross metrics, and heatmap labels.

Important:
- Rewrite popup impact values are approximations.
- Exact B always requires full `Analyze`.
- After edits/rewrites, status bar marks B as stale until Analyze.

## 4) Recommended Iterative Loop

Use this loop for practical throughput:

1. Press `Analyze` on the current document.
2. Inspect red and green contributors.
3. Make direct edits or request rewrites.
4. Apply one or more rewrite options (no auto-analyze).
5. Continue editing while watching stale indicator.
6. Re-run `Analyze` when you need exact score confirmation.
7. Press `Clear Priors` when overlays become visually noisy.
8. Repeat.

Suggested cadence:
- Run `Analyze` every few applied rewrites (for example every 3-8), or after a major section change.
- Run `Clear Priors` whenever faint backgrounds start obscuring current hot spots.

## 5) Requesting Rewrites

### A) Rewrite a single scored red segment

1. Ensure at least one successful `Analyze` has been run.
2. Right-click directly on a red (`LOW`) segment.
3. Wait for popup generation/scoring.
4. Review options `[1]`, `[2]`, `[3]` with approximate B impact.
5. Select an option by button or keyboard `1`, `2`, `3`.
6. Option text replaces that segment in left pane.

### B) Rewrite a highlighted block (multi-line)

1. Highlight any block in left pane (can include mixed LOW/HIGH/neutral lines).
2. Right-click while selection is active.
3. GUI rounds selection to full lines for rewrite generation.
4. If selection extends into unscored text, only scored overlap is rewritten.
5. Popup shows 3 options and approximate B deltas.
6. Select option by button or `1`/`2`/`3`.

Selection rewrite behavior:
- Unchanged source lines are preserved even if model output accidentally omits/collapses them.
- Trailing blank lines from selected source are preserved in replacement.

## 6) Rewrite Popup Behavior

Popup details:
- Scrollable option body for long rewrites.
- Status/progress lines while generating.
- Source label indicates rewrite backend:
  - `external LLM`,
  - `internal fallback`, or
  - `internal model`.
- `Quit` or `Esc` cancels with no change.

Option ordering:
- Sorted by expected increase in B (more human-like first).

## 7) Exact vs Approximate Metrics

Approximate values in popup:
- Designed for ranking options quickly.
- Do not recompute full document cross term.

Exact values in status bar:
- Updated only by `Analyze`.
- Include exact `B`, `Observer logPPL`, `Performer logPPL`, `Cross logXPPL`, and scored `Last Line`.

## 8) Prior Overlays and `Clear Priors`

What priors are:
- Faint background traces of previous analyses and edits.
- Useful for “before vs after” visual context.

When to clear:
- After several iteration cycles when old context is distracting.
- Before final review pass if you want only the latest heatmap visible.

What `Clear Priors` does:
- Removes faint prior backgrounds.
- Does not alter current text.
- Does not run Analyze.

## 9) Preview Pane Selection Mirroring

When you highlight a block on the left:
- Right preview mirrors the same selected block.
- Within selected block:
  - LOW lines use red-tinted background,
  - HIGH lines use green-tinted background,
  - neutral lines use neutral highlight style.

This makes it easier to inspect local context before requesting rewrites.

## 10) Saving and Output Files

`Save` writes timestamped files next to the source:

- `<stem>_edited_YYYYMMDDHHMM.md`
- If collision occurs, numeric suffix is appended.

Tip:
- Save snapshots at meaningful milestones (for example after each Analyze-confirmed improvement).

## 11) Optional External Rewrite Backend (`config.binoculars.llm.json`)

If configured and reachable, rewrite generation can use any OpenAI-compatible endpoint.
If config is absent/disabled/unreachable, GUI automatically falls back to internal performer model.

Typical fields:
- `llm.enabled`
- `llm.endpoint_url`
- `llm.request_path`
- `llm.model`
- `llm.api_key` or `llm.api_key_env`
- `llm.timeout_s`, `llm.max_tokens`, `llm.temperature`, `llm.top_p`
- context controls (`context_chars_each_side`, `context_paragraphs_each_side`, `context_window_max_chars`)
- `llm.extra_headers`, `llm.extra_body`

## 12) Troubleshooting

Rewrite menu does not open:
- Run `Analyze` at least once first.
- Ensure no analysis/rewrite operation is already in progress.

Rewrite generation seems slow:
- First run may include model warmup.
- External endpoint latency can dominate.
- Reduce context fields in `config.binoculars.llm.json` if needed.

B score seems inconsistent with popup deltas:
- Popup deltas are approximate.
- Run full `Analyze` for authoritative values.

Visual clutter is high:
- Use `Clear Priors`, then continue iteration.

---

Practical summary:
- Use rewrites + edits for fast local iteration,
- trust popup numbers for relative ranking,
- use `Analyze` as the exact checkpoint,
- use `Clear Priors` as periodic visual reset.
