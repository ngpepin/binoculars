# Shared Backend Architecture (VS Code + Tk)

## Recommendation

Yes. Refactoring the Tkinter GUI to use the same persistent bridge backend is the right long-term move.

## Target Shape

- Front ends:
  - Tkinter GUI
  - VS Code extension
- Shared backend:
  - `vscode-extension/python/binoculars_bridge.py`
  - line-delimited JSON protocol
  - persistent model lifecycle (observer/performer)

## API Surface

- `initialize`
- `health`
- `analyze_document`
- `analyze_chunk`
- `analyze_next_chunk`
- `rewrite_span`
- `shutdown`

## VS Code UI Surface

- Activity Bar container: `Binoculars`
- Activity view: `Controls` (commands + live status/chunk counters)
- Editor overlays:
  - LOW/HIGH colorization
  - unscored-region dimming
  - line-level contribution gutter bars

Dark mode is the primary tuned theme. Light-mode palette fallback is present but should be tuned in a dedicated pass.

## Migration Plan for Tk

1. Add a `--bridge` runtime mode in Tk path.
2. Route Analyze and Analyze Next button handlers to bridge calls.
3. Route rewrite popup generation/scoring to bridge `rewrite_span`.
4. Keep local fallback path during rollout.
5. Remove duplicated scoring logic once parity tests are stable.

## Benefits

- One scoring/rewrite implementation across both clients.
- Lower maintenance and fewer behavior drifts.
- Easier testing via protocol-level regression fixtures.
- Faster future client implementations.
