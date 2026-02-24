# Binoculars Obsidian User Guide

This guide covers day-to-day use of the Binoculars Obsidian plugin (`obsidian-plugin/`).

## 1) What The Plugin Does

Inside Obsidian, the plugin lets you:

- Analyze markdown notes in chunks (`Analyze Chunk`, `Analyze Next Chunk`, `Analyze All`).
- View major LOW/HIGH contributors in text colorization and contribution gutter bars.
- Open rewrite options for the current selection or active line.
- Keep prior major contributor context with faint red/green backgrounds after re-analysis.
- Toggle text colorization on/off at runtime.
- Persist and restore analysis state with hidden sidecars (`.<note>.binoculars`).
- Share the same Binoculars daemon used by the VS Code extension.

## 2) Install / Deploy

Manual deploy target:

- `<vault>/.obsidian/plugins/binoculars-obsidian/`

Files required in that folder:

- `main.js`
- `manifest.json`
- `styles.css`

Development helper from repo root:

```bash
./deploy-obsidian-plugin.sh
```

If already built and only copying is needed:

```bash
./deploy-obsidian-plugin.sh --no-build
```

Then in Obsidian:

1. Open `Settings -> Community plugins`.
2. Enable `Binoculars`.

## 3) First-Time Configuration

Open `Settings -> Binoculars` and verify:

- Python path
- Bridge script path (`vscode-extension/python/binoculars_bridge.py`)
- Binoculars config path
- Optional model overrides (observer/performer GGUF)
- Optional external rewrite LLM overrides
- Editor integration toggle (CM6 overlays)

Notes:

- Overlays (colorization/bars/hover) require editor integration and Source mode rendering.
- Backend operations use the shared daemon socket by default.

## 4) Commands

Main commands:

- `Binoculars: Open Controls`
- `Binoculars: Enable`
- `Binoculars: Disable`
- `Binoculars: Analyze Chunk`
- `Binoculars: Analyze Next Chunk`
- `Binoculars: Analyze All`
- `Binoculars: Rewrite Selection`
- `Binoculars: Clear Priors`
- `Binoculars: Toggle Colorization`
- `Binoculars: Restart Backend`
- `Binoculars: Show Status Log`
- `Binoculars: Copy Status Log Path`

You can run these from the command palette or from the Binoculars controls view in the right sidebar.

## 5) Recommended Workflow

1. Open a markdown note in Source mode.
2. Run `Analyze Chunk`.
3. Review major LOW/HIGH text and gutter bars.
4. Use `Rewrite Selection` on selected text or an active line.
5. Re-run `Analyze Chunk` for exact updated metrics.
6. Use `Analyze Next Chunk` or `Analyze All` for long notes.
7. Use `Clear Priors` when faint prior overlays become noisy.

Important:

- Rewrite options show approximate impact; exact metrics require Analyze.
- Status marks stale state until analysis is run again.

## 6) Colorization / Bars / Priors

Current rendering model:

- Major LOW/HIGH contributors: colorized.
- Minor contributors: neutral (not major red/green).
- Prior major contributors: faint red/green background overlays.
- Gutter bars: available independently of text colorization toggle.

`Toggle Colorization` affects text overlays only and does not remove analysis state.

## 7) Sidecar State Files

For markdown notes, state is saved next to the note as:

- `.<note-name>.binoculars`

Legacy candidates may still load if present.

Sidecar stores:

- chunk descriptors and metrics
- edited/rewrite ranges
- prior overlay ranges
- coverage cursor metadata

Delete behavior:

- Deleting a markdown note in Obsidian also trashes matching sidecar files.
- Trash destination follows Obsidian settings (system trash vs local `.trash`).

## 8) Busy-State Behavior Across Notes

When an analysis is in progress for a different note and you switch notes:

- Controls show status:
  - `An analysis is already in progress... please refresh or return later.`
- Analysis/re-analysis buttons are hidden.
- A `Refresh` button is shown.

This is intentional to keep multi-note analysis flow deterministic without push notifications.

## 9) Status And Diagnostics

Status text is rendered in the controls panel (notices are not required for normal progress state).

Status log tools:

- `Binoculars: Show Status Log`
- `Binoculars: Copy Status Log Path`

Log location:

- `<vault>/.obsidian/plugins/binoculars-obsidian/status.log`

Log retention:

- capped to the most recent 10,000 lines.

## 10) Troubleshooting

No overlays appear:

- Ensure Editor integration is enabled in plugin settings.
- Open the note in Source mode.
- Confirm note has analyzed chunks.

Analyze appears blocked in another note:

- Use `Refresh` in controls view.
- Return to the note that started analysis.
- Restart backend if needed.

Rewrite unavailable:

- Run `Analyze Chunk` first.
- Ensure a markdown editor context is active.

Backend timeout:

- Shared daemon may be busy with a long analysis (Obsidian or VS Code).
- Wait and retry, or run `Restart Backend`.

Stale sidecar behavior:

- Sidecar restore requires matching text hash.
- If note content changed externally, run a fresh analyze pass.

## 11) Quick Reference

- Open controls: `Binoculars: Open Controls`
- Analyze current chunk: `Binoculars: Analyze Chunk`
- Continue chunking: `Binoculars: Analyze Next Chunk`
- Analyze remaining text: `Binoculars: Analyze All`
- Rewrite: `Binoculars: Rewrite Selection`
- Clear prior backgrounds: `Binoculars: Clear Priors`
- Toggle overlays: `Binoculars: Toggle Colorization`
- Restart daemon backend: `Binoculars: Restart Backend`
