# Binoculars Obsidian Plugin

This plugin adds a full Obsidian modality for Binoculars with parity-focused behavior versus the VS Code extension:

- Analyze Chunk / Analyze Next Chunk / Analyze All
- Rewrite Selection workflow with ranked options and approximate B impact
- LOW/HIGH contributor colorization and minor-contributor neutral rendering
- Per-line contribution gutter bars
- Delayed whole-line hover diagnostics with stale/rewrite/manual-edit messaging
- Runtime `Toggle Colorization`
- Prior contributor overlays (`Clear Priors`)
- Stale-state live estimate (`B Est.`) using observer-only projection
- Sidecar state persistence to `.<note>.binoculars`
- Persistent Python bridge backend (`binoculars_bridge.py`) with restart command
- Diagnostic status log with `Show Status Log` / `Copy Status Log Path` commands

Detailed docs:

- `../USERGUIDE-OBS.md`
- `./ARCHITECTURE.md`

## Build

```bash
cd obsidian-plugin
npm install
npm run build
```

## Install (manual)

1. Create (or use) the folder:
   `.obsidian/plugins/binoculars-obsidian/`
2. Copy these files into that folder:
   - `main.js`
   - `manifest.json`
   - `styles.css`
3. Enable **Binoculars** in Obsidian Settings -> Community Plugins.

## Configure

Use Obsidian Settings -> Binoculars to set:

- Python path
- Bridge script path (`vscode-extension/python/binoculars_bridge.py`)
- Binoculars config path
- Optional GGUF overrides
- Optional external rewrite LLM overrides

## Diagnostics

- Command: `Binoculars: Show Status Log`
- Command: `Binoculars: Copy Status Log Path`
- File location: `.obsidian/plugins/binoculars-obsidian/status.log`
