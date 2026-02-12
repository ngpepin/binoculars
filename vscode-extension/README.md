# Binoculars VS Code Extension (Scaffold)

This directory contains a first-pass scaffold for a native VS Code extension front end that reuses Binoculars scoring/rewrite logic through a persistent Python bridge.

## Current Scope

Included in this scaffold:
- Persistent backend bridge process (`python/binoculars_bridge.py`)
- Extension commands + default hotkeys
- Settings contract for GGUF paths/config and external LLM details
- Analysis rendering in editor:
  - LOW/HIGH text colorization
  - Unscored-region rendering
  - Per-line contribution gutter bars
- Chunk workflows:
  - Analyze full document
  - Analyze next chunk
- Rewrite workflow:
  - Rewrite selection or active red contributor under cursor
  - QuickPick chooser with approximate B impact

Intentionally out of scope here:
- Markdown preview pane integration
- Synonym functionality (already covered by existing VS Code ecosystem)

## Directory Layout

- `package.json`: extension manifest (commands, keybindings, settings)
- `src/extension.ts`: VS Code UI logic + decorations + command handlers
- `src/backendClient.ts`: persistent bridge client (JSON over stdio)
- `python/binoculars_bridge.py`: backend API adaptor over core `binoculars.py`
- `schemas/bridge.protocol.schema.json`: protocol schema for request/response envelopes

## Commands / Hotkeys

Default bindings (users can remap via Keyboard Shortcuts):
- `Binoculars: Analyze Active Document` -> `Ctrl+Alt+B`
- `Binoculars: Analyze Next Chunk` -> `Ctrl+Alt+N`
- `Binoculars: Rewrite Selection` -> `Ctrl+Alt+R`
- `Binoculars: Clear Priors` -> `Ctrl+Alt+C`

## Settings

Key settings contributed by this extension:
- `binoculars.backend.pythonPath`
- `binoculars.backend.bridgeScriptPath`
- `binoculars.configPath`
- `binoculars.textMaxTokensOverride`
- `binoculars.topK`
- `binoculars.models.observerGgufPath`
- `binoculars.models.performerGgufPath`
- `binoculars.externalLlm.*`
- `binoculars.render.*`

Note: model overrides are declared now for forward compatibility. Current bridge behavior still relies on the supplied scoring config file.

## Build / Run (from this folder)

```bash
npm install
npm run compile
```

Then run the extension in VS Code Extension Development Host (F5) from this folder.

## Shared-Backend Direction (Tk + VS Code)

Yes, it makes architectural sense to refactor the Tk front end to consume the same persistent bridge API.

Why:
- Single source of truth for scoring/rewrite/chunk semantics.
- Faster feature parity between GUI clients.
- Lower regression risk (one API contract, one test harness).
- Easier future front ends (web/TUI) without duplicating model lifecycle code.

Recommended migration path:
1. Keep current Tk code functional.
2. Route Tk Analyze/AnalyzeNext/Rewrite calls through bridge methods behind a feature flag.
3. Move chunk-state logic and rewrite approximation baseline logic into bridge responses.
4. Retire duplicated Tk-only scoring paths after parity tests pass.

