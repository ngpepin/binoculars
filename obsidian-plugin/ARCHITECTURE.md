# Binoculars Obsidian Plugin Architecture

This document describes the current architecture of the Obsidian plugin, including UI/control surfaces, shared backend daemon usage, chunk-state model, rendering pipeline, and persistence behavior.

## 1) Design Goals

1. Keep Binoculars scoring faithful to local full-logit computation (`llama-cpp-python` via bridge backend).
2. Reuse the same daemon backend as the VS Code extension to avoid duplicate heavy processes.
3. Provide chunk-aware analysis and rewrite workflows directly inside Obsidian markdown editing.
4. Persist per-note state in hidden sidecar files for restart/reopen continuity.
5. Keep operations explicit and deterministic (user-triggered analyze/re-analyze, clear busy-state signaling).

## 2) High-Level Component Map

## 2.1 Obsidian Frontend (TypeScript)

- `obsidian-plugin/src/main.ts`
  - Plugin lifecycle (`onload`, `onunload`)
  - Command registration
  - Controls view rendering (right sidebar)
  - Ribbon icon enable/disable toggle
  - Analysis/rewrite orchestration
  - CM6 decoration + hover + gutter rendering
  - Sidecar save/load/delete orchestration
  - Status and diagnostics logging

- `obsidian-plugin/src/backendClient.ts`
  - Socket-based request/response client
  - Daemon connect/spawn/retry logic
  - Request timeouts and pending map management

- `obsidian-plugin/src/types.ts`
  - Shared request/response and state shape contracts

## 2.2 Shared Python Backend

- `vscode-extension/python/binoculars_bridge.py`
  - Serves both VS Code and Obsidian frontends
  - Daemon mode over Unix socket
  - Handles initialize/analyze/rewrite/estimate methods
  - Serializes heavy GPU scoring requests

## 2.3 Core Scoring Engine

- `binoculars.py`
  - Sequential observer/performer loading
  - Full-logit Binoculars scoring (`B`, `logPPL`, `logXPPL`)
  - Paragraph profile row generation

## 3) Backend Communication Model

## 3.1 Transport

- Newline-delimited JSON over Unix domain socket.
- Default socket path:
  - `/tmp/binoculars-vscode-<uid>.sock`
- One daemon process can serve both Obsidian and VS Code clients.

## 3.2 Session Model

- Each frontend connection initializes bridge settings for that session.
- Heavy requests are globally serialized in daemon to limit VRAM contention.

## 3.3 Startup / Shutdown

Startup path:

1. UI command needs backend (`ensureBackend`).
2. Client attempts socket connect.
3. If absent, client spawns daemon and retries connect.
4. Waits for daemon `ready` event.
5. Sends `initialize` with effective config/model overrides.

Shutdown paths:

- Plugin disable/restart requests daemon shutdown path.
- Connection close cleans pending request promises.

## 4) Plugin State Model

## 4.1 Per-Document State

`DocumentState` includes:

- analyzed `chunks[]`
- `nextChunkStart` coverage cursor
- stale/edit/rewrite flags/ranges
- prior major LOW/HIGH ranges
- forecast estimate fields for live estimate status

State key:

- vault-relative markdown path (for example `Notes/Foo.md`).

## 4.2 Runtime Caches

Plugin also maintains:

- loaded sidecar signatures (`text-hash:sidecar-hash`)
- live-estimate timers/epochs/recover-attempts
- hover and typing suppression caches
- per-doc version counters and decoration summary signatures

## 5) UI Architecture

## 5.1 Controls View (Right Sidebar)

`BinocularsControlsView` renders state-aware actions and status:

- command buttons
- status text
- analyzed chunk count
- busy-state fallback (`Refresh` only in blocked analysis scenarios)

Button visibility rules are dynamic and depend on:

- extension enabled/disabled
- has analysis
- has priors
- analysis busy state
- next-chunk availability

## 5.2 Ribbon Icon

- Owl icon is registered via custom SVG.
- Clicking toggles plugin enable/disable.
- Enabling can open/focus controls view.
- Tooltip text is dynamic (`Enable Binoculars` / `Disable Binoculars`).

## 6) Editor Integration (CodeMirror 6)

## 6.1 Extension Construction

`buildEditorExtension()` installs:

- decoration field/plugin
- forced line-number gutter (only when needed)
- contribution gutter bars
- hover tooltip provider with delayed reveal behavior

## 6.2 Decoration Model

For analyzed notes:

- major LOW/HIGH rows are colorized
- minor rows are neutral
- unscored intervals are dimmed
- prior major contributors are faint background overlays
- edited ranges have yellow background

Contribution bars are computed per line from strongest absolute row delta on that line.

## 6.3 Hover Model

Hover resolution logic:

- find best row/line match
- detect rewrite/edited overlap states
- provide LOW/HIGH diagnostics with stale/edit notes as needed
- delayed reveal + same-segment suppression reduce hover noise

## 7) Analysis And Rewrite Flow

## 7.1 Analyze

- First analyze starts at document beginning.
- Re-analyze starts from active chunk start.
- Overlapping chunk descriptors are replaced by newest analysis result.

## 7.2 Analyze Next / Analyze All

- `Analyze Next` advances from contiguous covered tail.
- `Analyze All` loops analyze-next semantics until coverage reaches end-of-text.

## 7.3 Rewrite

- Rewrite requests use selected range or active line fallback.
- Popup returns up to 3 options with approximate B impact.
- Apply marks document stale; explicit Analyze remains authoritative.

## 8) Busy-State And Cross-Note Behavior

Foreground operations increment busy counters with optional note/kind tags.

Controls view blocks analysis actions when:

- an analysis is in-flight
- and current controls note differs from busy note

Blocked status message:

- `An analysis is already in progress... please refresh or return later.`

This avoids cross-note action races while keeping behavior simple.

## 9) Persistence Model

## 9.1 Sidecar Path

Current format:

- `.<note-name>.binoculars` next to markdown note.

Fallback candidates include legacy visible `.binoculars`/`.json` variants for compatibility.

## 9.2 Load/Save Triggers

- load on active-leaf/file-open when markdown context is available
- save on vault modify for active markdown context

Hash guard:

- sidecar apply requires matching text hash to prevent stale restore.

## 9.3 Delete Coupling

On markdown note deletion:

- plugin clears in-memory state for that note
- matching sidecar files are discovered and trashed through `fileManager.trashFile`
- trash destination follows Obsidian user settings

## 10) Diagnostics And Logging

- Status text is rendered in controls panel.
- Diagnostic file:
  - `<vault>/.obsidian/plugins/binoculars-obsidian/status.log`
- Log retention:
  - capped to most recent 10,000 lines.
- Commands:
  - `Show Status Log`
  - `Copy Status Log Path`

## 11) Known Constraints

1. Obsidian plugin is desktop-only.
2. Overlay rendering depends on CM6 source-mode editor context.
3. Shared daemon means long requests from one client can delay another (serialized by design).
4. Sidecar restore is hash-guarded; changed note text requires fresh analysis.
