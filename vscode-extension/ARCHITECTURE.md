# Binoculars VS Code Extension Architecture

This document describes the current architecture of the VS Code extension, including the shared daemonized backend, chunk-aware analysis model, rendering pipeline, and persistence model.

## 1) Design Goals

1. Keep Binoculars scoring faithful to local full-logit computation (`llama-cpp-python`), not API approximations.
2. Support interactive editing loops in VS Code with chunk-aware analysis and rewrite workflows.
3. Share one backend daemon across VS Code windows to avoid duplicate heavyweight backend processes.
4. Preserve analysis state across saves/reopens via hidden sidecar `.binoculars` files.
5. Keep UI responsive with clear status, controlled overlays, and explicit user-triggered analysis.

## 2) High-Level Component Map

## 2.1 Extension Host (TypeScript)

- `src/extension.ts`
  - Command registration (`Analyze Chunk`, `Analyze Next Chunk`, `Rewrite Selection`, etc.)
  - Status bar messaging
  - Decoration and hover rendering
  - Document state tracking and sidecar persistence
  - Enable/disable mode handling

- `src/backendClient.ts`
  - JSON-RPC-like request/response client over Unix socket
  - Auto-connect to existing daemon
  - Auto-spawn daemon if not running
  - Request timeout/pending resolution

- `src/types.ts`
  - Shared request/response and analysis/rewrite typing

## 2.2 Python Backend

- `python/binoculars_bridge.py`
  - Core request handler logic for:
    - initialize
    - analyze_document
    - analyze_chunk
    - analyze_next_chunk
    - rewrite_span
    - shutdown
  - Daemon mode (`--daemon --socket-path ...`) for shared multi-window service
  - Per-client bridge state (one `BridgeState` per socket connection)
  - Global request serialization lock to prevent overlapping heavy GPU work

## 2.3 Core Scoring Engine

- Repository-level `binoculars.py`
  - Loads observer/performer sequentially
  - Computes Binoculars metrics (`B`, `logPPL`, `logXPPL`)
  - Produces paragraph profile rows and chunk truncation endpoints

## 3) Backend Communication Model

## 3.1 Transport

- Transport is newline-delimited JSON over a Unix domain socket.
- Default socket path: `/tmp/binoculars-vscode-<uid>.sock`.
- One daemon process serves multiple extension windows.

## 3.2 Session and State

- Each socket connection has an independent `BridgeState` in Python.
- This means each VS Code window/client keeps its own:
  - config path and model overrides
  - top-k setting
  - chunk cursor (`next_chunk_start`)
  - rewrite backend config path
- Requests from all clients are globally serialized by a daemon lock, preventing concurrent heavy scoring from exhausting VRAM.

## 3.3 Startup Sequence

1. Extension command needs backend -> `ensureBackend()`.
2. `BackendClient.start(...)` tries to connect to existing socket.
3. If not found, it spawns daemon:
   - `python binoculars_bridge.py --daemon --socket-path <path>`
4. Client connects, waits for `{"event":"ready", ...}`.
5. Client sends `initialize` before analysis/rewrite calls.

## 3.4 Shutdown Sequence

- `shutdown`: closes that client session state/connection.
- `shutdown_daemon`: stops the shared daemon process for all windows.
- Extension disable and restart use daemon shutdown so GPU/RAM allocations are released globally.

## 4) Extension Enable/Disable Architecture

## 4.1 Global Toggle

- Setting: `binoculars.enabled` (global user setting).
- Commands:
  - `Binoculars: Enable`
  - `Binoculars: Disable`

Because it is a global setting, all VS Code windows observe the same enabled/disabled state.

## 4.2 Disabled Behavior

When disabled:
- backend daemon is stopped
- decorations are cleared
- status bar is hidden (quiet)
- hover provider returns no Binoculars hovers
- analysis/rewrite commands are gated
- menus show only `Enable`

## 4.3 Re-enable Behavior

When re-enabled:
- sidecar/recent state restoration runs like startup
- decorations are reapplied for open documents
- status bar resumes normal operation
- backend is lazily started on first analysis/rewrite request

## 5) Analysis State Model in Extension

Per document (`DocumentState`):
- `chunks[]`: analyzed chunk descriptors
  - `charStart`
  - `charEnd`
  - `analyzedCharEnd`
  - chunk metrics
  - paragraph rows
- `nextChunkStart`: contiguous tail cursor for `Analyze Next Chunk`
- `stale`: true when text changed since analysis
- `editedRanges`: manual/rewrite dirty ranges
- `rewriteRanges`: rewritten spans
- `priorLowRanges` / `priorHighRanges`: faint background priors from previous major contributors
- `priorChunkB`: prior chunk score for status comparison

## 5.1 Chunk Semantics

- First analyze starts at document start.
- Analyze Next starts from contiguous covered tail.
- Re-analyze of active chunk starts at chunk start boundary.
- Chunk boundaries are mutable after edits (token density shifts endpoints).
- Old overlapping chunk descriptors are replaced by newest descriptors.

## 6) Rendering Architecture

## 6.1 Decorations

- Major LOW rows: red text
- Major HIGH rows: green text
- Minor rows: neutral light gray (dark theme) / black (light theme)
- Unscored regions: dimmed overlay
- Prior major contributors: faint red/green background
- Edited/rewrite ranges: faint yellow background
- Gutter bars: per-line contribution magnitude/sign

## 6.2 Hover Diagnostics

- Each scored row has hover diagnostics.
- Contributor and stale/edit hovers use a shared delayed reveal gate to reduce visual noise (currently ~1.15s).
- Same-segment suppression prevents immediate hover re-pop in the same segment for a short cooldown window.
- Rewritten rows show direct rewrite note.
- Manually edited rows show stale-statistics note.

## 6.3 Colorization Toggle

- `Toggle Colorization` hides/shows text overlays at runtime.
- Re-enabling restores in-memory overlays, including prior and edited backgrounds.

## 7) Status Bar Architecture

Status uses active editor + cursor + chunk coverage:

- No analysis:
  - `Binoculars Ready. Select Analyze Chunk to begin.`
- Cursor in analyzed chunk:
  - `B`, `Prior B` (if present), `Obs`, `Cross`, stale flag, analyze-next hint
  - Multi-chunk docs prefix with `Binoculars (chunk N): ...`
- Cursor in unanalyzed region:
  - Reports where unanalyzed region starts by line number

When extension is disabled, status bar is hidden.

## 8) Persistence and Restore

## 8.1 Sidecar Contract

For markdown docs, state is persisted to:
- `.<document_basename>.binoculars` in same directory

Includes:
- chunk descriptors and metrics
- edited/rewrite ranges
- prior range overlays
- chunk cursor/state metadata

## 8.2 Load/Restore Triggers

- activation/open/active-editor events
- recent-closed in-memory candidate map to survive editor churn
- save events auto-write sidecar

## 9) Menu and Command Surface

Editor context and view title menus are controlled by `binoculars.enabled`:

- Enabled:
  - Analyze Chunk
  - Analyze Next Chunk
  - Rewrite Selection
  - Clear Priors
  - Toggle Colorization
  - Disable
- Disabled:
  - Enable only

Command palette also conditionally hides/shows commands by enablement.

## 10) Concurrency and Resource Guarantees

1. One daemon process per user/session socket path.
2. Multiple windows connect to same daemon.
3. Requests are serialized in daemon to avoid concurrent VRAM spikes.
4. Disable/restart can stop daemon globally to free resources.

## 11) Known Constraints

1. Shared daemon is Unix-socket based (Linux/macOS-oriented pathing).
2. If multiple windows issue heavy work, requests queue (serialized by design).
3. Sidecar restore depends on matching text hash and compatible persisted shape.

## 12) Practical Trace: Typical User Flow

1. User runs `Analyze Chunk`.
2. Extension ensures daemon connection and initializes backend session.
3. Backend returns chunk metrics + paragraph profile rows.
4. Extension updates document state and overlays.
5. User edits or rewrites text:
   - stale flag set
   - edited/rewrite decorations applied
6. User runs `Analyze Next Chunk` or re-analyzes active chunk.
7. Extension merges/replaces overlapping chunk descriptors and updates status.
8. Sidecar saved on document save for restart recovery.
