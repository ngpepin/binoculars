# Binoculars VS Code UML (PlantUML)

This file provides architecture diagrams for the VS Code extension and shared daemon backend.

## Component Diagram

```plantuml
@startuml
title Binoculars VS Code - Component Diagram

actor User
node "VS Code Window A" as VSA {
  component "Extension Host\n(extension.ts)" as ExtA
  component "Backend Client\n(backendClient.ts)" as ClientA
}
node "VS Code Window B" as VSB {
  component "Extension Host\n(extension.ts)" as ExtB
  component "Backend Client\n(backendClient.ts)" as ClientB
}

database "Markdown Sidecar\n<doc>.json" as Sidecar
queue "Unix Socket\n/tmp/binoculars-vscode-<uid>.sock" as Sock

node "Shared Python Daemon\n(binoculars_bridge.py --daemon)" as Daemon {
  component "Daemon Server\n(ThreadingUnixServer)" as Server
  component "Per-Connection\nBridgeState" as SessionState
  component "Request Handler\n(_handle_request)" as Handler
}

component "Scoring Core\n(binoculars.py)" as Core
database "GGUF Models\nObserver + Performer" as Models

User --> ExtA : Analyze/Rewrite commands
User --> ExtB : Analyze/Rewrite commands
ExtA --> ClientA
ExtB --> ClientB
ClientA --> Sock
ClientB --> Sock
Sock --> Server
Server --> SessionState
Server --> Handler
Handler --> Core
Core --> Models
ExtA --> Sidecar : save/load state
ExtB --> Sidecar : save/load state

note right of Server
Global request lock serializes
heavy scoring requests across
all VS Code windows.
end note

@enduml
```

## Class Diagram

```plantuml
@startuml
title Binoculars VS Code - Class Diagram

class BackendClient {
  - socket: net.Socket
  - pending: Map<int, PendingRequest>
  - requestSeq: int
  + start(pythonPath, bridgeScriptPath)
  + initialize(...)
  + analyzeDocument(text, inputLabel)
  + analyzeChunk(text, inputLabel, startChar, endChar)
  + analyzeNextChunk(text, inputLabel, startChar?)
  + rewriteSpan(text, spanStart, spanEnd, baseMetrics, optionCount)
  + shutdown()
  + shutdownDaemon()
}

class DocumentState {
  + chunks: ChunkState[]
  + nextChunkStart: number
  + stale: boolean
  + editedRanges: EditedRange[]
  + rewriteRanges: EditedRange[]
  + priorLowRanges: EditedRange[]
  + priorHighRanges: EditedRange[]
  + priorChunkB: number?
}

class ChunkState {
  + charStart: number
  + charEnd: number
  + analyzedCharEnd: number
  + metrics: ChunkMetrics?
  + rows: ParagraphRow[]
}

class BinocularsControlsProvider {
  + getChildren()
  + refresh()
}

class BridgeState {
  + cfg_path: str?
  + top_k: int
  + text_max_tokens_override: int?
  + next_chunk_start: int
}

class DaemonServer {
  + request_lock: Lock
}

class DaemonRequestHandler {
  + state: BridgeState
  + handle()
}

class BinocularsCore {
  + analyze_text_document(...)
  + generate_rewrite_candidates_for_span(...)
  + estimate_rewrite_b_impact_options(...)
}

DocumentState "1" *-- "many" ChunkState
BackendClient ..> BridgeState : initialize/session state
BinocularsControlsProvider ..> DocumentState
DaemonServer o-- DaemonRequestHandler
DaemonRequestHandler *-- BridgeState
DaemonRequestHandler ..> BinocularsCore

@enduml
```

## Sequence Diagram: Analyze Chunk

```plantuml
@startuml
title Analyze Chunk Flow (with shared daemon)

actor User
participant "Extension Host" as Ext
participant "BackendClient" as Client
participant "Daemon\n(binoculars_bridge.py)" as Daemon
participant "Binoculars Core\n(binoculars.py)" as Core
database "Models\nObserver/Performer" as Models

User -> Ext : Run "Binoculars: Analyze Chunk"
Ext -> Client : ensureBackend()
Client -> Daemon : connect(socket)
alt daemon not running
  Client -> Client : spawn daemon (--daemon --socket-path)
  Client -> Daemon : reconnect(socket)
end
Daemon --> Client : event ready
Ext -> Client : initialize(cfg, topK, overrides)
Client -> Daemon : request initialize
Daemon --> Client : result ok

Ext -> Client : analyze_document(text, inputLabel)
Client -> Daemon : request analyze_document
Daemon -> Core : analyze_text_document(...)
Core -> Models : observer eval + performer eval
Models --> Core : logits/perplexity terms
Core --> Daemon : chunk/profile/metrics
Daemon --> Client : result analyze_document
Client --> Ext : AnalyzeResult
Ext -> Ext : merge chunk state + apply decorations + update status

@enduml
```

## Sequence Diagram: Disable / Re-enable

```plantuml
@startuml
title Enable/Disable Flow (global setting)

actor User
participant "Extension Host" as Ext
participant "BackendClient" as Client
participant "Daemon" as Daemon
database "VS Code User Settings" as Settings
database "Markdown Sidecar\n<doc>.json" as Sidecar

User -> Ext : Run "Binoculars: Disable"
Ext -> Settings : set binoculars.enabled = false (global)
Settings --> Ext : config changed event
Ext -> Client : shutdown_daemon()
Client -> Daemon : request shutdown_daemon
Daemon --> Client : ok + terminate
Ext -> Ext : clear decorations, hide status bar
Ext -> Ext : show only Enable command

User -> Ext : Run "Binoculars: Enable"
Ext -> Settings : set binoculars.enabled = true (global)
Settings --> Ext : config changed event
Ext -> Sidecar : load/restore document state
Ext -> Ext : show status + reapply overlays
note over Ext
Backend remains lazy-started until
next analysis/rewrite command.
end note

@enduml
```

