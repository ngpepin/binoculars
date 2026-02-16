/** JSON-RPC style request sent to the Python bridge. */
export interface BridgeRequest {
  id: number;
  method: string;
  params?: Record<string, unknown>;
}

/** Error payload returned by the Python bridge. */
export interface BridgeError {
  code?: string;
  message: string;
  details?: Record<string, unknown>;
}

/** Generic response envelope used by bridge requests and events. */
export interface BridgeResponse<T = unknown> {
  id?: number;
  result?: T;
  error?: BridgeError;
  event?: string;
  payload?: unknown;
}

/** Exact chunk-level Binoculars metrics from an Analyze run. */
export interface ChunkMetrics {
  binoculars_score: number;
  observer_logPPL: number;
  performer_logPPL: number;
  cross_logXPPL: number;
  transitions: number;
}

/** One scored paragraph/span entry returned by paragraph profiling. */
export interface ParagraphRow {
  paragraph_id?: number;
  char_start: number;
  char_end: number;
  logPPL?: number;
  delta_doc_logPPL_if_removed?: number;
  excerpt?: string;
}

/** Paragraph profile payload attached to chunk analysis results. */
export interface ParagraphProfile {
  rows?: ParagraphRow[];
  analyzed_char_end?: number;
  truncated_by_limit?: boolean;
}

/** Analyze result contract returned by the bridge. */
export interface AnalyzeResult {
  ok: boolean;
  analysis?: Record<string, unknown>;
  paragraph_profile?: ParagraphProfile | null;
  chunk: {
    char_start: number;
    char_end: number;
    analyzed_char_end: number;
    metrics?: ChunkMetrics;
  };
  next_chunk_start?: number;
}

/** One rewrite option (with optional approximate impact metrics). */
export interface RewriteOption {
  text: string;
  approx_B?: number;
  delta_B?: number;
}

/** Rewrite RPC result payload. */
export interface RewriteResult {
  ok: boolean;
  source: string;
  fallback_reason?: string | null;
  status_messages?: string[];
  rewrites: RewriteOption[];
}

/** In-memory analyzed chunk state cached by the extension. */
export interface ChunkState {
  charStart: number;
  charEnd: number;
  analyzedCharEnd: number;
  metrics?: ChunkMetrics;
  rows: ParagraphRow[];
}
