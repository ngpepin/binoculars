export interface BridgeRequest {
  id: number;
  method: string;
  params?: Record<string, unknown>;
}

export interface BridgeError {
  code?: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface BridgeResponse<T = unknown> {
  id?: number;
  result?: T;
  error?: BridgeError;
  event?: string;
  payload?: unknown;
}

export interface ChunkMetrics {
  binoculars_score: number;
  observer_logPPL: number;
  performer_logPPL: number;
  cross_logXPPL: number;
  transitions: number;
}

export interface ParagraphRow {
  paragraph_id?: number;
  char_start: number;
  char_end: number;
  logPPL?: number;
  delta_doc_logPPL_if_removed?: number;
  excerpt?: string;
}

export interface ParagraphProfile {
  rows?: ParagraphRow[];
  analyzed_char_end?: number;
  truncated_by_limit?: boolean;
}

export interface AnalyzeResult {
  // chunk + paragraph_profile mirror bridge payload shape for analyze* methods.
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

export interface RewriteOption {
  text: string;
  approx_B?: number;
  delta_B?: number;
}

export interface RewriteResult {
  ok: boolean;
  source: string;
  fallback_reason?: string | null;
  status_messages?: string[];
  rewrites: RewriteOption[];
}

export interface ChunkState {
  // Editor/runtime-normalized chunk model used by overlay + status logic.
  charStart: number;
  charEnd: number;
  analyzedCharEnd: number;
  metrics?: ChunkMetrics;
  rows: ParagraphRow[];
}
