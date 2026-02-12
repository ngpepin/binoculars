import * as path from 'node:path';
import * as vscode from 'vscode';
import { BackendClient } from './backendClient';
import { AnalyzeResult, ChunkState, ParagraphRow, RewriteOption } from './types';

interface DocumentState {
  chunks: ChunkState[];
  nextChunkStart: number;
  stale: boolean;
}

const docStates = new Map<string, DocumentState>();
let backend: BackendClient | undefined;
let backendStarted = false;
let statusBar: vscode.StatusBarItem;
let output: vscode.OutputChannel;

const lowDecoration = vscode.window.createTextEditorDecorationType({
  color: '#ff6b6b',
});

const highDecoration = vscode.window.createTextEditorDecorationType({
  color: '#3fd28a',
});

const unscoredDecoration = vscode.window.createTextEditorDecorationType({
  color: '#a8a8a8',
  opacity: '0.75',
});

class GutterPalette implements vscode.Disposable {
  private readonly redBars: vscode.TextEditorDecorationType[] = [];
  private readonly greenBars: vscode.TextEditorDecorationType[] = [];

  constructor(private readonly levels = 21) {
    for (let i = 0; i < levels; i += 1) {
      this.redBars.push(this.makeBarType('#ff6b6b', i));
      this.greenBars.push(this.makeBarType('#3fd28a', i));
    }
  }

  public bucketType(sign: 'red' | 'green', level: number): vscode.TextEditorDecorationType {
    const idx = Math.max(0, Math.min(this.levels - 1, level));
    return sign === 'red' ? this.redBars[idx] : this.greenBars[idx];
  }

  public dispose(): void {
    for (const d of this.redBars) {
      d.dispose();
    }
    for (const d of this.greenBars) {
      d.dispose();
    }
  }

  private makeBarType(color: string, level: number): vscode.TextEditorDecorationType {
    const width = Math.max(2, Math.round(((level + 1) / this.levels) * 12));
    const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='14' height='18'><rect x='0' y='1' width='${width}' height='16' rx='1' fill='${color}'/></svg>`;
    const gutterIconPath = vscode.Uri.parse(`data:image/svg+xml;utf8,${encodeURIComponent(svg)}`);
    return vscode.window.createTextEditorDecorationType({
      gutterIconPath,
      gutterIconSize: 'contain',
      overviewRulerColor: color,
      overviewRulerLane: vscode.OverviewRulerLane.Left,
    });
  }
}

let gutterPalette: GutterPalette;

export function activate(context: vscode.ExtensionContext): void {
  output = vscode.window.createOutputChannel('Binoculars');
  statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  statusBar.name = 'Binoculars';
  statusBar.show();
  gutterPalette = new GutterPalette();

  context.subscriptions.push(
    output,
    statusBar,
    gutterPalette,
    lowDecoration,
    highDecoration,
    unscoredDecoration,
    vscode.commands.registerCommand('binoculars.analyze', () => void runAnalyze()),
    vscode.commands.registerCommand('binoculars.analyzeNext', () => void runAnalyzeNext()),
    vscode.commands.registerCommand('binoculars.rewriteSelection', () => void runRewriteSelection()),
    vscode.commands.registerCommand('binoculars.clearPriors', () => clearPriors()),
    vscode.commands.registerCommand('binoculars.restartBackend', () => void restartBackend()),
    vscode.window.onDidChangeTextEditorSelection((evt) => updateStatusForEditor(evt.textEditor)),
    vscode.workspace.onDidChangeTextDocument((evt) => {
      const key = evt.document.uri.toString();
      const state = docStates.get(key);
      if (state && evt.contentChanges.length > 0) {
        state.stale = true;
      }
    }),
  );

  updateStatus('Ready. Run Binoculars: Analyze Active Document.');
}

export async function deactivate(): Promise<void> {
  await backend?.shutdown();
  backend = undefined;
  backendStarted = false;
}

async function runAnalyze(): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    return;
  }
  const client = await ensureBackend();
  const text = editor.document.getText();
  updateStatus('Analyzing document...');

  try {
    const result = await client.analyzeDocument(text, inputLabel(editor.document));
    const state: DocumentState = {
      chunks: [toChunkState(result)],
      nextChunkStart: result.nextChunkStart ?? result.chunk.analyzed_char_end,
      stale: false,
    };
    docStates.set(editor.document.uri.toString(), state);
    applyDecorations(editor, state);
    updateStatusForEditor(editor);
  } catch (err) {
    showError(`Analyze failed: ${(err as Error).message}`);
  }
}

async function runAnalyzeNext(): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    return;
  }

  const key = editor.document.uri.toString();
  const existing = docStates.get(key);
  if (!existing) {
    await runAnalyze();
    return;
  }

  const start = Math.max(0, existing.nextChunkStart);
  const fullText = editor.document.getText();
  if (start >= fullText.length) {
    updateStatus('All text already covered by analyzed chunks.');
    return;
  }

  const client = await ensureBackend();
  updateStatus(`Analyzing next chunk from char ${start}...`);

  try {
    const result = await client.analyzeNextChunk(fullText, inputLabel(editor.document), start);
    mergeChunk(existing, toChunkState(result));
    existing.nextChunkStart = result.nextChunkStart ?? result.chunk.analyzed_char_end;
    existing.stale = false;
    applyDecorations(editor, existing);
    updateStatusForEditor(editor);
  } catch (err) {
    showError(`Analyze Next failed: ${(err as Error).message}`);
  }
}

async function runRewriteSelection(): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    return;
  }

  const key = editor.document.uri.toString();
  const state = docStates.get(key);
  if (!state || state.chunks.length === 0) {
    showError('Run Analyze first before requesting rewrite suggestions.');
    return;
  }

  const text = editor.document.getText();
  const sel = editor.selection;
  const span = resolveRewriteSpan(editor, state);
  if (!span) {
    showError('No scored span available for rewrite at cursor/selection.');
    return;
  }

  const activeChunk = getActiveChunk(editor, text, state);
  const baseMetrics = activeChunk?.metrics;
  const client = await ensureBackend();
  updateStatus('Generating rewrite options...');

  try {
    const result = await client.rewriteSpan(
      text,
      span.start,
      span.end,
      baseMetrics
        ? {
            binoculars_score: baseMetrics.binoculars_score,
            observer_logPPL: baseMetrics.observer_logPPL,
            cross_logXPPL: baseMetrics.cross_logXPPL,
            transitions: baseMetrics.transitions,
          }
        : undefined,
      3,
    );

    const picks = (result.rewrites || []).slice(0, 3).map((rw: RewriteOption, idx: number) => ({
      label: `[${idx + 1}] ${formatApprox(rw.approx_B, rw.delta_B)}`,
      description: rw.text.slice(0, 180).replace(/\s+/g, ' '),
      detail: rw.text,
      option: rw,
    }));

    if (picks.length === 0) {
      showError('No rewrite options returned.');
      return;
    }

    const selected = await vscode.window.showQuickPick(picks, {
      title: `Rewrite options (${result.source})`,
      placeHolder: 'Select rewrite option to apply',
      canPickMany: false,
      ignoreFocusOut: true,
    });

    if (!selected) {
      updateStatus('Rewrite selection canceled.');
      return;
    }

    const range = new vscode.Range(positionAt(text, span.start), positionAt(text, span.end));
    await editor.edit((eb) => eb.replace(range, selected.option.text));

    const next = docStates.get(key);
    if (next) {
      next.stale = true;
    }
    updateStatus('Rewrite applied. Re-run Analyze for exact B score.');

    // Place caret after inserted text for iterative edits.
    const newDoc = editor.document.getText();
    const newPos = positionAt(newDoc, span.start + selected.option.text.length);
    editor.selection = new vscode.Selection(newPos, newPos);
  } catch (err) {
    showError(`Rewrite failed: ${(err as Error).message}`);
  }
}

function clearPriors(): void {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    return;
  }
  const key = editor.document.uri.toString();
  docStates.delete(key);
  clearAllDecorations(editor);
  updateStatus('Cleared prior highlights.');
}

async function restartBackend(): Promise<void> {
  try {
    await backend?.shutdown();
  } catch {
    // ignore
  }
  backend = undefined;
  backendStarted = false;
  await ensureBackend();
  updateStatus('Binoculars backend restarted.');
}

async function ensureBackend(): Promise<BackendClient> {
  if (!backend) {
    backend = new BackendClient(output);
  }

  const cfg = vscode.workspace.getConfiguration('binoculars');
  const pythonPath = cfg.get<string>('backend.pythonPath', 'python');
  const bridgeScriptPath = resolvePathSetting(cfg.get<string>('backend.bridgeScriptPath', ''));
  const configPath = resolvePathSetting(cfg.get<string>('configPath', ''));
  const topK = cfg.get<number>('topK', 5);
  const textMaxTokensOverride = cfg.get<number | null>('textMaxTokensOverride', null);

  if (!(backend as BackendClient)) {
    throw new Error('Backend client unavailable.');
  }

  if (!backendStarted) {
    await backend.start(pythonPath, bridgeScriptPath);
    backendStarted = true;
  }

  await backend.initialize(configPath, topK, textMaxTokensOverride);
  return backend;
}

function resolvePathSetting(raw: string): string {
  const ws = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  if (!raw) {
    return '';
  }
  if (raw.includes('${workspaceFolder}') && ws) {
    return raw.split('${workspaceFolder}').join(ws);
  }
  return path.isAbsolute(raw) ? raw : ws ? path.join(ws, raw) : raw;
}

function toChunkState(result: AnalyzeResult): ChunkState {
  return {
    charStart: result.chunk.char_start,
    charEnd: result.chunk.char_end,
    analyzedCharEnd: result.chunk.analyzed_char_end,
    metrics: result.chunk.metrics,
    rows: result.paragraph_profile?.rows ?? [],
  };
}

function mergeChunk(state: DocumentState, incoming: ChunkState): void {
  const kept = state.chunks.filter((c) => c.analyzedCharEnd <= incoming.charStart || c.charStart >= incoming.analyzedCharEnd);
  kept.push(incoming);
  kept.sort((a, b) => a.charStart - b.charStart);
  state.chunks = kept;
}

function applyDecorations(editor: vscode.TextEditor, state: DocumentState): void {
  clearAllDecorations(editor);

  const cfg = vscode.workspace.getConfiguration('binoculars');
  const enableColorize = cfg.get<boolean>('render.colorizeText', true);
  const enableBars = cfg.get<boolean>('render.contributionBars', true);

  const docText = editor.document.getText();
  const lowRanges: vscode.Range[] = [];
  const highRanges: vscode.Range[] = [];

  const lineContribution = new Map<number, { sign: 'red' | 'green'; mag: number }>();

  for (const chunk of state.chunks) {
    for (const row of chunk.rows) {
      const start = Math.max(0, Math.min(docText.length, row.char_start));
      const end = Math.max(start, Math.min(docText.length, row.char_end));
      if (end <= start) {
        continue;
      }
      const range = new vscode.Range(positionAt(docText, start), positionAt(docText, end));
      const delta = Number(row.delta_doc_logPPL_if_removed ?? 0.0);
      if (delta >= 0) {
        lowRanges.push(range);
      } else {
        highRanges.push(range);
      }

      const startLine = range.start.line;
      const endLine = range.end.line;
      const mag = Math.abs(delta);
      const sign: 'red' | 'green' = delta >= 0 ? 'red' : 'green';
      for (let line = startLine; line <= endLine; line += 1) {
        const prev = lineContribution.get(line);
        if (!prev || mag > prev.mag) {
          lineContribution.set(line, { sign, mag });
        }
      }
    }
  }

  if (enableColorize) {
    editor.setDecorations(lowDecoration, lowRanges);
    editor.setDecorations(highDecoration, highRanges);
  }

  const scoredIntervals = mergeIntervals(
    state.chunks
      .map((c) => ({ start: c.charStart, end: c.analyzedCharEnd }))
      .filter((x) => x.end > x.start),
  );
  const unscored = invertIntervals(scoredIntervals, docText.length);
  const unscoredRanges = unscored.map((iv) => new vscode.Range(positionAt(docText, iv.start), positionAt(docText, iv.end)));
  editor.setDecorations(unscoredDecoration, unscoredRanges);

  if (enableBars) {
    const mags = [...lineContribution.values()].map((x) => x.mag);
    const maxMag = mags.length > 0 ? Math.max(...mags) : 0;
    const buckets = 21;
    const grouped = new Map<vscode.TextEditorDecorationType, vscode.DecorationOptions[]>();
    for (const [line, info] of lineContribution) {
      const level = maxMag > 0 ? Math.min(buckets - 1, Math.round((info.mag / maxMag) * (buckets - 1))) : 0;
      const decType = gutterPalette.bucketType(info.sign, level);
      const lineRange = editor.document.lineAt(line).range;
      const arr = grouped.get(decType) ?? [];
      arr.push({ range: lineRange });
      grouped.set(decType, arr);
    }
    for (const [decType, opts] of grouped) {
      editor.setDecorations(decType, opts);
    }
  }
}

function clearAllDecorations(editor: vscode.TextEditor): void {
  editor.setDecorations(lowDecoration, []);
  editor.setDecorations(highDecoration, []);
  editor.setDecorations(unscoredDecoration, []);
  for (let i = 0; i < 21; i += 1) {
    editor.setDecorations(gutterPalette.bucketType('red', i), []);
    editor.setDecorations(gutterPalette.bucketType('green', i), []);
  }
}

function getActiveChunk(editor: vscode.TextEditor, text: string, state: DocumentState): ChunkState | undefined {
  if (state.chunks.length === 0) {
    return undefined;
  }

  const offset = offsetAt(text, editor.selection.active);
  let containing = state.chunks.find((c) => offset >= c.charStart && offset < c.analyzedCharEnd);
  if (containing) {
    return containing;
  }

  containing = state.chunks.find((c) => offset >= c.charStart && offset <= c.charEnd);
  if (containing) {
    return containing;
  }

  const sorted = [...state.chunks].sort((a, b) => Math.abs(a.charStart - offset) - Math.abs(b.charStart - offset));
  return sorted[0];
}

function resolveRewriteSpan(editor: vscode.TextEditor, state: DocumentState): { start: number; end: number } | undefined {
  const docText = editor.document.getText();
  const sel = editor.selection;
  if (!sel.isEmpty) {
    const start = offsetAt(docText, sel.start);
    const end = offsetAt(docText, sel.end);
    return { start: Math.min(start, end), end: Math.max(start, end) };
  }

  const cursor = offsetAt(docText, sel.active);
  const active = getActiveChunk(editor, docText, state);
  if (!active) {
    return undefined;
  }

  const row = active.rows
    .filter((r) => Number(r.delta_doc_logPPL_if_removed ?? 0) >= 0)
    .find((r) => cursor >= r.char_start && cursor <= r.char_end);
  if (!row) {
    return undefined;
  }
  return { start: row.char_start, end: row.char_end };
}

function updateStatusForEditor(editor: vscode.TextEditor): void {
  const key = editor.document.uri.toString();
  const state = docStates.get(key);
  if (!state || state.chunks.length === 0) {
    updateStatus('Ready. Run Binoculars: Analyze Active Document.');
    return;
  }
  const text = editor.document.getText();
  const chunk = getActiveChunk(editor, text, state);
  if (!chunk || !chunk.metrics) {
    updateStatus('Binoculars analyzed chunk available.');
    return;
  }
  const staleSuffix = state.stale ? ' | stale (run Analyze)' : '';
  const covered = Math.max(0, state.nextChunkStart);
  const hasMore = covered < text.length;
  const moreSuffix = hasMore ? ` | Analyze Next available (${covered}/${text.length})` : '';
  updateStatus(
    `B ${chunk.metrics.binoculars_score.toFixed(6)} | observer ${chunk.metrics.observer_logPPL.toFixed(6)} | cross ${chunk.metrics.cross_logXPPL.toFixed(6)}${staleSuffix}${moreSuffix}`,
  );
}

function updateStatus(message: string): void {
  statusBar.text = `$(telescope) Binoculars: ${message}`;
}

function showError(message: string): void {
  void vscode.window.showErrorMessage(message);
  updateStatus(message);
}

function inputLabel(doc: vscode.TextDocument): string {
  return doc.uri.scheme === 'file' ? doc.uri.fsPath : '<unsaved>';
}

function formatApprox(approxB?: number, deltaB?: number): string {
  if (typeof approxB !== 'number' || Number.isNaN(approxB)) {
    return 'rewrite option';
  }
  const d = typeof deltaB === 'number' && Number.isFinite(deltaB) ? ` (${deltaB >= 0 ? '+' : ''}${deltaB.toFixed(6)})` : '';
  return `approx B ${approxB.toFixed(6)}${d}`;
}

function positionAt(text: string, offset: number): vscode.Position {
  const bounded = Math.max(0, Math.min(text.length, offset));
  const lines = text.slice(0, bounded).split('\n');
  const line = Math.max(0, lines.length - 1);
  const ch = lines[lines.length - 1]?.length ?? 0;
  return new vscode.Position(line, ch);
}

function offsetAt(text: string, pos: vscode.Position): number {
  const lines = text.split('\n');
  let offset = 0;
  const line = Math.max(0, Math.min(lines.length - 1, pos.line));
  for (let i = 0; i < line; i += 1) {
    offset += lines[i].length + 1;
  }
  return Math.max(0, Math.min(text.length, offset + pos.character));
}

function mergeIntervals(intervals: Array<{ start: number; end: number }>): Array<{ start: number; end: number }> {
  if (intervals.length === 0) {
    return [];
  }
  const sorted = [...intervals].sort((a, b) => a.start - b.start);
  const merged: Array<{ start: number; end: number }> = [sorted[0]];
  for (let i = 1; i < sorted.length; i += 1) {
    const prev = merged[merged.length - 1];
    const cur = sorted[i];
    if (cur.start <= prev.end) {
      prev.end = Math.max(prev.end, cur.end);
    } else {
      merged.push({ start: cur.start, end: cur.end });
    }
  }
  return merged;
}

function invertIntervals(intervals: Array<{ start: number; end: number }>, totalLen: number): Array<{ start: number; end: number }> {
  const out: Array<{ start: number; end: number }> = [];
  let cursor = 0;
  for (const iv of intervals) {
    if (iv.start > cursor) {
      out.push({ start: cursor, end: iv.start });
    }
    cursor = Math.max(cursor, iv.end);
  }
  if (cursor < totalLen) {
    out.push({ start: cursor, end: totalLen });
  }
  return out.filter((x) => x.end > x.start);
}
