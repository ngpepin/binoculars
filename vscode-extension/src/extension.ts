import * as crypto from 'node:crypto';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as vscode from 'vscode';
import { BackendClient } from './backendClient';
import { AnalyzeResult, ChunkMetrics, ChunkState, ParagraphRow, RewriteOption } from './types';

interface EditedRange {
  start: number;
  end: number;
}

interface DocumentState {
  chunks: ChunkState[];
  nextChunkStart: number;
  stale: boolean;
  editedRanges: EditedRange[];
}

interface RenderPalette {
  lowColor: string;
  highColor: string;
  unscoredColor: string;
  unscoredOpacity: string;
}

interface PersistedSidecarPayload {
  binoculars_gui_state: boolean;
  version: number;
  saved_at: string;
  document_path: string;
  document_basename: string;
  text_sha256: string;
  state: Record<string, unknown>;
}

interface RecentClosedStateCandidate {
  savedAtMs: number;
  state: DocumentState;
}

const DISPLAY_DECIMALS = 5;

const docStates = new Map<string, DocumentState>();
const loadedSidecarSignatures = new Map<string, string>();
const recentClosedStatesByTextHash = new Map<string, RecentClosedStateCandidate>();
const RECENT_CLOSED_STATE_TTL_MS = 120_000;
let backend: BackendClient | undefined;
let backendStarted = false;
let statusBar: vscode.StatusBarItem;
let output: vscode.OutputChannel;
let lastStatusMessage = 'Ready. Run Binoculars: Analyze Chunk.';
let controlsProvider: BinocularsControlsProvider | undefined;
let renderPalette: RenderPalette;
let extensionCtx: vscode.ExtensionContext;

function resolveRenderPalette(): RenderPalette {
  // Dark-first palette. Light-mode palette remains intentionally conservative
  // until dedicated light-theme tuning is implemented.
  if (vscode.window.activeColorTheme.kind === vscode.ColorThemeKind.Light) {
    return {
      lowColor: '#b23636',
      highColor: '#1f8f57',
      unscoredColor: '#7d8792',
      unscoredOpacity: '0.72',
    };
  }
  return {
    lowColor: '#ff6b6b',
    highColor: '#3fd28a',
    unscoredColor: '#a8a8a8',
    unscoredOpacity: '0.75',
  };
}

const lowDecoration = vscode.window.createTextEditorDecorationType({
  color: resolveRenderPalette().lowColor,
});

const highDecoration = vscode.window.createTextEditorDecorationType({
  color: resolveRenderPalette().highColor,
});

const unscoredDecoration = vscode.window.createTextEditorDecorationType({
  color: resolveRenderPalette().unscoredColor,
  opacity: resolveRenderPalette().unscoredOpacity,
});

const editedDecoration = vscode.window.createTextEditorDecorationType({
  color: '#ffd54f',
});

class BinocularsControlItem extends vscode.TreeItem {
  constructor(
    label: string,
    opts?: {
      description?: string;
      tooltip?: string;
      commandId?: string;
      icon?: vscode.ThemeIcon;
    },
  ) {
    super(label, vscode.TreeItemCollapsibleState.None);
    this.description = opts?.description;
    this.tooltip = opts?.tooltip;
    this.command = opts?.commandId
      ? {
          command: opts.commandId,
          title: label,
        }
      : undefined;
    this.iconPath = opts?.icon ?? new vscode.ThemeIcon('circle-outline');
    this.contextValue = opts?.commandId ? 'binocularsCommand' : 'binocularsInfo';
  }
}

class BinocularsControlsProvider implements vscode.TreeDataProvider<BinocularsControlItem> {
  private readonly emitter = new vscode.EventEmitter<BinocularsControlItem | undefined>();
  readonly onDidChangeTreeData = this.emitter.event;

  refresh(): void {
    this.emitter.fire(undefined);
  }

  getTreeItem(element: BinocularsControlItem): vscode.TreeItem {
    return element;
  }

  getChildren(_element?: BinocularsControlItem): Thenable<BinocularsControlItem[]> {
    const editor = vscode.window.activeTextEditor;
    const key = editor?.document.uri.toString() ?? '';
    const state = key ? docStates.get(key) : undefined;
    const chunkCount = state?.chunks.length ?? 0;
    const nextAvailable = (() => {
      if (!editor || !state) {
        return false;
      }
      const covered = Math.max(0, state.nextChunkStart);
      return covered < editor.document.getText().length;
    })();

    const items: BinocularsControlItem[] = [
      new BinocularsControlItem('Analyze Chunk', {
        commandId: 'binoculars.analyze',
        icon: new vscode.ThemeIcon('run'),
        description: 'Ctrl+Alt+B',
      }),
      new BinocularsControlItem('Analyze Next Chunk', {
        commandId: 'binoculars.analyzeNext',
        icon: new vscode.ThemeIcon('debug-step-over'),
        description: nextAvailable ? 'available' : 'n/a',
      }),
      new BinocularsControlItem('Rewrite Selection', {
        commandId: 'binoculars.rewriteSelectionOrLine',
        icon: new vscode.ThemeIcon('sparkle'),
        description: 'Ctrl+Alt+R',
      }),
      new BinocularsControlItem('Clear Priors', {
        commandId: 'binoculars.clearPriors',
        icon: new vscode.ThemeIcon('clear-all'),
        description: 'Ctrl+Alt+C',
      }),
      new BinocularsControlItem('Restart Backend', {
        commandId: 'binoculars.restartBackend',
        icon: new vscode.ThemeIcon('debug-restart'),
      }),
      new BinocularsControlItem('Status', {
        description: lastStatusMessage,
        tooltip: lastStatusMessage,
        icon: new vscode.ThemeIcon('info'),
      }),
      new BinocularsControlItem('Analyzed Chunks', {
        description: `${chunkCount}`,
        icon: new vscode.ThemeIcon('list-unordered'),
      }),
    ];
    return Promise.resolve(items);
  }
}

class GutterPalette implements vscode.Disposable {
  private readonly redBars: vscode.TextEditorDecorationType[] = [];
  private readonly greenBars: vscode.TextEditorDecorationType[] = [];

  constructor(
    private readonly redColor: string,
    private readonly greenColor: string,
    private readonly levels = 21,
  ) {
    for (let i = 0; i < levels; i += 1) {
      this.redBars.push(this.makeBarType(this.redColor, i));
      this.greenBars.push(this.makeBarType(this.greenColor, i));
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
    const width = Math.max(2, Math.round(((level + 1) / this.levels) * 18));
    const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='22' height='18'><rect x='0' y='1' width='${width}' height='16' rx='1' fill='${color}'/></svg>`;
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
  extensionCtx = context;
  output = vscode.window.createOutputChannel('Binoculars');
  statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  statusBar.name = 'Binoculars';
  statusBar.show();
  renderPalette = resolveRenderPalette();
  gutterPalette = new GutterPalette(renderPalette.lowColor, renderPalette.highColor);
  controlsProvider = new BinocularsControlsProvider();

  context.subscriptions.push(
    output,
    statusBar,
    gutterPalette,
    lowDecoration,
    highDecoration,
    unscoredDecoration,
    editedDecoration,
    vscode.commands.registerCommand('binoculars.analyze', () => void runAnalyze()),
    vscode.commands.registerCommand('binoculars.analyzeNext', () => void runAnalyzeNext()),
    vscode.commands.registerCommand('binoculars.rewriteSelection', () => void runRewriteSelection()),
    vscode.commands.registerCommand('binoculars.rewriteSelectionOrLine', () => void runRewriteSelectionOrLine()),
    vscode.commands.registerCommand('binoculars.recommendRewrite', () => void runRewriteSelectionOrLine()),
    vscode.commands.registerCommand('binoculars.clearPriors', () => clearPriors()),
    vscode.commands.registerCommand('binoculars.restartBackend', () => void restartBackend()),
    vscode.window.registerTreeDataProvider('binoculars.controlsView', controlsProvider),
    vscode.languages.registerHoverProvider([{ language: 'markdown' }, { language: 'plaintext' }], {
      provideHover(document, position) {
        return hoverForPosition(document, position);
      },
    }),
    vscode.window.onDidChangeTextEditorSelection((evt) => updateStatusForEditor(evt.textEditor)),
    vscode.window.onDidChangeActiveTextEditor((editor) => {
      if (editor) {
        maybeLoadStateSidecar(editor.document, 'activate');
        maybeRestoreStateFromRecentClosed(editor.document, 'activate');
        const state = docStates.get(editor.document.uri.toString());
        if (state) {
          applyDecorations(editor, state);
        } else {
          clearAllDecorations(editor);
        }
        updateStatusForEditor(editor);
      } else {
        updateStatus('Ready. Run Binoculars: Analyze Chunk.');
      }
    }),
    vscode.workspace.onDidOpenTextDocument((doc) => {
      maybeLoadStateSidecar(doc, 'open');
      maybeRestoreStateFromRecentClosed(doc, 'open');
    }),
    vscode.workspace.onDidCloseTextDocument((doc) => {
      const key = doc.uri.toString();
      const state = docStates.get(key);
      if (state && state.chunks.length > 0) {
        rememberClosedStateCandidate(doc, state);
      }
      docStates.delete(key);
      loadedSidecarSignatures.delete(key);
      controlsProvider?.refresh();
    }),
    vscode.workspace.onDidSaveTextDocument((doc) => {
      autoSaveStateSidecar(doc);
    }),
    vscode.workspace.onDidChangeConfiguration((evt) => {
      if (evt.affectsConfiguration('binoculars')) {
        void restartBackend();
      }
    }),
    vscode.workspace.onDidChangeTextDocument((evt) => {
      const key = evt.document.uri.toString();
      const state = docStates.get(key);
      if (state && evt.contentChanges.length > 0) {
        state.stale = true;
        state.editedRanges = applyContentChangesToEditedRanges(state.editedRanges, evt.contentChanges);
        const visibleEditors = vscode.window.visibleTextEditors.filter((editor) => editor.document.uri.toString() === key);
        for (const editor of visibleEditors) {
          applyEditedDecorations(editor, state);
        }
        controlsProvider?.refresh();
      }
    }),
  );

  for (const doc of vscode.workspace.textDocuments) {
    maybeLoadStateSidecar(doc, 'activate');
  }
  if (vscode.window.activeTextEditor) {
    const active = vscode.window.activeTextEditor;
    const state = docStates.get(active.document.uri.toString());
    if (state) {
      applyDecorations(active, state);
    } else {
      clearAllDecorations(active);
    }
    updateStatusForEditor(active);
  } else {
    updateStatus('Ready. Run Binoculars: Analyze Chunk.');
  }
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
  const text = editor.document.getText();
  updateStatus('Analyzing chunk...');

  try {
    const result = await runWithBusyQuickPick('Binoculars: Analyze Chunk', 'Analyzing current chunk...', async () => {
      const client = await ensureBackend();
      return await client.analyzeDocument(text, inputLabel(editor.document));
    });
    const state: DocumentState = {
      chunks: [toChunkState(result)],
      nextChunkStart: result.next_chunk_start ?? result.chunk.analyzed_char_end,
      stale: false,
      editedRanges: [],
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

  updateStatus(`Analyzing next chunk from char ${start}...`);

  try {
    const result = await runWithBusyQuickPick(
      'Binoculars: Analyze Next Chunk',
      `Analyzing next chunk from char ${start}...`,
      async () => {
        const client = await ensureBackend();
        return await client.analyzeNextChunk(fullText, inputLabel(editor.document), start);
      },
    );
    mergeChunk(existing, toChunkState(result));
    existing.nextChunkStart = result.next_chunk_start ?? result.chunk.analyzed_char_end;
    existing.stale = false;
    existing.editedRanges = [];
    applyDecorations(editor, existing);
    updateStatusForEditor(editor);
  } catch (err) {
    showError(`Analyze Next failed: ${(err as Error).message}`);
  }
}

async function runWithBusyQuickPick<T>(title: string, placeholder: string, action: () => Promise<T>): Promise<T> {
  const picker = vscode.window.createQuickPick<vscode.QuickPickItem>();
  picker.title = title;
  picker.placeholder = placeholder;
  picker.busy = true;
  picker.enabled = false;
  picker.ignoreFocusOut = true;
  picker.show();
  try {
    return await action();
  } finally {
    picker.dispose();
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
  const span = resolveRewriteSpan(editor, state);
  if (!span) {
    showError('No scored span available for rewrite at cursor/selection.');
    return;
  }

  await runRewriteForSpan(editor, key, state, text, span.start, span.end);
}

async function runRewriteSelectionOrLine(): Promise<void> {
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
  const span = resolveSelectionOrLineSpan(editor);
  if (!span) {
    showError('No line or selection is available for rewrite at cursor position.');
    return;
  }

  await runRewriteForSpan(editor, key, state, text, span.start, span.end);
}

async function runRewriteForSpan(
  editor: vscode.TextEditor,
  stateKey: string,
  state: DocumentState,
  fullText: string,
  start: number,
  end: number,
): Promise<void> {
  const activeChunk = getActiveChunk(editor, fullText, state);
  const baseMetrics = activeChunk?.metrics;
  const client = await ensureBackend();
  const initialVersion = editor.document.version;
  const picker = vscode.window.createQuickPick<vscode.QuickPickItem>();
  let pickerDismissed = false;
  const dismissDisposable = picker.onDidHide(() => {
    pickerDismissed = true;
  });
  picker.title = 'Binoculars: Rewrite';
  picker.placeholder = 'Generating rewrite options...';
  picker.busy = true;
  picker.enabled = false;
  picker.ignoreFocusOut = true;
  picker.matchOnDescription = true;
  picker.matchOnDetail = true;
  picker.show();
  updateStatus('Generating rewrite options...');

  try {
    const result = await client.rewriteSpan(
      fullText,
      start,
      end,
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
    const options = (result.rewrites || []).slice(0, 3);
    if (options.length === 0) {
      showError('No rewrite options returned.');
      return;
    }
    if (pickerDismissed) {
      updateStatus('Rewrite selection canceled.');
      return;
    }
    picker.hide();
    updateStatus('Reviewing rewrite options...');
    const selected = await pickRewriteOptionWithWebview(result.source, options);
    if (!selected || !selected.text || !selected.text.trim()) {
      updateStatus('Rewrite selection canceled.');
      return;
    }

    if (editor.document.version !== initialVersion) {
      showError('Document changed while rewrite options were open. Re-run rewrite for current text.');
      return;
    }

    const range = new vscode.Range(positionAt(fullText, start), positionAt(fullText, end));
    await editor.edit((eb) => eb.replace(range, selected.text));
    const next = docStates.get(stateKey);
    if (next) {
      next.stale = true;
    }
    updateStatus('Rewrite applied. Re-run Analyze for exact B score.');

    const newDoc = editor.document.getText();
    const newPos = positionAt(newDoc, start + selected.text.length);
    editor.selection = new vscode.Selection(newPos, newPos);
  } catch (err) {
    showError(`Rewrite failed: ${(err as Error).message}`);
  } finally {
    dismissDisposable.dispose();
    picker.dispose();
  }
}

async function pickRewriteOptionWithWebview(source: string, options: RewriteOption[]): Promise<RewriteOption | undefined> {
  const picks = options.slice(0, 3);
  if (picks.length === 0) {
    return undefined;
  }
  const panel = vscode.window.createWebviewPanel(
    'binoculars.rewriteOptions',
    `Binoculars: Rewrite (${source})`,
    vscode.ViewColumn.Beside,
    {
      enableScripts: true,
      retainContextWhenHidden: false,
    },
  );
  panel.webview.html = renderRewriteOptionsHtml(panel.webview, source, picks);

  return await new Promise<RewriteOption | undefined>((resolve) => {
    let settled = false;
    const finish = (value: RewriteOption | undefined) => {
      if (settled) {
        return;
      }
      settled = true;
      msgDisposable.dispose();
      disposeDisposable.dispose();
      resolve(value);
    };
    const msgDisposable = panel.webview.onDidReceiveMessage((msg: unknown) => {
      if (!msg || typeof msg !== 'object') {
        return;
      }
      const payload = msg as { type?: string; index?: number };
      if (payload.type === 'apply' && typeof payload.index === 'number') {
        const idx = Math.max(0, Math.min(picks.length - 1, Math.floor(payload.index)));
        finish(picks[idx]);
        panel.dispose();
        return;
      }
      if (payload.type === 'cancel') {
        finish(undefined);
        panel.dispose();
      }
    });
    const disposeDisposable = panel.onDidDispose(() => finish(undefined));
  });
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
  const observerModelPath = resolvePathSetting(cfg.get<string>('models.observerGgufPath', ''));
  const performerModelPath = resolvePathSetting(cfg.get<string>('models.performerGgufPath', ''));
  const rewriteLlmConfigPath = resolveRewriteLlmConfigPath(cfg);

  if (!(backend as BackendClient)) {
    throw new Error('Backend client unavailable.');
  }

  if (!backendStarted) {
    await backend.start(pythonPath, bridgeScriptPath);
    backendStarted = true;
  }

  await backend.initialize(
    configPath,
    topK,
    textMaxTokensOverride,
    observerModelPath,
    performerModelPath,
    rewriteLlmConfigPath,
  );
  return backend;
}

function resolveRewriteLlmConfigPath(cfg: vscode.WorkspaceConfiguration): string {
  const enabled = cfg.get<boolean>('externalLlm.enabled', true);
  const configuredPath = resolvePathSetting(cfg.get<string>('externalLlm.configPath', ''));
  const endpoint = String(cfg.get<string>('externalLlm.endpoint', '') ?? '').trim();
  const model = String(cfg.get<string>('externalLlm.model', '') ?? '').trim();
  const temperature = cfg.get<number>('externalLlm.temperature', 0.7);
  const maxTokens = cfg.get<number>('externalLlm.maxTokens', 280);

  const hasInlineOverrides = endpoint.length > 0 || model.length > 0;
  if (enabled && !hasInlineOverrides) {
    return configuredPath;
  }

  const storageDir = extensionCtx?.globalStorageUri?.fsPath;
  if (!storageDir) {
    return configuredPath;
  }
  try {
    fs.mkdirSync(storageDir, { recursive: true });
  } catch {
    return configuredPath;
  }

  const runtimePath = path.join(storageDir, 'rewrite-llm.runtime.json');
  let payload: Record<string, unknown> = {};

  if (configuredPath && fs.existsSync(configuredPath)) {
    try {
      const raw = fs.readFileSync(configuredPath, 'utf8');
      const parsed = JSON.parse(raw) as unknown;
      if (parsed && typeof parsed === 'object') {
        payload = parsed as Record<string, unknown>;
      }
    } catch {
      payload = {};
    }
  }

  const llmObj =
    payload.llm && typeof payload.llm === 'object' ? { ...(payload.llm as Record<string, unknown>) } : {};
  llmObj.enabled = enabled;
  if (endpoint) {
    llmObj.endpoint_url = endpoint;
  }
  if (model) {
    llmObj.model = model;
  }
  llmObj.temperature = temperature;
  llmObj.max_tokens = maxTokens;
  payload.llm = llmObj;

  try {
    fs.writeFileSync(runtimePath, JSON.stringify(payload, null, 2), { encoding: 'utf8' });
    return runtimePath;
  } catch {
    return configuredPath;
  }
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

function isMarkdownSidecarEligible(doc: vscode.TextDocument): boolean {
  if (doc.uri.scheme !== 'file') {
    return false;
  }
  return path.extname(doc.uri.fsPath).toLowerCase() === '.md';
}

function sidecarStatePathForDocument(docPath: string): string {
  const abs = path.resolve(docPath);
  const parsed = path.parse(abs);
  return path.join(parsed.dir, `${parsed.name}.json`);
}

function sha256Hex(value: string): string {
  return crypto.createHash('sha256').update(value, 'utf8').digest('hex');
}

function cloneDocumentState(state: DocumentState): DocumentState {
  return {
    nextChunkStart: state.nextChunkStart,
    stale: state.stale,
    editedRanges: state.editedRanges.map((r) => ({ start: r.start, end: r.end })),
    chunks: state.chunks.map((chunk) => ({
      charStart: chunk.charStart,
      charEnd: chunk.charEnd,
      analyzedCharEnd: chunk.analyzedCharEnd,
      metrics: chunk.metrics
        ? {
            binoculars_score: chunk.metrics.binoculars_score,
            observer_logPPL: chunk.metrics.observer_logPPL,
            performer_logPPL: chunk.metrics.performer_logPPL,
            cross_logXPPL: chunk.metrics.cross_logXPPL,
            transitions: chunk.metrics.transitions,
          }
        : undefined,
      rows: chunk.rows.map((row) => ({
        paragraph_id: row.paragraph_id,
        char_start: row.char_start,
        char_end: row.char_end,
        logPPL: row.logPPL,
        delta_doc_logPPL_if_removed: row.delta_doc_logPPL_if_removed,
        excerpt: row.excerpt,
      })),
    })),
  };
}

function pruneRecentClosedStateCandidates(nowMs = Date.now()): void {
  for (const [textHash, candidate] of recentClosedStatesByTextHash.entries()) {
    if (nowMs - candidate.savedAtMs > RECENT_CLOSED_STATE_TTL_MS) {
      recentClosedStatesByTextHash.delete(textHash);
    }
  }
}

function rememberClosedStateCandidate(doc: vscode.TextDocument, state: DocumentState): void {
  if (!isMarkdownSidecarEligible(doc)) {
    return;
  }
  pruneRecentClosedStateCandidates();
  const textHash = sha256Hex(doc.getText());
  recentClosedStatesByTextHash.set(textHash, {
    savedAtMs: Date.now(),
    state: cloneDocumentState(state),
  });
}

function stateFromRecentClosedCandidate(doc: vscode.TextDocument): DocumentState | undefined {
  if (!isMarkdownSidecarEligible(doc)) {
    return undefined;
  }
  pruneRecentClosedStateCandidates();
  const textHash = sha256Hex(doc.getText());
  const candidate = recentClosedStatesByTextHash.get(textHash);
  if (!candidate) {
    return undefined;
  }
  return cloneDocumentState(candidate.state);
}

function maybeRestoreStateFromRecentClosed(doc: vscode.TextDocument, reason: 'open' | 'activate' | 'save'): boolean {
  if (!isMarkdownSidecarEligible(doc)) {
    return false;
  }
  const key = doc.uri.toString();
  if (docStates.has(key)) {
    return false;
  }
  const restored = stateFromRecentClosedCandidate(doc);
  if (!restored) {
    return false;
  }
  docStates.set(key, restored);
  const visibleEditors = vscode.window.visibleTextEditors.filter((editor) => editor.document.uri.toString() === key);
  for (const editor of visibleEditors) {
    applyDecorations(editor, restored);
  }
  if (vscode.window.activeTextEditor?.document.uri.toString() === key) {
    updateStatusForEditor(vscode.window.activeTextEditor);
  }
  controlsProvider?.refresh();
  output.appendLine(`[state] restored analysis from recent closed doc (${reason}): ${doc.uri.fsPath}`);
  return true;
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return undefined;
  }
  return value as Record<string, unknown>;
}

function asFiniteNumber(value: unknown): number | undefined {
  const n = Number(value);
  return Number.isFinite(n) ? n : undefined;
}

function asClampedInt(value: unknown, min: number, max: number): number | undefined {
  const n = asFiniteNumber(value);
  if (typeof n !== 'number') {
    return undefined;
  }
  return Math.max(min, Math.min(max, Math.trunc(n)));
}

function parseChunkMetrics(raw: unknown): ChunkMetrics | undefined {
  const obj = asRecord(raw);
  if (!obj) {
    return undefined;
  }
  const binocularsScore = asFiniteNumber(obj.binoculars_score);
  const observerLogPPL = asFiniteNumber(obj.observer_logPPL);
  const performerLogPPL = asFiniteNumber(obj.performer_logPPL);
  const crossLogXPPL = asFiniteNumber(obj.cross_logXPPL);
  const transitions = asClampedInt(obj.transitions, 0, Number.MAX_SAFE_INTEGER);
  if (
    typeof binocularsScore !== 'number' ||
    typeof observerLogPPL !== 'number' ||
    typeof performerLogPPL !== 'number' ||
    typeof crossLogXPPL !== 'number' ||
    typeof transitions !== 'number'
  ) {
    return undefined;
  }
  return {
    binoculars_score: binocularsScore,
    observer_logPPL: observerLogPPL,
    performer_logPPL: performerLogPPL,
    cross_logXPPL: crossLogXPPL,
    transitions,
  };
}

function parseParagraphRow(raw: unknown, textLen: number): ParagraphRow | undefined {
  const obj = asRecord(raw);
  if (!obj) {
    return undefined;
  }
  const charStart = asClampedInt(obj.char_start, 0, textLen);
  const charEnd = asClampedInt(obj.char_end, 0, textLen);
  if (typeof charStart !== 'number' || typeof charEnd !== 'number' || charEnd <= charStart) {
    return undefined;
  }
  const row: ParagraphRow = {
    char_start: charStart,
    char_end: charEnd,
  };
  const paragraphId = asClampedInt(obj.paragraph_id, 0, Number.MAX_SAFE_INTEGER);
  if (typeof paragraphId === 'number') {
    row.paragraph_id = paragraphId;
  }
  const logPPL = asFiniteNumber(obj.logPPL);
  if (typeof logPPL === 'number') {
    row.logPPL = logPPL;
  }
  const delta = asFiniteNumber(obj.delta_doc_logPPL_if_removed);
  if (typeof delta === 'number') {
    row.delta_doc_logPPL_if_removed = delta;
  }
  if (typeof obj.excerpt === 'string') {
    row.excerpt = obj.excerpt;
  }
  return row;
}

function parseChunkState(raw: unknown, textLen: number): ChunkState | undefined {
  const obj = asRecord(raw);
  if (!obj) {
    return undefined;
  }
  const charStart = asClampedInt(obj.char_start, 0, textLen);
  const rawCharEnd = asClampedInt(obj.char_end, 0, textLen);
  const analyzedCharEnd = asClampedInt(obj.analyzed_char_end ?? obj.char_end, 0, textLen);
  if (typeof charStart !== 'number' || typeof analyzedCharEnd !== 'number' || analyzedCharEnd <= charStart) {
    return undefined;
  }
  const charEnd = Math.max(analyzedCharEnd, typeof rawCharEnd === 'number' ? rawCharEnd : analyzedCharEnd);
  const rowsRaw = Array.isArray(obj.rows) ? obj.rows : [];
  const rows = rowsRaw
    .map((entry) => parseParagraphRow(entry, textLen))
    .filter((row): row is ParagraphRow => typeof row !== 'undefined');
  return {
    charStart,
    charEnd,
    analyzedCharEnd,
    metrics: parseChunkMetrics(obj.metrics),
    rows,
  };
}

function computeContiguousCoverage(chunks: ChunkState[], textLen: number): number {
  const merged = mergeIntervals(
    chunks
      .map((chunk) => ({
        start: Math.max(0, Math.min(textLen, chunk.charStart)),
        end: Math.max(0, Math.min(textLen, chunk.analyzedCharEnd)),
      }))
      .filter((x) => x.end > x.start),
  );
  let contiguous = 0;
  for (const interval of merged) {
    if (interval.start > contiguous) {
      break;
    }
    contiguous = Math.max(contiguous, interval.end);
  }
  return Math.max(0, Math.min(textLen, contiguous));
}

function documentStateFromPersisted(rawState: Record<string, unknown>, textLen: number): DocumentState | undefined {
  const rawChunks = Array.isArray(rawState.analysis_chunks) ? rawState.analysis_chunks : [];
  const chunks = rawChunks
    .map((entry) => parseChunkState(entry, textLen))
    .filter((chunk): chunk is ChunkState => typeof chunk !== 'undefined')
    .sort((a, b) => a.charStart - b.charStart);
  if (chunks.length === 0) {
    return undefined;
  }
  const contiguousCoverage = computeContiguousCoverage(chunks, textLen);
  const rawCoverage = asClampedInt(rawState.analysis_covered_until, 0, textLen);
  const nextChunkStart = Math.max(contiguousCoverage, typeof rawCoverage === 'number' ? rawCoverage : contiguousCoverage);
  const editedRangesRaw = Array.isArray(rawState.edited_ranges) ? rawState.edited_ranges : [];
  const editedRanges = editedRangesRaw
    .map((entry) => asRecord(entry))
    .filter((entry): entry is Record<string, unknown> => typeof entry !== 'undefined')
    .map((entry) => ({
      start: asClampedInt(entry.start, 0, textLen),
      end: asClampedInt(entry.end, 0, textLen),
    }))
    .filter((entry): entry is { start: number; end: number } => typeof entry.start === 'number' && typeof entry.end === 'number')
    .filter((entry) => entry.end > entry.start);
  return {
    chunks,
    nextChunkStart,
    stale: Boolean(rawState.b_score_stale),
    editedRanges,
  };
}

function serializeChunkForPersistedState(chunk: ChunkState, index: number): Record<string, unknown> {
  return {
    id: index + 1,
    char_start: chunk.charStart,
    char_end: chunk.charEnd,
    analyzed_char_end: chunk.analyzedCharEnd,
    metrics: chunk.metrics
      ? {
          binoculars_score: chunk.metrics.binoculars_score,
          observer_logPPL: chunk.metrics.observer_logPPL,
          performer_logPPL: chunk.metrics.performer_logPPL,
          cross_logXPPL: chunk.metrics.cross_logXPPL,
          transitions: chunk.metrics.transitions,
        }
      : {},
    profile: null,
    rows: chunk.rows.map((row) => ({
      paragraph_id: row.paragraph_id,
      char_start: row.char_start,
      char_end: row.char_end,
      logPPL: row.logPPL,
      delta_doc_logPPL_if_removed: row.delta_doc_logPPL_if_removed,
      excerpt: row.excerpt,
    })),
  };
}

function buildPersistedSidecarPayload(
  docPath: string,
  textSnapshot: string,
  state: DocumentState | undefined,
): PersistedSidecarPayload {
  const textLen = textSnapshot.length;
  const chunks = state?.chunks ?? [];
  const hasAnalysis = chunks.length > 0;
  const coverage = hasAnalysis ? computeContiguousCoverage(chunks, textLen) : 0;
  const nextAvailable = hasAnalysis && coverage < textLen;
  const metrics = [...chunks].reverse().find((chunk) => chunk.metrics)?.metrics;
  const lastAnalysisMetrics = metrics
    ? {
        binoculars_score: metrics.binoculars_score,
        observer_logPPL: metrics.observer_logPPL,
        performer_logPPL: metrics.performer_logPPL,
        cross_logXPPL: metrics.cross_logXPPL,
        transitions: metrics.transitions,
      }
    : null;
  const lastAnalysisStatusCore = metrics
    ? `B ${formatStatusMetric(metrics.binoculars_score)} | Obs ${formatStatusMetric(metrics.observer_logPPL)} | Cross ${formatStatusMetric(metrics.cross_logXPPL)}`
    : '';
  const statePayload: Record<string, unknown> = {
    baseline_text: textSnapshot,
    prev_text: textSnapshot,
    has_analysis: hasAnalysis,
    b_score_stale: Boolean(state?.stale),
    last_b_score: metrics?.binoculars_score ?? null,
    last_analysis_status_core: lastAnalysisStatusCore,
    last_analysis_metrics: lastAnalysisMetrics,
    analyzed_char_end: coverage,
    analysis_chunks: chunks.map((chunk, idx) => serializeChunkForPersistedState(chunk, idx)),
    analysis_chunk_id_seq: chunks.length,
    analysis_covered_until: coverage,
    analysis_next_available: nextAvailable,
    edited_ranges: (state?.editedRanges ?? []).map((r) => ({
      start: r.start,
      end: r.end,
    })),
  };
  return {
    binoculars_gui_state: true,
    version: 1,
    saved_at: new Date().toISOString().replace(/\.\d{3}Z$/, 'Z'),
    document_path: path.resolve(docPath),
    document_basename: path.basename(docPath),
    text_sha256: sha256Hex(textSnapshot),
    state: statePayload,
  };
}

function maybeLoadStateSidecar(doc: vscode.TextDocument, reason: 'open' | 'activate'): void {
  if (!isMarkdownSidecarEligible(doc)) {
    return;
  }
  const key = doc.uri.toString();
  const textSnapshot = doc.getText();
  const textHash = sha256Hex(textSnapshot);
  const sidecarPath = sidecarStatePathForDocument(doc.uri.fsPath);
  if (!fs.existsSync(sidecarPath)) {
    return;
  }

  let sidecarRaw: string;
  try {
    sidecarRaw = fs.readFileSync(sidecarPath, 'utf8');
  } catch (err) {
    output.appendLine(`[state] sidecar read failed: ${sidecarPath} (${(err as Error).message})`);
    return;
  }
  const signature = `${textHash}:${sha256Hex(sidecarRaw)}`;
  if (loadedSidecarSignatures.get(key) === signature) {
    return;
  }

  let payloadUnknown: unknown;
  try {
    payloadUnknown = JSON.parse(sidecarRaw);
  } catch (err) {
    output.appendLine(`[state] sidecar JSON parse failed: ${sidecarPath} (${(err as Error).message})`);
    loadedSidecarSignatures.set(key, signature);
    return;
  }
  const payload = asRecord(payloadUnknown);
  if (!payload || payload.binoculars_gui_state !== true) {
    output.appendLine(`[state] sidecar ignored (unrecognized format): ${sidecarPath}`);
    loadedSidecarSignatures.set(key, signature);
    return;
  }

  const expectedHash = String(payload.text_sha256 ?? '').trim();
  if (expectedHash && expectedHash !== textHash) {
    output.appendLine(`[state] sidecar ignored (text hash mismatch): ${sidecarPath}`);
    loadedSidecarSignatures.set(key, signature);
    return;
  }

  const rawState = asRecord(payload.state);
  if (!rawState) {
    output.appendLine(`[state] sidecar ignored (missing state object): ${sidecarPath}`);
    loadedSidecarSignatures.set(key, signature);
    return;
  }

  const persistedState = documentStateFromPersisted(rawState, textSnapshot.length);
  loadedSidecarSignatures.set(key, signature);
  if (!persistedState) {
    return;
  }

  docStates.set(key, persistedState);
  const visibleEditors = vscode.window.visibleTextEditors.filter((editor) => editor.document.uri.toString() === key);
  for (const editor of visibleEditors) {
    applyDecorations(editor, persistedState);
  }
  if (vscode.window.activeTextEditor?.document.uri.toString() === key) {
    updateStatusForEditor(vscode.window.activeTextEditor);
  }
  controlsProvider?.refresh();
  output.appendLine(`[state] loaded sidecar (${reason}): ${sidecarPath}`);
}

function autoSaveStateSidecar(doc: vscode.TextDocument): void {
  if (!isMarkdownSidecarEligible(doc)) {
    return;
  }
  const key = doc.uri.toString();
  const textSnapshot = doc.getText();
  if (!docStates.has(key)) {
    maybeRestoreStateFromRecentClosed(doc, 'save');
  }
  const state = docStates.get(key);
  const sidecarPath = sidecarStatePathForDocument(doc.uri.fsPath);
  const payload = buildPersistedSidecarPayload(doc.uri.fsPath, textSnapshot, state);
  const serialized = JSON.stringify(payload, null, 2);
  try {
    fs.writeFileSync(sidecarPath, serialized, { encoding: 'utf8' });
    loadedSidecarSignatures.set(key, `${sha256Hex(textSnapshot)}:${sha256Hex(serialized)}`);
  } catch (err) {
    const message = `State sidecar save failed: ${(err as Error).message}`;
    output.appendLine(`[state] ${message} (${sidecarPath})`);
    void vscode.window.showWarningMessage(message);
  }
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

function mergeEditedRanges(ranges: EditedRange[]): EditedRange[] {
  if (ranges.length === 0) {
    return [];
  }
  const sorted = [...ranges]
    .filter((r) => Number.isFinite(r.start) && Number.isFinite(r.end) && r.end > r.start)
    .sort((a, b) => a.start - b.start);
  if (sorted.length === 0) {
    return [];
  }
  const out: EditedRange[] = [{ start: sorted[0].start, end: sorted[0].end }];
  for (let i = 1; i < sorted.length; i += 1) {
    const cur = sorted[i];
    const prev = out[out.length - 1];
    if (cur.start <= prev.end) {
      prev.end = Math.max(prev.end, cur.end);
    } else {
      out.push({ start: cur.start, end: cur.end });
    }
  }
  return out;
}

function shiftEditedRangesForSplice(ranges: EditedRange[], start: number, end: number, insertedLen: number): EditedRange[] {
  const removedLen = Math.max(0, end - start);
  const delta = insertedLen - removedLen;
  const out: EditedRange[] = [];
  for (const r of ranges) {
    if (r.end <= start) {
      out.push({ start: r.start, end: r.end });
      continue;
    }
    if (r.start >= end) {
      out.push({ start: r.start + delta, end: r.end + delta });
      continue;
    }
    const newStart = Math.min(r.start, start);
    const tailEnd = r.end > end ? r.end + delta : start + insertedLen;
    const newEnd = Math.max(start + insertedLen, tailEnd);
    if (newEnd > newStart) {
      out.push({ start: newStart, end: newEnd });
    }
  }
  return mergeEditedRanges(out);
}

function applyContentChangesToEditedRanges(
  prevRanges: EditedRange[],
  changes: readonly vscode.TextDocumentContentChangeEvent[],
): EditedRange[] {
  if (!changes || changes.length === 0) {
    return prevRanges;
  }
  const sorted = [...changes].sort((a, b) => a.rangeOffset - b.rangeOffset);
  let delta = 0;
  let ranges = mergeEditedRanges(prevRanges);
  for (const ch of sorted) {
    const originalStart = Math.max(0, ch.rangeOffset);
    const originalEnd = Math.max(originalStart, ch.rangeOffset + Math.max(0, ch.rangeLength));
    const start = Math.max(0, originalStart + delta);
    const end = Math.max(start, originalEnd + delta);
    const insertedLen = ch.text.length;
    ranges = shiftEditedRangesForSplice(ranges, start, end, insertedLen);
    if (insertedLen > 0) {
      ranges.push({ start, end: start + insertedLen });
      ranges = mergeEditedRanges(ranges);
    }
    delta += insertedLen - (originalEnd - originalStart);
  }
  return ranges;
}

function editedRangesToDecorationRanges(docText: string, ranges: EditedRange[]): vscode.Range[] {
  const docLen = docText.length;
  return mergeEditedRanges(ranges)
    .map((r) => ({
      start: Math.max(0, Math.min(docLen, r.start)),
      end: Math.max(0, Math.min(docLen, r.end)),
    }))
    .filter((r) => r.end > r.start)
    .map((r) => new vscode.Range(positionAt(docText, r.start), positionAt(docText, r.end)));
}

function applyEditedDecorations(editor: vscode.TextEditor, state: DocumentState): void {
  const docText = editor.document.getText();
  const ranges = editedRangesToDecorationRanges(docText, state.editedRanges);
  editor.setDecorations(editedDecoration, ranges);
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
  applyEditedDecorations(editor, state);
}

function clearAllDecorations(editor: vscode.TextEditor): void {
  editor.setDecorations(lowDecoration, []);
  editor.setDecorations(highDecoration, []);
  editor.setDecorations(unscoredDecoration, []);
  editor.setDecorations(editedDecoration, []);
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

function resolveSelectionOrLineSpan(editor: vscode.TextEditor): { start: number; end: number } | undefined {
  const docText = editor.document.getText();
  const sel = editor.selection;
  if (!sel.isEmpty) {
    const start = offsetAt(docText, sel.start);
    const end = offsetAt(docText, sel.end);
    if (end > start) {
      return { start: Math.min(start, end), end: Math.max(start, end) };
    }
  }
  const line = editor.selection.active.line;
  if (line < 0 || line >= editor.document.lineCount) {
    return undefined;
  }
  const lineRange = editor.document.lineAt(line).range;
  const start = offsetAt(docText, lineRange.start);
  const end = offsetAt(docText, lineRange.end);
  if (end <= start) {
    return undefined;
  }
  return { start, end };
}

function hoverForPosition(document: vscode.TextDocument, position: vscode.Position): vscode.Hover | undefined {
  const key = document.uri.toString();
  const state = docStates.get(key);
  if (!state || state.chunks.length === 0) {
    return undefined;
  }
  const text = document.getText();
  const charOffset = offsetAt(text, position);
  let bestMatch:
    | {
        chunk: ChunkState;
        row: ParagraphRow;
      }
    | undefined;
  let bestSpan = Number.MAX_SAFE_INTEGER;

  for (const chunk of state.chunks) {
    for (const row of chunk.rows) {
      const rowStart = Math.max(0, Number(row.char_start ?? 0));
      const rowEnd = Math.max(rowStart, Number(row.char_end ?? rowStart));
      if (charOffset < rowStart || charOffset >= rowEnd) {
        continue;
      }
      const span = rowEnd - rowStart;
      if (!bestMatch || span < bestSpan) {
        bestMatch = { chunk, row };
        bestSpan = span;
      }
    }
  }

  if (!bestMatch) {
    return undefined;
  }

  const delta = Number(bestMatch.row.delta_doc_logPPL_if_removed ?? NaN);
  const chunkB = Number(bestMatch.chunk.metrics?.binoculars_score ?? NaN);
  const nextChunkB = Number.isFinite(delta) && Number.isFinite(chunkB) ? chunkB + delta : NaN;
  const paragraphLogPPL = Number(bestMatch.row.logPPL ?? NaN);
  const label = Number.isFinite(delta) ? (delta >= 0 ? 'LOW' : 'HIGH') : 'UNKNOWN';
  const lines: string[] = [
    `**Binoculars ${label} Perplexity Segment**`,
    '',
    `Delta if removed: \`${formatSignedMetric(delta)}\` (Chunk changes to \`${formatSignedMetric(nextChunkB)}\`)`,
    `Paragraph LogPPL: \`${formatStatusMetric(paragraphLogPPL)}\``,
  ];
  const md = new vscode.MarkdownString(lines.join('  \n'));
  md.isTrusted = false;
  const range = new vscode.Range(positionAt(text, bestMatch.row.char_start), positionAt(text, bestMatch.row.char_end));
  return new vscode.Hover(md, range);
}

function updateStatusForEditor(editor: vscode.TextEditor): void {
  const key = editor.document.uri.toString();
  const state = docStates.get(key);
  if (!state || state.chunks.length === 0) {
    updateStatus('Ready. Run Binoculars: Analyze Chunk.');
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
    `B ${formatStatusMetric(chunk.metrics.binoculars_score)} | Obs ${formatStatusMetric(chunk.metrics.observer_logPPL)} | Cross ${formatStatusMetric(chunk.metrics.cross_logXPPL)}${staleSuffix}${moreSuffix}`,
  );
}

function updateStatus(message: string): void {
  lastStatusMessage = message;
  statusBar.text = `$(telescope) Binoculars: ${message}`;
  controlsProvider?.refresh();
}

function showError(message: string): void {
  void vscode.window.showErrorMessage(message);
  updateStatus(message);
}

function inputLabel(doc: vscode.TextDocument): string {
  return doc.uri.scheme === 'file' ? doc.uri.fsPath : '<unsaved>';
}

function formatStatusMetric(value: number): string {
  if (!Number.isFinite(value)) {
    return 'n/a';
  }
  return value.toFixed(DISPLAY_DECIMALS);
}

function formatSignedMetric(value: number): string {
  if (!Number.isFinite(value)) {
    return 'n/a';
  }
  return `${value >= 0 ? '+' : ''}${value.toFixed(DISPLAY_DECIMALS)}`;
}

function formatApprox(approxB?: number, deltaB?: number): string {
  if (typeof approxB !== 'number' || Number.isNaN(approxB)) {
    return 'rewrite option';
  }
  const d = typeof deltaB === 'number' && Number.isFinite(deltaB) ? ` (${formatSignedMetric(deltaB)})` : '';
  return `approx B ${approxB.toFixed(DISPLAY_DECIMALS)}${d}`;
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function createNonce(): string {
  return crypto.randomBytes(12).toString('base64');
}

function renderRewriteOptionsHtml(webview: vscode.Webview, source: string, options: RewriteOption[]): string {
  const nonce = createNonce();
  const cardHtml = options
    .map((rw: RewriteOption, idx: number) => {
      const heading = escapeHtml(`[${idx + 1}] ${formatApprox(rw.approx_B, rw.delta_B)}`);
      const text = escapeHtml(String(rw.text || ''));
      return `
        <section class="card">
          <div class="card-head">
            <h3>${heading}</h3>
            <button class="apply" data-index="${idx}" title="Apply rewrite ${idx + 1}">Apply ${idx + 1}</button>
          </div>
          <pre>${text}</pre>
        </section>
      `;
    })
    .join('\n');
  const payload = JSON.stringify({ count: options.length }).replace(/</g, '\\u003c');
  const safeSource = escapeHtml(source);
  return `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';" />
    <title>Binoculars Rewrite Options</title>
    <style>
      :root {
        color-scheme: dark;
      }
      body {
        margin: 0;
        padding: 12px 14px 16px;
        font-family: var(--vscode-font-family);
        font-size: var(--vscode-font-size);
        color: var(--vscode-foreground);
        background: var(--vscode-editor-background);
      }
      .top {
        position: sticky;
        top: 0;
        z-index: 20;
        background: linear-gradient(var(--vscode-editor-background), var(--vscode-editor-background));
        padding-bottom: 10px;
        margin-bottom: 10px;
        border-bottom: 1px solid var(--vscode-editorWidget-border);
      }
      .title {
        font-size: 1.05rem;
        font-weight: 700;
      }
      .meta {
        margin-top: 4px;
        color: var(--vscode-descriptionForeground);
      }
      .actions {
        margin-top: 8px;
        display: flex;
        gap: 8px;
        align-items: center;
      }
      .hint {
        color: var(--vscode-descriptionForeground);
      }
      .card {
        border: 1px solid var(--vscode-editorWidget-border);
        border-radius: 8px;
        padding: 10px;
        margin: 0 0 10px;
        background: color-mix(in srgb, var(--vscode-editor-background) 82%, var(--vscode-sideBar-background) 18%);
      }
      .card-head {
        display: flex;
        gap: 10px;
        align-items: center;
        justify-content: space-between;
      }
      h3 {
        margin: 0;
        font-size: 1rem;
      }
      pre {
        margin: 10px 0 0;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        line-height: 1.35;
        font-family: var(--vscode-editor-font-family);
        background: color-mix(in srgb, var(--vscode-editor-background) 76%, var(--vscode-panel-background) 24%);
        border: 1px solid var(--vscode-editorWidget-border);
        border-radius: 6px;
        padding: 10px;
        max-height: 260px;
        overflow: auto;
      }
      button {
        border: 1px solid var(--vscode-button-border, transparent);
        border-radius: 6px;
        background: var(--vscode-button-background);
        color: var(--vscode-button-foreground);
        padding: 4px 10px;
        cursor: pointer;
        font-size: 0.92rem;
      }
      button:hover {
        background: var(--vscode-button-hoverBackground);
      }
      #cancel {
        background: transparent;
        color: var(--vscode-foreground);
        border-color: var(--vscode-editorWidget-border);
      }
    </style>
  </head>
  <body>
    <div class="top">
      <div class="title">Binoculars: Rewrite (${safeSource})</div>
      <div class="meta">Full rewrite text shown below. Choose an option to apply.</div>
      <div class="actions">
        <button id="cancel" title="Cancel rewrite selection">Cancel</button>
        <span class="hint">Keyboard: 1 / 2 / 3 apply, Esc cancel</span>
      </div>
    </div>
    ${cardHtml}
    <script nonce="${nonce}">
      const vscode = acquireVsCodeApi();
      const state = ${payload};
      for (const el of document.querySelectorAll('.apply')) {
        el.addEventListener('click', () => {
          const idx = Number(el.getAttribute('data-index'));
          vscode.postMessage({ type: 'apply', index: idx });
        });
      }
      const cancel = document.getElementById('cancel');
      if (cancel) {
        cancel.addEventListener('click', () => vscode.postMessage({ type: 'cancel' }));
      }
      window.addEventListener('keydown', (ev) => {
        if (ev.key === 'Escape' || ev.key.toLowerCase() === 'q') {
          ev.preventDefault();
          vscode.postMessage({ type: 'cancel' });
          return;
        }
        const digit = Number(ev.key);
        if (Number.isInteger(digit) && digit >= 1 && digit <= state.count) {
          ev.preventDefault();
          vscode.postMessage({ type: 'apply', index: digit - 1 });
        }
      });
    </script>
  </body>
</html>`;
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
