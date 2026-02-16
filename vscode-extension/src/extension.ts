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
  rewriteRanges: EditedRange[];
  priorLowRanges: EditedRange[];
  priorHighRanges: EditedRange[];
  priorChunkB?: number;
  forecastEstimate?: {
    chunkStart: number;
    docVersion: number;
    b: number;
  };
  forecastPending?: boolean;
}

interface RenderPalette {
  lowColor: string;
  highColor: string;
  lowMinorColor: string;
  highMinorColor: string;
  priorLowBg: string;
  priorHighBg: string;
  unscoredColor: string;
  unscoredOpacity: string;
}

interface HoverDecision {
  hover: vscode.Hover;
  isMinorContributor: boolean;
  segmentStart: number;
  segmentEnd: number;
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
  docKey: string;
  state: DocumentState;
}

const DISPLAY_DECIMALS = 5;
const LIVE_ESTIMATE_DEBOUNCE_MS = 900;
const HOVER_TYPING_SUPPRESS_MS = 1300;
const HOVER_SAME_SEGMENT_SUPPRESS_MS = 900;
const HOVER_CONTRIBUTOR_DELAY_MS = 1500;

const docStates = new Map<string, DocumentState>();
const loadedSidecarSignatures = new Map<string, string>();
const recentClosedStatesByTextHash = new Map<string, RecentClosedStateCandidate>();
const recentClosedHashByDocKey = new Map<string, string>();
const liveEstimateTimers = new Map<string, NodeJS.Timeout>();
const liveEstimateEpochs = new Map<string, number>();
const liveEstimateRecoverAttempts = new Map<string, number>();
const recentTypingActivity = new Map<string, { atMs: number; start: number; end: number }>();
const hoverSeenSegments = new Map<string, { start: number; end: number; lastSeenMs: number }>();
const hoverDelayGates = new Map<
  string,
  { start: number; end: number; anchorOffset: number; readyAtMs: number; timer?: NodeJS.Timeout }
>();
const RECENT_CLOSED_STATE_TTL_MS = 6 * 60 * 60 * 1000;
let backend: BackendClient | undefined;
let backendStarted = false;
let statusBar: vscode.StatusBarItem;
let output: vscode.OutputChannel;
let lastStatusMessage = 'Binoculars Ready. Select Analyze Chunk to begin.';
let controlsProvider: BinocularsControlsProvider | undefined;
let renderPalette: RenderPalette;
let extensionCtx: vscode.ExtensionContext;
let runtimeColorizationEnabled = true;
let foregroundBusyOperationCount = 0;

// Is Extension Enabled.
function isExtensionEnabled(): boolean {
  // Global enablement switch shared by menus, status, and bridge operations.
  const cfg = vscode.workspace.getConfiguration('binoculars');
  return cfg.get<boolean>('enabled', true);
}

// Set Enablement Context.
async function setEnablementContext(enabled: boolean): Promise<void> {
  await vscode.commands.executeCommand('setContext', 'binoculars:isEnabled', enabled);
}

// Has Next Chunk Available.
function hasNextChunkAvailable(editor: vscode.TextEditor | undefined): boolean {
  if (!editor || !isExtensionEnabled()) {
    return false;
  }
  const textLen = editor.document.getText().length;
  if (textLen <= 0) {
    return false;
  }
  const state = docStates.get(editor.document.uri.toString());
  if (!state || state.chunks.length === 0) {
    return false;
  }
  const covered = Math.max(0, state.nextChunkStart);
  return covered < textLen;
}

// Has Analysis For Editor.
function hasAnalysisForEditor(editor: vscode.TextEditor | undefined): boolean {
  if (!editor || !isExtensionEnabled()) {
    return false;
  }
  const state = docStates.get(editor.document.uri.toString());
  return Boolean(state && state.chunks.length > 0);
}

// Can Analyze All.
function canAnalyzeAll(editor: vscode.TextEditor | undefined): boolean {
  if (!editor || !isExtensionEnabled()) {
    return false;
  }
  const textLen = editor.document.getText().length;
  if (textLen <= 0) {
    return false;
  }
  const state = docStates.get(editor.document.uri.toString());
  if (!state || state.chunks.length === 0) {
    return false;
  }
  const covered = Math.max(0, state.nextChunkStart);
  return covered < textLen;
}

// Has Priors For Editor.
function hasPriorsForEditor(editor: vscode.TextEditor | undefined): boolean {
  if (!editor || !isExtensionEnabled()) {
    return false;
  }
  const state = docStates.get(editor.document.uri.toString());
  if (!state) {
    return false;
  }
  return (state.priorLowRanges?.length ?? 0) > 0 || (state.priorHighRanges?.length ?? 0) > 0;
}

// Refresh Analyze Next Context.
async function refreshAnalyzeNextContext(editor?: vscode.TextEditor): Promise<void> {
  // Keep command/menu contexts aligned with the active editor state.
  const active = editor ?? vscode.window.activeTextEditor;
  await vscode.commands.executeCommand('setContext', 'binoculars.hasNextChunk', hasNextChunkAvailable(active));
  await vscode.commands.executeCommand('setContext', 'binoculars.hasAnalysis', hasAnalysisForEditor(active));
  await vscode.commands.executeCommand('setContext', 'binoculars.canAnalyzeAll', canAnalyzeAll(active));
  await vscode.commands.executeCommand('setContext', 'binoculars.hasPriors', hasPriorsForEditor(active));
}

// Active Editor For Doc Key.
function activeEditorForDocKey(key: string): vscode.TextEditor | undefined {
  const active = vscode.window.activeTextEditor;
  if (active && active.document.uri.toString() === key) {
    return active;
  }
  return vscode.window.visibleTextEditors.find((editor) => editor.document.uri.toString() === key);
}

// Note Typing Activity.
function noteTypingActivity(docKey: string, changes: readonly vscode.TextDocumentContentChangeEvent[], docLen: number): void {
  // Record most recent edit span to suppress hover popups while the user types.
  if (changes.length === 0) {
    return;
  }
  let start = Number.MAX_SAFE_INTEGER;
  let end = 0;
  for (const ch of changes) {
    const s = Math.max(0, Math.min(docLen, Number(ch.rangeOffset ?? 0)));
    const e = Math.max(s, Math.min(docLen, s + String(ch.text ?? '').length));
    start = Math.min(start, s);
    end = Math.max(end, e);
  }
  if (!Number.isFinite(start) || start === Number.MAX_SAFE_INTEGER) {
    return;
  }
  recentTypingActivity.set(docKey, {
    atMs: Date.now(),
    start,
    end: Math.max(start, end),
  });
}

// Should Suppress Hover During Typing.
function shouldSuppressHoverDuringTyping(document: vscode.TextDocument, charOffset: number): boolean {
  // Suppress hover near fresh edits for a short window to reduce UI interference.
  const key = document.uri.toString();
  const activity = recentTypingActivity.get(key);
  if (!activity) {
    return false;
  }
  const ageMs = Date.now() - activity.atMs;
  if (ageMs > HOVER_TYPING_SUPPRESS_MS) {
    recentTypingActivity.delete(key);
    return false;
  }
  const padding = 120;
  return charOffset >= Math.max(0, activity.start - padding) && charOffset <= activity.end + padding;
}

function clearHoverDelayGate(docKey: string): void {
  const gate = hoverDelayGates.get(docKey);
  if (gate?.timer) {
    clearTimeout(gate.timer);
  }
  hoverDelayGates.delete(docKey);
}

function scheduleHoverGateReveal(docKey: string, start: number, end: number, anchorOffset: number, delayMs: number): void {
  const timer = setTimeout(() => {
    const active = vscode.window.activeTextEditor;
    if (!active || active.document.uri.toString() !== docKey) {
      return;
    }
    const text = active.document.getText();
    const normalizedStart = Math.max(0, Math.min(text.length, start));
    const normalizedEnd = Math.max(normalizedStart, Math.min(text.length, end));
    if (normalizedEnd <= normalizedStart) {
      return;
    }
    const clampedAnchor = Math.max(normalizedStart, Math.min(normalizedEnd - 1, anchorOffset));
    if (shouldSuppressHoverDuringTyping(active.document, clampedAnchor)) {
      return;
    }
    const cursorOffset = offsetAt(text, active.selection.active);
    const cursorInsideSegment = cursorOffset >= normalizedStart && cursorOffset < normalizedEnd;
    const anchorPos = positionAt(text, clampedAnchor);
    const targetSelection = new vscode.Selection(anchorPos, anchorPos);

    let shouldRestoreSelection = false;
    let priorSelections: vscode.Selection[] | undefined;
    if (!cursorInsideSegment && active.selections.length === 1 && active.selection.isEmpty) {
      const lineVisible = active.visibleRanges.some((range) => anchorPos.line >= range.start.line && anchorPos.line <= range.end.line);
      if (!lineVisible) {
        return;
      }
      priorSelections = active.selections.map((sel) => new vscode.Selection(sel.start, sel.end));
      if (!active.selection.isEqual(targetSelection)) {
        active.selections = [targetSelection];
        shouldRestoreSelection = true;
      }
    }

    const restoreIfNeeded = (): void => {
      if (!shouldRestoreSelection || !priorSelections) {
        return;
      }
      const latestEditor = vscode.window.activeTextEditor;
      if (!latestEditor || latestEditor.document.uri.toString() !== docKey) {
        return;
      }
      // Only restore if the temporary target selection is still active.
      if (latestEditor.selections.length === 1 && latestEditor.selection.isEqual(targetSelection)) {
        latestEditor.selections = priorSelections;
      }
    };
    void vscode.commands.executeCommand('editor.action.showHover').then(
      () => restoreIfNeeded(),
      () => restoreIfNeeded(),
    );
  }, Math.max(0, delayMs));
  hoverDelayGates.set(docKey, {
    start,
    end,
    anchorOffset,
    readyAtMs: Date.now() + Math.max(0, delayMs),
    timer,
  });
}

// Clear Live Estimate Timer.
function clearLiveEstimateTimer(key: string): void {
  const t = liveEstimateTimers.get(key);
  if (t) {
    clearTimeout(t);
    liveEstimateTimers.delete(key);
  }
}

// Schedule Live Estimate.
function scheduleLiveEstimate(key: string): void {
  // Debounce live estimate requests; only stale analyzed docs are eligible.
  if (!isExtensionEnabled() || foregroundBusyOperationCount > 0) {
    return;
  }
  const editor = activeEditorForDocKey(key);
  const state = docStates.get(key);
  if (!editor || !state || !state.stale || state.chunks.length === 0) {
    return;
  }

  clearLiveEstimateTimer(key);
  const nextEpoch = (liveEstimateEpochs.get(key) ?? 0) + 1;
  liveEstimateEpochs.set(key, nextEpoch);
  state.forecastPending = true;
  updateStatusForEditor(editor);
  const timer = setTimeout(() => {
    void computeLiveEstimate(key, nextEpoch);
  }, LIVE_ESTIMATE_DEBOUNCE_MS);
  liveEstimateTimers.set(key, timer);
}

// Compute Live Estimate.
async function computeLiveEstimate(key: string, epoch: number): Promise<void> {
  // Observer-only directional forecast used for stale chunk status updates.
  if (!isExtensionEnabled() || foregroundBusyOperationCount > 0) {
    return;
  }
  if (liveEstimateEpochs.get(key) !== epoch) {
    return;
  }
  clearLiveEstimateTimer(key);
  const editor = activeEditorForDocKey(key);
  const state = docStates.get(key);
  if (!editor || !state || !state.stale || state.chunks.length === 0) {
    return;
  }
  const fullText = editor.document.getText();
  const cursorOffset = offsetAt(fullText, editor.selection.active);
  const orderedChunks = [...state.chunks].sort((a, b) => a.charStart - b.charStart);
  const cursorChunk = orderedChunks.find(
    (chunk) => cursorOffset >= chunk.charStart && cursorOffset < chunk.analyzedCharEnd,
  );
  const activeChunk = cursorChunk ?? getActiveChunk(editor, fullText, state);
  if (!activeChunk || cursorOffset < activeChunk.charStart || cursorOffset >= activeChunk.analyzedCharEnd) {
    state.forecastPending = false;
    updateStatusForEditor(editor);
    return;
  }

  const chunkStart = Math.max(0, activeChunk.charStart);
  const chunkEnd = Math.max(chunkStart, Math.min(fullText.length, activeChunk.analyzedCharEnd));
  const baseCross = Number(activeChunk.metrics?.cross_logXPPL ?? NaN);
  if (!Number.isFinite(baseCross) || baseCross === 0) {
    state.forecastPending = false;
    state.forecastEstimate = undefined;
    updateStatusForEditor(editor);
    return;
  }
  const version = editor.document.version;
  try {
    const client = await ensureBackend();
    const result = await client.estimateLiveB(fullText, inputLabel(editor.document), chunkStart, chunkEnd, baseCross);
    if (liveEstimateEpochs.get(key) !== epoch) {
      return;
    }
    const liveState = docStates.get(key);
    const liveEditor = activeEditorForDocKey(key);
    if (!liveState || !liveEditor) {
      return;
    }
    if (liveEditor.document.version !== version) {
      return;
    }
    const approxB = Number(result.approx_b ?? NaN);
    const observerLogPpl = Number((result as { observer_logPPL?: number }).observer_logPPL ?? NaN);
    const estimatedB =
      Number.isFinite(approxB)
        ? approxB
        : Number.isFinite(observerLogPpl) && Number.isFinite(baseCross) && baseCross !== 0
          ? observerLogPpl / baseCross
          : NaN;
    liveState.forecastPending = false;
    if (Number.isFinite(estimatedB)) {
      liveState.forecastEstimate = {
        chunkStart,
        docVersion: version,
        b: estimatedB,
      };
      liveEstimateRecoverAttempts.delete(key);
    } else {
      const attempt = liveEstimateRecoverAttempts.get(key) ?? 0;
      output.appendLine(
        `[estimate] non-finite estimate (attempt=${attempt}) approx_b=${String(result.approx_b)} observer_logPPL=${String((result as { observer_logPPL?: number }).observer_logPPL)} baseCross=${String(baseCross)}`,
      );
      if (attempt < 1) {
        liveEstimateRecoverAttempts.set(key, attempt + 1);
        await stopBackend({ shutdownDaemon: true });
        scheduleLiveEstimate(key);
        return;
      }
    }
    updateStatusForEditor(liveEditor);
  } catch (err) {
    if (liveEstimateEpochs.get(key) !== epoch) {
      return;
    }
    const liveState = docStates.get(key);
    const liveEditor = activeEditorForDocKey(key);
    if (liveState) {
      liveState.forecastPending = false;
    }
    const attempt = liveEstimateRecoverAttempts.get(key) ?? 0;
    if (attempt < 1) {
      liveEstimateRecoverAttempts.set(key, attempt + 1);
      try {
        await stopBackend({ shutdownDaemon: true });
      } catch {
        // ignore restart path failures and surface original error below
      }
      scheduleLiveEstimate(key);
      return;
    }
    if (liveEditor) {
      updateStatusForEditor(liveEditor);
    }
    output.appendLine(`[estimate] live estimate failed: ${(err as Error).message}`);
  }
}

// Ensure Enabled Or Notify.
function ensureEnabledOrNotify(): boolean {
  if (isExtensionEnabled()) {
    return true;
  }
  void vscode.window.showInformationMessage('Binoculars is disabled. Run "Binoculars: Enable" to re-enable.');
  return false;
}

// Resolve Render Palette.
function resolveRenderPalette(): RenderPalette {
  // Dark-first palette. Light-mode palette remains intentionally conservative
  // until dedicated light-theme tuning is implemented.
  if (vscode.window.activeColorTheme.kind === vscode.ColorThemeKind.Light) {
    return {
      lowColor: '#b23636',
      highColor: '#1f8f57',
      lowMinorColor: '#000000',
      highMinorColor: '#000000',
      priorLowBg: 'rgba(178, 54, 54, 0.14)',
      priorHighBg: 'rgba(31, 143, 87, 0.14)',
      unscoredColor: '#7d8792',
      unscoredOpacity: '0.72',
    };
  }
  return {
    lowColor: '#ff6b6b',
    highColor: '#3fd28a',
    lowMinorColor: '#dcdcdc',
    highMinorColor: '#dcdcdc',
    priorLowBg: 'rgba(255, 107, 107, 0.16)',
    priorHighBg: 'rgba(63, 210, 138, 0.16)',
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

const lowMinorDecoration = vscode.window.createTextEditorDecorationType({
  color: resolveRenderPalette().lowMinorColor,
});

const highMinorDecoration = vscode.window.createTextEditorDecorationType({
  color: resolveRenderPalette().highMinorColor,
});

const priorLowDecoration = vscode.window.createTextEditorDecorationType({
  backgroundColor: resolveRenderPalette().priorLowBg,
  borderRadius: '2px',
});

const priorHighDecoration = vscode.window.createTextEditorDecorationType({
  backgroundColor: resolveRenderPalette().priorHighBg,
  borderRadius: '2px',
});

const unscoredDecoration = vscode.window.createTextEditorDecorationType({
  color: resolveRenderPalette().unscoredColor,
  opacity: resolveRenderPalette().unscoredOpacity,
});

const editedDecoration = vscode.window.createTextEditorDecorationType({
  backgroundColor: 'rgba(255, 213, 79, 0.22)',
  borderRadius: '2px',
});

// Binoculars Control Item.
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

// Binoculars Controls Provider.
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
    if (!isExtensionEnabled()) {
      return Promise.resolve([
        new BinocularsControlItem('Enable', {
          commandId: 'binoculars.enable',
          icon: new vscode.ThemeIcon('play'),
        }),
      ]);
    }

    const editor = vscode.window.activeTextEditor;
    const key = editor?.document.uri.toString() ?? '';
    const state = key ? docStates.get(key) : undefined;
    const chunkCount = state?.chunks.length ?? 0;
    const nextAvailable = hasNextChunkAvailable(editor);

    const items: BinocularsControlItem[] = [
      new BinocularsControlItem('Analyze Chunk', {
        commandId: 'binoculars.analyze',
        icon: new vscode.ThemeIcon('run'),
        description: 'Ctrl+Alt+B',
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
      new BinocularsControlItem('Toggle Colorization', {
        commandId: 'binoculars.toggleColorization',
        icon: new vscode.ThemeIcon('symbol-color'),
        description: runtimeColorizationEnabled ? 'on' : 'off',
      }),
      new BinocularsControlItem('Disable', {
        commandId: 'binoculars.disable',
        icon: new vscode.ThemeIcon('debug-stop'),
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
    if (nextAvailable) {
      items.splice(
        1,
        0,
        new BinocularsControlItem('Analyze Next Chunk', {
          commandId: 'binoculars.analyzeNext',
          icon: new vscode.ThemeIcon('debug-step-over'),
          description: 'available',
        }),
        new BinocularsControlItem('Analyze All', {
          commandId: 'binoculars.analyzeAll',
          icon: new vscode.ThemeIcon('debug-continue'),
          description: 'may take a while',
        }),
      );
    }
    return Promise.resolve(items);
  }
}

// Gutter Palette.
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

// Activate.
export function activate(context: vscode.ExtensionContext): void {
  // Register extension commands, providers, listeners, and UI artifacts.
  extensionCtx = context;
  output = vscode.window.createOutputChannel('Binoculars');
  statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  statusBar.name = 'Binoculars';
  renderPalette = resolveRenderPalette();
  gutterPalette = new GutterPalette(renderPalette.lowColor, renderPalette.highColor);
  controlsProvider = new BinocularsControlsProvider();

  context.subscriptions.push(
    output,
    statusBar,
    gutterPalette,
    lowDecoration,
    highDecoration,
    lowMinorDecoration,
    highMinorDecoration,
    priorLowDecoration,
    priorHighDecoration,
    unscoredDecoration,
    editedDecoration,
    vscode.commands.registerCommand('binoculars.enable', () => void enableExtension()),
    vscode.commands.registerCommand('binoculars.disable', () => void disableExtension()),
    vscode.commands.registerCommand('binoculars.analyze', () => void runAnalyze()),
    vscode.commands.registerCommand('binoculars.analyzeNext', () => void runAnalyzeNext()),
    vscode.commands.registerCommand('binoculars.analyzeAll', () => void runAnalyzeAll()),
    vscode.commands.registerCommand('binoculars.rewriteSelection', () => void runRewriteSelection()),
    vscode.commands.registerCommand('binoculars.rewriteSelectionOrLine', () => void runRewriteSelectionOrLine()),
    vscode.commands.registerCommand('binoculars.recommendRewrite', () => void runRewriteSelectionOrLine()),
    vscode.commands.registerCommand('binoculars.clearPriors', () => clearPriors()),
    vscode.commands.registerCommand('binoculars.toggleColorization', () => toggleColorization()),
    vscode.commands.registerCommand('binoculars.restartBackend', () => void restartBackend()),
    vscode.window.registerTreeDataProvider('binoculars.controlsView', controlsProvider),
    vscode.languages.registerHoverProvider([{ language: 'markdown' }, { language: 'plaintext' }], {
      provideHover(document, position, token) {
        if (!isExtensionEnabled()) {
          return undefined;
        }
        const docKey = document.uri.toString();
        const docText = document.getText();
        const charOffset = offsetAt(docText, position);
        if (shouldSuppressHoverDuringTyping(document, charOffset)) {
          clearHoverDelayGate(docKey);
          return undefined;
        }
        const lastSeen = hoverSeenSegments.get(docKey);
        if (lastSeen) {
          if (charOffset >= lastSeen.start && charOffset < lastSeen.end) {
            const now = Date.now();
            if (now - lastSeen.lastSeenMs <= HOVER_SAME_SEGMENT_SUPPRESS_MS) {
              return undefined;
            }
          } else {
            hoverSeenSegments.delete(docKey);
          }
        }
        const decision = hoverForPosition(document, position);
        if (!decision) {
          clearHoverDelayGate(docKey);
          return undefined;
        }
        if (token.isCancellationRequested) {
          return undefined;
        }
        if (decision.segmentEnd > decision.segmentStart && HOVER_CONTRIBUTOR_DELAY_MS > 0) {
          const gate = hoverDelayGates.get(docKey);
          const now = Date.now();
          const sameSegmentGate = gate && gate.start === decision.segmentStart && gate.end === decision.segmentEnd;
          if (!sameSegmentGate) {
            clearHoverDelayGate(docKey);
            scheduleHoverGateReveal(
              docKey,
              decision.segmentStart,
              decision.segmentEnd,
              charOffset,
              HOVER_CONTRIBUTOR_DELAY_MS,
            );
            return undefined;
          }
          if (now < gate.readyAtMs) {
            return undefined;
          }
          clearHoverDelayGate(docKey);
        }
        if (decision.segmentEnd > decision.segmentStart) {
          hoverSeenSegments.set(docKey, {
            start: decision.segmentStart,
            end: decision.segmentEnd,
            lastSeenMs: Date.now(),
          });
        }
        return decision.hover;
      },
    }),
    vscode.window.onDidChangeTextEditorSelection((evt) => {
      if (isExtensionEnabled()) {
        void refreshAnalyzeNextContext(evt.textEditor);
        updateStatusForEditor(evt.textEditor);
      }
    }),
    vscode.window.onDidChangeActiveTextEditor((editor) => {
      if (editor) {
        if (!isExtensionEnabled()) {
          clearAllDecorations(editor);
          void refreshAnalyzeNextContext(editor);
          return;
        }
        maybeLoadStateSidecar(editor.document, 'activate');
        maybeRestoreStateFromRecentClosed(editor.document, 'activate');
        const state = docStates.get(editor.document.uri.toString());
        if (state) {
          applyDecorations(editor, state);
        } else {
          clearAllDecorations(editor);
        }
        void refreshAnalyzeNextContext(editor);
        updateStatusForEditor(editor);
      } else {
        void refreshAnalyzeNextContext(undefined);
        if (isExtensionEnabled()) {
          updateStatus('Binoculars Ready. Select Analyze Chunk to begin.');
        }
      }
    }),
    vscode.workspace.onDidOpenTextDocument((doc) => {
      if (!isExtensionEnabled()) {
        return;
      }
      maybeLoadStateSidecar(doc, 'open');
      maybeRestoreStateFromRecentClosed(doc, 'open');
    }),
    vscode.workspace.onDidCloseTextDocument((doc) => {
      const key = doc.uri.toString();
      const state = docStates.get(key);
      if (state && state.chunks.length > 0) {
        rememberClosedStateCandidate(doc, state);
      }
      clearLiveEstimateTimer(key);
      liveEstimateEpochs.delete(key);
      liveEstimateRecoverAttempts.delete(key);
      recentTypingActivity.delete(key);
      hoverSeenSegments.delete(key);
      clearHoverDelayGate(key);
      docStates.delete(key);
      loadedSidecarSignatures.delete(key);
      void refreshAnalyzeNextContext(undefined);
      controlsProvider?.refresh();
    }),
    vscode.workspace.onDidSaveTextDocument((doc) => {
      if (!isExtensionEnabled()) {
        return;
      }
      autoSaveStateSidecar(doc);
    }),
    vscode.workspace.onDidChangeConfiguration((evt) => {
      if (evt.affectsConfiguration('binoculars.enabled')) {
        void applyEnablementMode();
        return;
      }
      if (evt.affectsConfiguration('binoculars') && isExtensionEnabled()) {
        void restartBackend();
      }
    }),
    vscode.workspace.onDidChangeTextDocument((evt) => {
      if (!isExtensionEnabled()) {
        return;
      }
      const key = evt.document.uri.toString();
      if (evt.contentChanges.length > 0) {
        noteTypingActivity(key, evt.contentChanges, evt.document.getText().length);
        hoverSeenSegments.delete(key);
        clearHoverDelayGate(key);
      }
      const state = docStates.get(key);
      if (state && evt.contentChanges.length > 0) {
        state.stale = true;
        state.priorChunkB = undefined;
        state.forecastPending = state.chunks.length > 0;
        shiftChunkStateForContentChanges(state, evt.contentChanges, evt.document.getText().length);
        state.editedRanges = applyContentChangesToEditedRanges(state.editedRanges, evt.contentChanges);
        state.rewriteRanges = applyContentChangesToRewriteRanges(state.rewriteRanges, evt.contentChanges);
        state.priorLowRanges = applyContentChangesToPriorRanges(state.priorLowRanges, evt.contentChanges);
        state.priorHighRanges = applyContentChangesToPriorRanges(state.priorHighRanges, evt.contentChanges);
        rememberClosedStateCandidate(evt.document, state);
        const visibleEditors = vscode.window.visibleTextEditors.filter((editor) => editor.document.uri.toString() === key);
        for (const editor of visibleEditors) {
          if (isTextOverlayColorizationEnabled()) {
            applyEditedDecorations(editor, state);
            applyPriorDecorations(editor, state);
          } else {
            editor.setDecorations(editedDecoration, []);
            editor.setDecorations(priorLowDecoration, []);
            editor.setDecorations(priorHighDecoration, []);
          }
        }
        controlsProvider?.refresh();
      }
      const active = vscode.window.activeTextEditor;
      if (active && active.document.uri.toString() === key) {
        scheduleLiveEstimate(key);
        void refreshAnalyzeNextContext(active);
        updateStatusForEditor(active);
      }
    }),
  );

  void applyEnablementMode();
}

// Deactivate.
export async function deactivate(): Promise<void> {
  // Release backend resources when extension unloads.
  await stopBackend();
}

// Run Analyze.
async function runAnalyze(): Promise<void> {
  // Analyze the active chunk (or first chunk) and refresh all decorations/metrics.
  if (!ensureEnabledOrNotify()) {
    return;
  }
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    return;
  }
  const key = editor.document.uri.toString();
  clearLiveEstimateTimer(key);
  const text = editor.document.getText();
  let previous = docStates.get(key);
  if (!previous) {
    maybeLoadStateSidecar(editor.document, 'activate');
    maybeRestoreStateFromRecentClosed(editor.document, 'activate');
    previous = docStates.get(key);
  }
  const activeChunk = previous && previous.chunks.length > 0 ? getActiveChunk(editor, text, previous) : undefined;
  const analyzeStart = Math.max(0, Math.min(text.length, activeChunk?.charStart ?? 0));
  const startLine = lineNumberFromOffset(text, analyzeStart);
  if (previous && previous.chunks.length > 0) {
    const coverage = computeContiguousCoverage(previous.chunks, text.length);
    output.appendLine(
      `[analyze] active-chunk start line=${startLine} offset=${analyzeStart} chunks=${previous.chunks.length} coverage=${coverage}/${text.length}`,
    );
  } else {
    output.appendLine('[analyze] no prior chunk state found; running initial Analyze from line 1.');
  }
  updateStatus(`Analyzing chunk from line ${startLine}...`);

  try {
    const result = await runWithBusyQuickPick(
      'Binoculars: Analyze Chunk',
      `Analyzing chunk from line ${startLine}...`,
      async () => {
      const client = await ensureBackend();
        if (previous && previous.chunks.length > 0) {
          return await client.analyzeChunk(text, inputLabel(editor.document), analyzeStart, text.length);
        }
        return await client.analyzeDocument(text, inputLabel(editor.document));
      },
    );
    const incoming = toChunkState(result);
    const topKRaw = vscode.workspace.getConfiguration('binoculars').get<number>('topK', 5);
    const topK = Math.max(1, Math.trunc(Number.isFinite(topKRaw) ? topKRaw : 5));
    const priorRanges = previous ? priorContributorRangesForIncoming(previous, incoming, text.length, topK) : undefined;
    let state: DocumentState;
    if (previous && previous.chunks.length > 0) {
      previous.priorLowRanges = mergeEditedRanges([...(previous.priorLowRanges ?? []), ...(priorRanges?.low ?? [])]);
      previous.priorHighRanges = mergeEditedRanges([...(previous.priorHighRanges ?? []), ...(priorRanges?.high ?? [])]);
      previous.priorChunkB = priorChunkScoreForIncoming(previous, incoming);
      mergeChunk(previous, incoming);
      // Recompute contiguous analyzed coverage after in-place chunk replacement.
      previous.nextChunkStart = computeContiguousCoverage(previous.chunks, text.length);
      previous.stale = false;
      previous.forecastEstimate = undefined;
      previous.forecastPending = false;
      previous.editedRanges = [];
      previous.rewriteRanges = [];
      state = previous;
    } else {
      state = {
        chunks: [incoming],
        nextChunkStart: result.next_chunk_start ?? result.chunk.analyzed_char_end,
        stale: false,
        editedRanges: [],
        rewriteRanges: [],
        priorLowRanges: mergeEditedRanges([...(previous?.priorLowRanges ?? []), ...(priorRanges?.low ?? [])]),
        priorHighRanges: mergeEditedRanges([...(previous?.priorHighRanges ?? []), ...(priorRanges?.high ?? [])]),
        priorChunkB: priorChunkScoreForIncoming(previous, incoming),
        forecastEstimate: undefined,
        forecastPending: false,
      };
    }
    docStates.set(key, state);
    rememberClosedStateCandidate(editor.document, state);
    applyDecorations(editor, state);
    autoSaveStateSidecar(editor.document);
    void refreshAnalyzeNextContext(editor);
    updateStatusForEditor(editor);
  } catch (err) {
    showError(`Analyze failed: ${(err as Error).message}`);
  }
}

// Run Analyze Next.
async function runAnalyzeNext(): Promise<void> {
  // Analyze the next contiguous unscored chunk after current analyzed coverage.
  if (!ensureEnabledOrNotify()) {
    return;
  }
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    return;
  }

  const key = editor.document.uri.toString();
  clearLiveEstimateTimer(key);
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

  const startLine = lineNumberFromOffset(fullText, start);
  updateStatus(`Analyzing next chunk from line ${startLine}...`);

  try {
    const result = await runWithBusyQuickPick(
      'Binoculars: Analyze Next Chunk',
      `Analyzing next chunk from line ${startLine}...`,
      async () => {
        const client = await ensureBackend();
        return await client.analyzeNextChunk(fullText, inputLabel(editor.document), start);
      },
    );
    const incoming = toChunkState(result);
    const topKRaw = vscode.workspace.getConfiguration('binoculars').get<number>('topK', 5);
    const topK = Math.max(1, Math.trunc(Number.isFinite(topKRaw) ? topKRaw : 5));
    const priorRanges = priorContributorRangesForIncoming(existing, incoming, fullText.length, topK);
    existing.priorLowRanges = mergeEditedRanges([...(existing.priorLowRanges ?? []), ...priorRanges.low]);
    existing.priorHighRanges = mergeEditedRanges([...(existing.priorHighRanges ?? []), ...priorRanges.high]);
    existing.priorChunkB = priorChunkScoreForIncoming(existing, incoming);
    mergeChunk(existing, incoming);
    existing.nextChunkStart = result.next_chunk_start ?? result.chunk.analyzed_char_end;
    existing.stale = false;
    existing.forecastEstimate = undefined;
    existing.forecastPending = false;
    existing.editedRanges = [];
    existing.rewriteRanges = [];
    rememberClosedStateCandidate(editor.document, existing);
    applyDecorations(editor, existing);
    autoSaveStateSidecar(editor.document);
    void refreshAnalyzeNextContext(editor);
    updateStatusForEditor(editor);
  } catch (err) {
    showError(`Analyze Next failed: ${(err as Error).message}`);
  }
}

// Run Analyze All.
async function runAnalyzeAll(): Promise<void> {
  // Sequentially analyze remaining chunks until full-document coverage is reached.
  if (!ensureEnabledOrNotify()) {
    return;
  }
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    return;
  }

  const key = editor.document.uri.toString();
  clearLiveEstimateTimer(key);
  const docText = editor.document.getText();
  const current = docStates.get(key);
  const start = Math.max(0, current?.nextChunkStart ?? 0);
  if (docText.length <= 0 || start >= docText.length) {
    updateStatus('All text already covered by analyzed chunks.');
    void refreshAnalyzeNextContext(editor);
    return;
  }

  const proceed = await vscode.window.showWarningMessage(
    'Analyze All may take a while on long documents. Are you sure?',
    { modal: true },
    'Yes',
    'No',
  );
  if (proceed !== 'Yes') {
    updateStatus('Analyze All canceled.');
    return;
  }

  const initialVersion = editor.document.version;
  try {
    await runWithBusyQuickPick('Binoculars: Analyze All', 'Analyzing all remaining chunks...', async () => {
      const client = await ensureBackend();
      let fullText = editor.document.getText();
      if (editor.document.version !== initialVersion) {
        throw new Error('Document changed while Analyze All was running. Re-run Analyze All on current text.');
      }

      let state = docStates.get(key);
      if (!state || state.chunks.length === 0) {
        updateStatus('Analyze All in progress: analyzing initial chunk...');
        const result = await client.analyzeDocument(fullText, inputLabel(editor.document));
        const incoming = toChunkState(result);
        const topKRaw = vscode.workspace.getConfiguration('binoculars').get<number>('topK', 5);
        const topK = Math.max(1, Math.trunc(Number.isFinite(topKRaw) ? topKRaw : 5));
        const priorRanges = state ? priorContributorRangesForIncoming(state, incoming, fullText.length, topK) : undefined;
        state = {
          chunks: [incoming],
          nextChunkStart: result.next_chunk_start ?? result.chunk.analyzed_char_end,
          stale: false,
          editedRanges: [],
          rewriteRanges: [],
          priorLowRanges: mergeEditedRanges([...(state?.priorLowRanges ?? []), ...(priorRanges?.low ?? [])]),
          priorHighRanges: mergeEditedRanges([...(state?.priorHighRanges ?? []), ...(priorRanges?.high ?? [])]),
          priorChunkB: priorChunkScoreForIncoming(state, incoming),
          forecastEstimate: undefined,
          forecastPending: false,
        };
        docStates.set(key, state);
        rememberClosedStateCandidate(editor.document, state);
        applyDecorations(editor, state);
      }

      let safetyCounter = 0;
      while (state && state.nextChunkStart < fullText.length) {
        safetyCounter += 1;
        if (safetyCounter > 512) {
          throw new Error('Analyze All aborted due to unexpected excessive chunk iterations.');
        }
        if (editor.document.version !== initialVersion) {
          throw new Error('Document changed while Analyze All was running. Re-run Analyze All on current text.');
        }

        const loopStart = Math.max(0, state.nextChunkStart);
        const startLine = lineNumberFromOffset(fullText, loopStart);
        updateStatus(`Analyze All in progress: analyzing from line ${startLine}...`);
        const result = await client.analyzeNextChunk(fullText, inputLabel(editor.document), loopStart);
        const incoming = toChunkState(result);
        const topKRaw = vscode.workspace.getConfiguration('binoculars').get<number>('topK', 5);
        const topK = Math.max(1, Math.trunc(Number.isFinite(topKRaw) ? topKRaw : 5));
        const priorRanges = priorContributorRangesForIncoming(state, incoming, fullText.length, topK);
        state.priorLowRanges = mergeEditedRanges([...(state.priorLowRanges ?? []), ...priorRanges.low]);
        state.priorHighRanges = mergeEditedRanges([...(state.priorHighRanges ?? []), ...priorRanges.high]);
        state.priorChunkB = priorChunkScoreForIncoming(state, incoming);
        mergeChunk(state, incoming);
        const prevNext = loopStart;
        state.nextChunkStart = result.next_chunk_start ?? result.chunk.analyzed_char_end;
        state.stale = false;
        state.forecastEstimate = undefined;
        state.forecastPending = false;
        state.editedRanges = [];
        state.rewriteRanges = [];
        rememberClosedStateCandidate(editor.document, state);
        applyDecorations(editor, state);
        if (state.nextChunkStart <= prevNext) {
          throw new Error('Analyze All made no forward progress. Try Analyze Chunk again.');
        }
      }
    });
    autoSaveStateSidecar(editor.document);
    void refreshAnalyzeNextContext(editor);
    updateStatusForEditor(editor);
  } catch (err) {
    showError(`Analyze All failed: ${(err as Error).message}`);
  }
}

async function runWithBusyQuickPick<T>(title: string, placeholder: string, action: () => Promise<T>): Promise<T> {
  // Shared light-weight busy UI for bridge operations.
  const picker = vscode.window.createQuickPick<vscode.QuickPickItem>();
  picker.title = title;
  picker.placeholder = placeholder;
  picker.busy = true;
  picker.enabled = false;
  picker.ignoreFocusOut = true;
  picker.show();
  foregroundBusyOperationCount += 1;
  try {
    return await action();
  } finally {
    foregroundBusyOperationCount = Math.max(0, foregroundBusyOperationCount - 1);
    picker.dispose();
  }
}

// Run Rewrite Selection.
async function runRewriteSelection(): Promise<void> {
  if (!ensureEnabledOrNotify()) {
    return;
  }
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

// Run Rewrite Selection Or Line.
async function runRewriteSelectionOrLine(): Promise<void> {
  if (!ensureEnabledOrNotify()) {
    return;
  }
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

// Run Rewrite For Span.
async function runRewriteForSpan(
  editor: vscode.TextEditor,
  stateKey: string,
  state: DocumentState,
  fullText: string,
  start: number,
  end: number,
): Promise<void> {
  // Request rewrite options, let user choose, then apply tracked replacement edit.
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
    const statusMessages = Array.isArray(result.status_messages) ? result.status_messages : [];
    for (const msg of statusMessages) {
      if (typeof msg === 'string' && msg.trim()) {
        output.appendLine(`[rewrite] ${msg}`);
      }
    }
    if (statusMessages.some((msg) => typeof msg === 'string' && msg.toLowerCase().includes('impact scoring unavailable'))) {
      void vscode.window.setStatusBarMessage('Binoculars rewrite options ready (impact estimate unavailable).', 3500);
    }
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
    const editApplied = await editor.edit((eb) => eb.replace(range, selected.text), {
      undoStopBefore: true,
      undoStopAfter: true,
    });
    if (!editApplied) {
      showError('Rewrite could not be applied to the editor.');
      return;
    }
    const next = docStates.get(stateKey);
    if (next) {
      next.stale = true;
      next.priorChunkB = undefined;
      next.editedRanges = mergeEditedRanges([
        ...next.editedRanges,
        {
          start,
          end: start + selected.text.length,
        },
      ]);
      next.rewriteRanges = mergeEditedRanges([
        ...next.rewriteRanges,
        {
          start,
          end: start + selected.text.length,
        },
      ]);
      const exactApproxB = Number(selected.approx_B ?? NaN);
      const rewrittenText = editor.document.getText();
      const rewrittenChunk = getActiveChunk(editor, rewrittenText, next);
      if (rewrittenChunk && Number.isFinite(exactApproxB)) {
        next.forecastEstimate = {
          chunkStart: rewrittenChunk.charStart,
          docVersion: editor.document.version,
          b: exactApproxB,
        };
      }
      next.forecastPending = true;
      applyDecorations(editor, next);
      scheduleLiveEstimate(stateKey);
      updateStatusForEditor(editor);
    }
    void vscode.window.setStatusBarMessage('Binoculars rewrite applied. Analyze to confirm exact B.', 2500);

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

// Pick Rewrite Option With Webview.
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

// Clear Priors.
function clearPriors(): void {
  if (!ensureEnabledOrNotify()) {
    return;
  }
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    return;
  }
  const key = editor.document.uri.toString();
  const state = docStates.get(key);
  if (!state) {
    return;
  }
  state.priorLowRanges = [];
  state.priorHighRanges = [];
  applyDecorations(editor, state);
  updateStatus('Cleared prior highlights.');
}

// Toggle Colorization.
function toggleColorization(): void {
  // Toggle text overlays at runtime without discarding analyzed state.
  if (!ensureEnabledOrNotify()) {
    return;
  }
  runtimeColorizationEnabled = !runtimeColorizationEnabled;
  for (const editor of vscode.window.visibleTextEditors) {
    const state = docStates.get(editor.document.uri.toString());
    if (state) {
      applyDecorations(editor, state);
    } else {
      clearAllDecorations(editor);
    }
  }
  const active = vscode.window.activeTextEditor;
  if (active) {
    updateStatusForEditor(active);
  }
  controlsProvider?.refresh();
  const mode = runtimeColorizationEnabled ? 'ON' : 'OFF';
  void vscode.window.setStatusBarMessage(`Binoculars colorization ${mode}.`, 2500);
}

// Is Text Overlay Colorization Enabled.
function isTextOverlayColorizationEnabled(): boolean {
  if (!isExtensionEnabled()) {
    return false;
  }
  const cfg = vscode.workspace.getConfiguration('binoculars');
  const configured = cfg.get<boolean>('render.colorizeText', true);
  return configured && runtimeColorizationEnabled;
}

// Stop Backend.
async function stopBackend(opts?: { shutdownDaemon?: boolean }): Promise<void> {
  // Tear down active client, with optional shared-daemon shutdown.
  const shutdownDaemon = opts?.shutdownDaemon === true;
  if (shutdownDaemon) {
    try {
      if (backend) {
        await backend.shutdownDaemon();
      } else {
        const probe = new BackendClient(output);
        await probe.shutdownDaemon();
        probe.dispose();
      }
    } catch {
      // ignore
    }
  }
  try {
    await backend?.shutdown();
  } catch {
    // ignore
  }
  backend = undefined;
  backendStarted = false;
}

// Enable Extension.
async function enableExtension(): Promise<void> {
  const cfg = vscode.workspace.getConfiguration('binoculars');
  await cfg.update('enabled', true, vscode.ConfigurationTarget.Global);
}

// Disable Extension.
async function disableExtension(): Promise<void> {
  const cfg = vscode.workspace.getConfiguration('binoculars');
  await cfg.update('enabled', false, vscode.ConfigurationTarget.Global);
}

// Apply Enablement Mode.
async function applyEnablementMode(): Promise<void> {
  // Enforce enabled/disabled mode across backend, contexts, state, and visuals.
  const enabled = isExtensionEnabled();
  await setEnablementContext(enabled);
  await refreshAnalyzeNextContext(undefined);
  if (!enabled) {
    await stopBackend({ shutdownDaemon: true });
    for (const key of liveEstimateTimers.keys()) {
      clearLiveEstimateTimer(key);
    }
    liveEstimateEpochs.clear();
    docStates.clear();
    loadedSidecarSignatures.clear();
    recentClosedStatesByTextHash.clear();
    recentClosedHashByDocKey.clear();
    for (const editor of vscode.window.visibleTextEditors) {
      clearAllDecorations(editor);
    }
    statusBar.hide();
    lastStatusMessage = 'Binoculars disabled.';
    controlsProvider?.refresh();
    return;
  }

  for (const doc of vscode.workspace.textDocuments) {
    maybeLoadStateSidecar(doc, 'activate');
    maybeRestoreStateFromRecentClosed(doc, 'activate');
  }
  for (const editor of vscode.window.visibleTextEditors) {
    const state = docStates.get(editor.document.uri.toString());
    if (state) {
      applyDecorations(editor, state);
    } else {
      clearAllDecorations(editor);
    }
  }
  statusBar.show();
  if (vscode.window.activeTextEditor) {
    updateStatusForEditor(vscode.window.activeTextEditor);
  } else {
    updateStatus('Binoculars Ready. Select Analyze Chunk to begin.');
  }
  controlsProvider?.refresh();
}

// Restart Backend.
async function restartBackend(): Promise<void> {
  // Explicit full backend recycle command used by users during troubleshooting.
  if (!isExtensionEnabled()) {
    await stopBackend({ shutdownDaemon: true });
    return;
  }
  await stopBackend({ shutdownDaemon: true });
  await ensureBackend();
  updateStatus('Binoculars backend restarted.');
}

// Ensure Backend.
async function ensureBackend(): Promise<BackendClient> {
  // Start daemon if needed and (re)apply effective runtime configuration.
  if (!isExtensionEnabled()) {
    throw new Error('Binoculars is disabled. Run "Binoculars: Enable" to re-enable.');
  }
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

  await backend.start(pythonPath, bridgeScriptPath);
  backendStarted = true;

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

// Resolve Rewrite Llm Config Path.
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

// Resolve Path Setting.
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

// Is Markdown Sidecar Eligible.
function isMarkdownSidecarEligible(doc: vscode.TextDocument): boolean {
  if (doc.uri.scheme !== 'file') {
    return false;
  }
  return path.extname(doc.uri.fsPath).toLowerCase() === '.md';
}

// Sidecar State Path For Document.
function sidecarStatePathForDocument(docPath: string): string {
  const abs = path.resolve(docPath);
  const parsed = path.parse(abs);
  return path.join(parsed.dir, `${parsed.name}.json`);
}

// Sha256 Hex.
function sha256Hex(value: string): string {
  return crypto.createHash('sha256').update(value, 'utf8').digest('hex');
}

// Clone Document State.
function cloneDocumentState(state: DocumentState): DocumentState {
  return {
    nextChunkStart: state.nextChunkStart,
    stale: state.stale,
    editedRanges: state.editedRanges.map((r) => ({ start: r.start, end: r.end })),
    rewriteRanges: state.rewriteRanges.map((r) => ({ start: r.start, end: r.end })),
    priorLowRanges: state.priorLowRanges.map((r) => ({ start: r.start, end: r.end })),
    priorHighRanges: state.priorHighRanges.map((r) => ({ start: r.start, end: r.end })),
    priorChunkB: state.priorChunkB,
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

// Prune Recent Closed State Candidates.
function pruneRecentClosedStateCandidates(nowMs = Date.now()): void {
  for (const [textHash, candidate] of recentClosedStatesByTextHash.entries()) {
    if (nowMs - candidate.savedAtMs > RECENT_CLOSED_STATE_TTL_MS) {
      recentClosedStatesByTextHash.delete(textHash);
      const trackedHash = recentClosedHashByDocKey.get(candidate.docKey);
      if (trackedHash === textHash) {
        recentClosedHashByDocKey.delete(candidate.docKey);
      }
    }
  }
}

// Remember Closed State Candidate.
function rememberClosedStateCandidate(doc: vscode.TextDocument, state: DocumentState): void {
  if (!isMarkdownSidecarEligible(doc)) {
    return;
  }
  pruneRecentClosedStateCandidates();
  const docKey = doc.uri.toString();
  const textHash = sha256Hex(doc.getText());
  const prevHash = recentClosedHashByDocKey.get(docKey);
  if (prevHash && prevHash !== textHash) {
    const prevCandidate = recentClosedStatesByTextHash.get(prevHash);
    if (prevCandidate && prevCandidate.docKey === docKey) {
      recentClosedStatesByTextHash.delete(prevHash);
    }
  }
  recentClosedStatesByTextHash.set(textHash, {
    savedAtMs: Date.now(),
    docKey,
    state: cloneDocumentState(state),
  });
  recentClosedHashByDocKey.set(docKey, textHash);
}

// State From Recent Closed Candidate.
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

// State From Open Twin Document.
function stateFromOpenTwinDocument(doc: vscode.TextDocument): DocumentState | undefined {
  if (!isMarkdownSidecarEligible(doc)) {
    return undefined;
  }
  const targetKey = doc.uri.toString();
  const targetHash = sha256Hex(doc.getText());
  for (const candidateDoc of vscode.workspace.textDocuments) {
    const candidateKey = candidateDoc.uri.toString();
    if (candidateKey === targetKey) {
      continue;
    }
    if (!isMarkdownSidecarEligible(candidateDoc)) {
      continue;
    }
    const candidateState = docStates.get(candidateKey);
    if (!candidateState || candidateState.chunks.length === 0) {
      continue;
    }
    if (sha256Hex(candidateDoc.getText()) !== targetHash) {
      continue;
    }
    return cloneDocumentState(candidateState);
  }
  return undefined;
}

// Maybe Restore State From Recent Closed.
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

// Maybe Restore State From Open Twin.
function maybeRestoreStateFromOpenTwin(doc: vscode.TextDocument, reason: 'save'): boolean {
  if (!isMarkdownSidecarEligible(doc)) {
    return false;
  }
  const key = doc.uri.toString();
  if (docStates.has(key)) {
    return false;
  }
  const restored = stateFromOpenTwinDocument(doc);
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
  output.appendLine(`[state] restored analysis from open twin doc (${reason}): ${doc.uri.fsPath}`);
  return true;
}

// As Record.
function asRecord(value: unknown): Record<string, unknown> | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return undefined;
  }
  return value as Record<string, unknown>;
}

// As Finite Number.
function asFiniteNumber(value: unknown): number | undefined {
  const n = Number(value);
  return Number.isFinite(n) ? n : undefined;
}

// As Clamped Int.
function asClampedInt(value: unknown, min: number, max: number): number | undefined {
  const n = asFiniteNumber(value);
  if (typeof n !== 'number') {
    return undefined;
  }
  return Math.max(min, Math.min(max, Math.trunc(n)));
}

// Parse Chunk Metrics.
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

// Parse Paragraph Row.
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

// Parse Chunk State.
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

// Compute Contiguous Coverage.
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

// Document State From Persisted.
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
  const rewriteRangesRaw = Array.isArray(rawState.rewrite_ranges) ? rawState.rewrite_ranges : [];
  const rewriteRanges = rewriteRangesRaw
    .map((entry) => asRecord(entry))
    .filter((entry): entry is Record<string, unknown> => typeof entry !== 'undefined')
    .map((entry) => ({
      start: asClampedInt(entry.start, 0, textLen),
      end: asClampedInt(entry.end, 0, textLen),
    }))
    .filter((entry): entry is { start: number; end: number } => typeof entry.start === 'number' && typeof entry.end === 'number')
    .filter((entry) => entry.end > entry.start);
  const priorLowRangesRaw = Array.isArray(rawState.prior_low_ranges) ? rawState.prior_low_ranges : [];
  const priorLowRanges = priorLowRangesRaw
    .map((entry) => asRecord(entry))
    .filter((entry): entry is Record<string, unknown> => typeof entry !== 'undefined')
    .map((entry) => ({
      start: asClampedInt(entry.start, 0, textLen),
      end: asClampedInt(entry.end, 0, textLen),
    }))
    .filter((entry): entry is { start: number; end: number } => typeof entry.start === 'number' && typeof entry.end === 'number')
    .filter((entry) => entry.end > entry.start);
  const priorHighRangesRaw = Array.isArray(rawState.prior_high_ranges) ? rawState.prior_high_ranges : [];
  const priorHighRanges = priorHighRangesRaw
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
    rewriteRanges,
    priorLowRanges,
    priorHighRanges,
    // Prior B is an in-session comparison value and should not be restored from persisted state.
    priorChunkB: undefined,
  };
}

// Serialize Chunk For Persisted State.
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

// Build Persisted Sidecar Payload.
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
    rewrite_ranges: (state?.rewriteRanges ?? []).map((r) => ({
      start: r.start,
      end: r.end,
    })),
    prior_low_ranges: (state?.priorLowRanges ?? []).map((r) => ({
      start: r.start,
      end: r.end,
    })),
    prior_high_ranges: (state?.priorHighRanges ?? []).map((r) => ({
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

// Maybe Load State Sidecar.
function maybeLoadStateSidecar(doc: vscode.TextDocument, reason: 'open' | 'activate'): void {
  // Load sidecar state only when it matches document content hash.
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
  const resolvedState = persistedState ?? stateFromMatchingSidecarOnDisk(doc, textSnapshot, textHash);
  if (!resolvedState) {
    return;
  }

  docStates.set(key, resolvedState);
  const visibleEditors = vscode.window.visibleTextEditors.filter((editor) => editor.document.uri.toString() === key);
  for (const editor of visibleEditors) {
    applyDecorations(editor, resolvedState);
  }
  if (vscode.window.activeTextEditor?.document.uri.toString() === key) {
    updateStatusForEditor(vscode.window.activeTextEditor);
  }
  controlsProvider?.refresh();
  if (persistedState) {
    output.appendLine(`[state] loaded sidecar (${reason}): ${sidecarPath}`);
  } else {
    output.appendLine(`[state] restored from matching sidecar (${reason}): ${doc.uri.fsPath}`);
  }
}

// Auto Save State Sidecar.
function autoSaveStateSidecar(doc: vscode.TextDocument): void {
  // Persist current in-memory state to sidecar after document saves.
  if (!isMarkdownSidecarEligible(doc)) {
    return;
  }
  const key = doc.uri.toString();
  const textSnapshot = doc.getText();
  if (!docStates.has(key)) {
    if (!maybeRestoreStateFromRecentClosed(doc, 'save')) {
      maybeRestoreStateFromOpenTwin(doc, 'save');
    }
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

// State From Matching Sidecar On Disk.
function stateFromMatchingSidecarOnDisk(
  doc: vscode.TextDocument,
  textSnapshot: string,
  textHash: string,
): DocumentState | undefined {
  if (!isMarkdownSidecarEligible(doc)) {
    return undefined;
  }
  const docDir = path.dirname(doc.uri.fsPath);
  const targetSidecarPath = sidecarStatePathForDocument(doc.uri.fsPath);
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(docDir, { withFileTypes: true });
  } catch {
    return undefined;
  }

  let bestState: DocumentState | undefined;
  let bestSavedAtMs = -1;
  for (const ent of entries) {
    if (!ent.isFile() || path.extname(ent.name).toLowerCase() !== '.json') {
      continue;
    }
    const sidecarPath = path.join(docDir, ent.name);
    if (sidecarPath === targetSidecarPath) {
      continue;
    }
    let raw: string;
    try {
      raw = fs.readFileSync(sidecarPath, 'utf8');
    } catch {
      continue;
    }
    let payloadUnknown: unknown;
    try {
      payloadUnknown = JSON.parse(raw);
    } catch {
      continue;
    }
    const payload = asRecord(payloadUnknown);
    if (!payload || payload.binoculars_gui_state !== true) {
      continue;
    }
    if (String(payload.text_sha256 ?? '').trim() !== textHash) {
      continue;
    }
    const rawState = asRecord(payload.state);
    if (!rawState) {
      continue;
    }
    const candidate = documentStateFromPersisted(rawState, textSnapshot.length);
    if (!candidate || candidate.chunks.length === 0) {
      continue;
    }
    const savedAtMs = Date.parse(String(payload.saved_at ?? ''));
    const score = Number.isFinite(savedAtMs) ? savedAtMs : -1;
    if (!bestState || score >= bestSavedAtMs) {
      bestState = candidate;
      bestSavedAtMs = score;
    }
  }
  return bestState;
}

// To Chunk State.
function toChunkState(result: AnalyzeResult): ChunkState {
  return {
    charStart: result.chunk.char_start,
    charEnd: result.chunk.char_end,
    analyzedCharEnd: result.chunk.analyzed_char_end,
    metrics: result.chunk.metrics,
    rows: result.paragraph_profile?.rows ?? [],
  };
}

// Merge Chunk.
function mergeChunk(state: DocumentState, incoming: ChunkState): void {
  const kept = state.chunks.filter((c) => c.analyzedCharEnd <= incoming.charStart || c.charStart >= incoming.analyzedCharEnd);
  kept.push(incoming);
  kept.sort((a, b) => a.charStart - b.charStart);
  state.chunks = kept;
}

// Prior Chunk Score For Incoming.
function priorChunkScoreForIncoming(state: DocumentState | undefined, incoming: ChunkState): number | undefined {
  if (!state || !state.chunks.length) {
    return undefined;
  }
  let bestOverlap = 0;
  let bestScore: number | undefined;
  for (const chunk of state.chunks) {
    if (!chunk.metrics) {
      continue;
    }
    const start = Math.max(chunk.charStart, incoming.charStart);
    const end = Math.min(chunk.analyzedCharEnd, incoming.analyzedCharEnd);
    const overlap = Math.max(0, end - start);
    if (overlap <= 0 || overlap < bestOverlap) {
      continue;
    }
    bestOverlap = overlap;
    bestScore = chunk.metrics.binoculars_score;
  }
  return bestScore;
}

// Prior Contributor Ranges For Incoming.
function priorContributorRangesForIncoming(
  state: DocumentState,
  incoming: ChunkState,
  textLen: number,
  topK: number,
): { low: EditedRange[]; high: EditedRange[] } {
  const low: EditedRange[] = [];
  const high: EditedRange[] = [];
  const majorRows = computeMajorContributorRows(state, topK);
  for (const chunk of state.chunks) {
    const overlapStart = Math.max(chunk.charStart, incoming.charStart);
    const overlapEnd = Math.min(chunk.analyzedCharEnd, incoming.analyzedCharEnd);
    if (overlapEnd <= overlapStart) {
      continue;
    }
    for (const row of chunk.rows) {
      if (!majorRows.has(row)) {
        continue;
      }
      const rowStart = Math.max(0, Math.min(textLen, Number(row.char_start ?? 0)));
      const rowEnd = Math.max(rowStart, Math.min(textLen, Number(row.char_end ?? rowStart)));
      if (rowEnd <= rowStart) {
        continue;
      }
      const start = Math.max(overlapStart, rowStart);
      const end = Math.min(overlapEnd, rowEnd);
      if (end <= start) {
        continue;
      }
      const delta = Number(row.delta_doc_logPPL_if_removed ?? 0.0);
      if (delta >= 0) {
        low.push({ start, end });
      } else {
        high.push({ start, end });
      }
    }
  }
  return {
    low: mergeEditedRanges(low),
    high: mergeEditedRanges(high),
  };
}

// Merge Edited Ranges.
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

// Normalized Ranges Without Touch Merge.
function normalizedRangesWithoutTouchMerge(ranges: EditedRange[]): EditedRange[] {
  return [...ranges]
    .filter((r) => Number.isFinite(r.start) && Number.isFinite(r.end) && r.end > r.start)
    .sort((a, b) => (a.start - b.start) || (a.end - b.end));
}

// Smallest Containing Range.
function smallestContainingRange(ranges: EditedRange[], offset: number): EditedRange | undefined {
  const candidateOffset = Math.max(0, Math.trunc(offset));
  let best: EditedRange | undefined;
  let bestSpan = Number.MAX_SAFE_INTEGER;
  for (const r of normalizedRangesWithoutTouchMerge(ranges)) {
    if (candidateOffset < r.start || candidateOffset >= r.end) {
      continue;
    }
    const span = r.end - r.start;
    if (!best || span < bestSpan) {
      best = r;
      bestSpan = span;
    }
  }
  return best;
}

// Has Range Overlap.
function hasRangeOverlap(ranges: EditedRange[], start: number, end: number): boolean {
  const boundedStart = Math.max(0, Math.trunc(start));
  const boundedEnd = Math.max(boundedStart, Math.trunc(end));
  if (boundedEnd <= boundedStart) {
    return false;
  }
  for (const r of normalizedRangesWithoutTouchMerge(ranges)) {
    const ovStart = Math.max(boundedStart, r.start);
    const ovEnd = Math.min(boundedEnd, r.end);
    if (ovEnd > ovStart) {
      return true;
    }
  }
  return false;
}

// Shift Edited Ranges For Splice.
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

// Remap Offset For Splice.
function remapOffsetForSplice(offset: number, start: number, end: number, insertedLen: number): number {
  const safeOffset = Math.max(0, offset);
  const safeStart = Math.max(0, start);
  const safeEnd = Math.max(safeStart, end);
  const delta = insertedLen - (safeEnd - safeStart);
  if (safeOffset < safeStart) {
    return safeOffset;
  }
  if (safeOffset >= safeEnd) {
    return safeOffset + delta;
  }
  return safeStart;
}

// Remap Span For Splice.
function remapSpanForSplice(
  spanStart: number,
  spanEnd: number,
  spliceStart: number,
  spliceEnd: number,
  insertedLen: number,
): { start: number; end: number } | undefined {
  const start = remapOffsetForSplice(spanStart, spliceStart, spliceEnd, insertedLen);
  const end = remapOffsetForSplice(spanEnd, spliceStart, spliceEnd, insertedLen);
  if (end <= start) {
    return undefined;
  }
  return { start, end };
}

// Shift Chunk State For Content Changes.
function shiftChunkStateForContentChanges(
  state: DocumentState,
  changes: readonly vscode.TextDocumentContentChangeEvent[],
  textLength: number,
): void {
  if (!changes || changes.length === 0 || state.chunks.length === 0) {
    state.nextChunkStart = Math.max(0, Math.min(textLength, state.nextChunkStart));
    return;
  }
  const sorted = [...changes].sort((a, b) => a.rangeOffset - b.rangeOffset);
  let chunks: ChunkState[] = state.chunks.map((chunk) => ({
    charStart: chunk.charStart,
    charEnd: chunk.charEnd,
    analyzedCharEnd: chunk.analyzedCharEnd,
    metrics: chunk.metrics,
    rows: chunk.rows.map((row) => ({ ...row })),
  }));
  let nextChunkStart = state.nextChunkStart;
  let delta = 0;

  for (const ch of sorted) {
    const originalStart = Math.max(0, ch.rangeOffset);
    const originalEnd = Math.max(originalStart, ch.rangeOffset + Math.max(0, ch.rangeLength));
    const spliceStart = Math.max(0, originalStart + delta);
    const spliceEnd = Math.max(spliceStart, originalEnd + delta);
    const insertedLen = ch.text.length;

    nextChunkStart = remapOffsetForSplice(nextChunkStart, spliceStart, spliceEnd, insertedLen);

    const nextChunks: ChunkState[] = [];
    for (const chunk of chunks) {
      const remappedChunk = remapSpanForSplice(chunk.charStart, chunk.charEnd, spliceStart, spliceEnd, insertedLen);
      const remappedAnalyzed = remapSpanForSplice(
        chunk.charStart,
        chunk.analyzedCharEnd,
        spliceStart,
        spliceEnd,
        insertedLen,
      );
      if (!remappedChunk || !remappedAnalyzed) {
        continue;
      }
      const charStart = remappedChunk.start;
      const analyzedCharEnd = Math.max(charStart, remappedAnalyzed.end);
      const charEnd = Math.max(remappedChunk.end, analyzedCharEnd);
      if (charEnd <= charStart) {
        continue;
      }

      const rows = chunk.rows
        .map((row) => {
          const remapped = remapSpanForSplice(row.char_start, row.char_end, spliceStart, spliceEnd, insertedLen);
          if (!remapped) {
            return undefined;
          }
          const clippedStart = Math.max(charStart, remapped.start);
          const clippedEnd = Math.min(charEnd, remapped.end);
          if (clippedEnd <= clippedStart) {
            return undefined;
          }
          return {
            ...row,
            char_start: clippedStart,
            char_end: clippedEnd,
          };
        })
        .filter((row): row is ParagraphRow => typeof row !== 'undefined');

      nextChunks.push({
        ...chunk,
        charStart,
        charEnd,
        analyzedCharEnd,
        rows,
      });
    }
    chunks = nextChunks;
    delta += insertedLen - (originalEnd - originalStart);
  }

  chunks.sort((a, b) => a.charStart - b.charStart);
  state.chunks = chunks;
  state.nextChunkStart = Math.max(0, Math.min(textLength, nextChunkStart));
}

// Apply Content Changes To Ranges.
function applyContentChangesToRanges(
  prevRanges: EditedRange[],
  changes: readonly vscode.TextDocumentContentChangeEvent[],
  includeInsertedText: boolean,
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
    if (includeInsertedText && insertedLen > 0) {
      ranges.push({ start, end: start + insertedLen });
      ranges = mergeEditedRanges(ranges);
    }
    delta += insertedLen - (originalEnd - originalStart);
  }
  return ranges;
}

// Apply Content Changes To Edited Ranges.
function applyContentChangesToEditedRanges(
  prevRanges: EditedRange[],
  changes: readonly vscode.TextDocumentContentChangeEvent[],
): EditedRange[] {
  return applyContentChangesToRanges(prevRanges, changes, true);
}

// Apply Content Changes To Rewrite Ranges.
function applyContentChangesToRewriteRanges(
  prevRanges: EditedRange[],
  changes: readonly vscode.TextDocumentContentChangeEvent[],
): EditedRange[] {
  return applyContentChangesToRanges(prevRanges, changes, false);
}

// Apply Content Changes To Prior Ranges.
function applyContentChangesToPriorRanges(
  prevRanges: EditedRange[],
  changes: readonly vscode.TextDocumentContentChangeEvent[],
): EditedRange[] {
  return applyContentChangesToRanges(prevRanges, changes, false);
}

// Edited Ranges To Decoration Ranges.
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

// Apply Edited Decorations.
function applyEditedDecorations(editor: vscode.TextEditor, state: DocumentState): void {
  const docText = editor.document.getText();
  const ranges = editedRangesToDecorationRanges(docText, state.editedRanges);
  editor.setDecorations(editedDecoration, ranges);
}

// Apply Prior Decorations.
function applyPriorDecorations(editor: vscode.TextEditor, state: DocumentState): void {
  const docText = editor.document.getText();
  const low = editedRangesToDecorationRanges(docText, state.priorLowRanges);
  const high = editedRangesToDecorationRanges(docText, state.priorHighRanges);
  editor.setDecorations(priorLowDecoration, low);
  editor.setDecorations(priorHighDecoration, high);
}

// Apply Decorations.
function applyDecorations(editor: vscode.TextEditor, state: DocumentState): void {
  // Render current+prior contributor visuals and optional contribution bars.
  clearAllDecorations(editor);
  if (!isExtensionEnabled()) {
    return;
  }

  const cfg = vscode.workspace.getConfiguration('binoculars');
  const enableColorize = isTextOverlayColorizationEnabled();
  const enableBars = cfg.get<boolean>('render.contributionBars', true);
  const topKRaw = cfg.get<number>('topK', 5);
  const topK = Math.max(1, Math.trunc(Number.isFinite(topKRaw) ? topKRaw : 5));

  const docText = editor.document.getText();
  const lowRanges: vscode.Range[] = [];
  const highRanges: vscode.Range[] = [];
  const lowMinorRanges: vscode.Range[] = [];
  const highMinorRanges: vscode.Range[] = [];
  const rowEntries: Array<{
    range: vscode.Range;
    delta: number;
  }> = [];

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
      rowEntries.push({ range, delta });

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

  const byDelta = rowEntries.map((entry, idx) => ({ idx, delta: entry.delta }));
  const lowIdx = new Set(
    byDelta
      .filter((x) => Number.isFinite(x.delta) && x.delta >= 0)
      .sort((a, b) => b.delta - a.delta)
      .slice(0, topK)
      .map((x) => x.idx),
  );
  const highIdx = new Set(
    byDelta
      .filter((x) => Number.isFinite(x.delta) && x.delta < 0)
      .sort((a, b) => a.delta - b.delta)
      .slice(0, topK)
      .map((x) => x.idx),
  );
  for (let i = 0; i < rowEntries.length; i += 1) {
    if (lowIdx.has(i)) {
      lowRanges.push(rowEntries[i].range);
    } else if (highIdx.has(i)) {
      highRanges.push(rowEntries[i].range);
    } else if (rowEntries[i].delta >= 0) {
      lowMinorRanges.push(rowEntries[i].range);
    } else {
      highMinorRanges.push(rowEntries[i].range);
    }
  }

  if (enableColorize) {
    editor.setDecorations(lowDecoration, lowRanges);
    editor.setDecorations(highDecoration, highRanges);
    editor.setDecorations(lowMinorDecoration, lowMinorRanges);
    editor.setDecorations(highMinorDecoration, highMinorRanges);
  }

  if (enableColorize) {
    const scoredIntervals = mergeIntervals(
      state.chunks
        .map((c) => ({ start: c.charStart, end: c.analyzedCharEnd }))
        .filter((x) => x.end > x.start),
    );
    const unscored = invertIntervals(scoredIntervals, docText.length);
    const unscoredRanges = unscored.map((iv) => new vscode.Range(positionAt(docText, iv.start), positionAt(docText, iv.end)));
    editor.setDecorations(unscoredDecoration, unscoredRanges);
    applyPriorDecorations(editor, state);
  }

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
  if (enableColorize) {
    applyEditedDecorations(editor, state);
  }
}

// Clear All Decorations.
function clearAllDecorations(editor: vscode.TextEditor): void {
  editor.setDecorations(lowDecoration, []);
  editor.setDecorations(highDecoration, []);
  editor.setDecorations(lowMinorDecoration, []);
  editor.setDecorations(highMinorDecoration, []);
  editor.setDecorations(priorLowDecoration, []);
  editor.setDecorations(priorHighDecoration, []);
  editor.setDecorations(unscoredDecoration, []);
  editor.setDecorations(editedDecoration, []);
  for (let i = 0; i < 21; i += 1) {
    editor.setDecorations(gutterPalette.bucketType('red', i), []);
    editor.setDecorations(gutterPalette.bucketType('green', i), []);
  }
}

// Get Active Chunk.
function getActiveChunk(editor: vscode.TextEditor, text: string, state: DocumentState): ChunkState | undefined {
  // Resolve active chunk using selection/cursor/visible-window priority.
  if (state.chunks.length === 0) {
    return undefined;
  }

  const offset = offsetAt(text, editor.selection.active);
  const containing = state.chunks.find((c) => offset >= c.charStart && offset < c.analyzedCharEnd);
  if (containing) {
    return containing;
  }

  // Choose nearest analyzed interval boundary when cursor is in an unscored gap.
  const distanceToChunk = (chunk: ChunkState): number => {
    if (offset < chunk.charStart) {
      return chunk.charStart - offset;
    }
    if (offset > chunk.analyzedCharEnd) {
      return offset - chunk.analyzedCharEnd;
    }
    return 0;
  };

  const sorted = [...state.chunks].sort((a, b) => {
    const da = distanceToChunk(a);
    const db = distanceToChunk(b);
    if (da !== db) {
      return da - db;
    }
    // Stable tie-break to keep behavior deterministic.
    return a.charStart - b.charStart;
  });
  return sorted[0];
}

// Resolve Rewrite Span.
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

// Resolve Selection Or Line Span.
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

// Find Best Row Match.
function findBestRowMatch(state: DocumentState, charOffset: number): { chunk: ChunkState; row: ParagraphRow } | undefined {
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
  return bestMatch;
}

// Compute Major Contributor Rows.
function computeMajorContributorRows(state: DocumentState, topK: number): Set<ParagraphRow> {
  const entries: Array<{ row: ParagraphRow; delta: number }> = [];
  for (const chunk of state.chunks) {
    for (const row of chunk.rows) {
      entries.push({
        row,
        delta: Number(row.delta_doc_logPPL_if_removed ?? 0.0),
      });
    }
  }
  if (entries.length === 0) {
    return new Set<ParagraphRow>();
  }
  const lows = entries
    .filter((x) => Number.isFinite(x.delta) && x.delta >= 0)
    .sort((a, b) => b.delta - a.delta)
    .slice(0, topK)
    .map((x) => x.row);
  const highs = entries
    .filter((x) => Number.isFinite(x.delta) && x.delta < 0)
    .sort((a, b) => a.delta - b.delta)
    .slice(0, topK)
    .map((x) => x.row);
  return new Set([...lows, ...highs]);
}

// Hover For Position.
function hoverForPosition(document: vscode.TextDocument, position: vscode.Position): HoverDecision | undefined {
  // Return contextual hover diagnostics for rewritten/edited/contributor spans.
  const key = document.uri.toString();
  const state = docStates.get(key);
  if (!state || state.chunks.length === 0) {
    return undefined;
  }
  const text = document.getText();
  const charOffset = offsetAt(text, position);
  const topKRaw = vscode.workspace.getConfiguration('binoculars').get<number>('topK', 5);
  const topK = Math.max(1, Math.trunc(Number.isFinite(topKRaw) ? topKRaw : 5));
  const bestMatch = findBestRowMatch(state, charOffset);
  const majorRows = computeMajorContributorRows(state, topK);
  const isMinorContributor = bestMatch ? !majorRows.has(bestMatch.row) : false;
  const rewriteRange = smallestContainingRange(state.rewriteRanges, charOffset);
  if (rewriteRange) {
    const range = new vscode.Range(positionAt(text, rewriteRange.start), positionAt(text, rewriteRange.end));
    const md = new vscode.MarkdownString('Segment rewritten - Select Analyze to determine new score.');
    md.isTrusted = false;
    return {
      hover: new vscode.Hover(md, range),
      isMinorContributor,
      segmentStart: rewriteRange.start,
      segmentEnd: rewriteRange.end,
    };
  }
  const manuallyEditedRange = smallestContainingRange(state.editedRanges, charOffset);
  if (!bestMatch) {
    if (manuallyEditedRange) {
      const range = new vscode.Range(positionAt(text, manuallyEditedRange.start), positionAt(text, manuallyEditedRange.end));
      const md = new vscode.MarkdownString(
        '*Note: some text has been manually changed which may impact score - select Analyze again to obtain accurate statistics.*',
      );
      md.isTrusted = false;
      return {
        hover: new vscode.Hover(md, range),
        isMinorContributor: false,
        segmentStart: manuallyEditedRange.start,
        segmentEnd: manuallyEditedRange.end,
      };
    }
    return undefined;
  }

  const delta = Number(bestMatch.row.delta_doc_logPPL_if_removed ?? NaN);
  const rowStart = Math.max(0, Number(bestMatch.row.char_start ?? 0));
  const rowEnd = Math.max(rowStart, Number(bestMatch.row.char_end ?? rowStart));
  const rowHasManualEdits = hasRangeOverlap(state.editedRanges, rowStart, rowEnd);
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
  if (manuallyEditedRange || rowHasManualEdits) {
    lines.push(
      '',
      '*Note: some text has been manually changed which may impact score - select Analyze again to obtain accurate statistics.*',
    );
  }
  const md = new vscode.MarkdownString(lines.join('  \n'));
  md.isTrusted = false;
  const lineStart = Math.max(0, text.lastIndexOf('\n', Math.max(0, charOffset - 1)) + 1);
  const rawLineEnd = text.indexOf('\n', Math.max(0, charOffset));
  const lineEnd = rawLineEnd >= 0 ? rawLineEnd : text.length;
  const hoverStart = Math.max(rowStart, lineStart);
  const hoverEnd = Math.max(hoverStart, Math.min(rowEnd, lineEnd));
  const range = new vscode.Range(
    positionAt(text, hoverStart),
    positionAt(text, hoverEnd > hoverStart ? hoverEnd : rowEnd),
  );
  return {
    hover: new vscode.Hover(md, range),
    isMinorContributor,
    segmentStart: rowStart,
    segmentEnd: rowEnd,
  };
}

// Update Status For Editor.
function updateStatusForEditor(editor: vscode.TextEditor): void {
  // Build status line from resolved active chunk, stale flags, and estimates.
  if (!isExtensionEnabled()) {
    void refreshAnalyzeNextContext(editor);
    return;
  }
  void refreshAnalyzeNextContext(editor);
  const key = editor.document.uri.toString();
  const state = docStates.get(key);
  if (!state || state.chunks.length === 0) {
    updateStatus('Binoculars Ready. Select Analyze Chunk to begin.');
    return;
  }
  const text = editor.document.getText();
  const orderedChunks = [...state.chunks].sort((a, b) => a.charStart - b.charStart);
  const cursorOffset = offsetAt(text, editor.selection.active);
  const covered = computeContiguousCoverage(orderedChunks, text.length);
  const hasMore = covered < text.length;
  const scoredIntervals = mergeIntervals(
    orderedChunks
      .map((chunk) => ({
        start: Math.max(0, Math.min(text.length, chunk.charStart)),
        end: Math.max(0, Math.min(text.length, chunk.analyzedCharEnd)),
      }))
      .filter((interval) => interval.end > interval.start),
  );
  const inAnalyzedCoverage = scoredIntervals.some((interval) => cursorOffset >= interval.start && cursorOffset < interval.end);
  const cursorChunk = inAnalyzedCoverage
    ? orderedChunks.find((chunk) => cursorOffset >= chunk.charStart && cursorOffset < chunk.analyzedCharEnd)
    : undefined;

  if (!inAnalyzedCoverage || !cursorChunk) {
    const unscoredIntervals = invertIntervals(scoredIntervals, text.length);
    const unscoredAtCursor =
      unscoredIntervals.find((interval) => cursorOffset >= interval.start && cursorOffset < interval.end) ??
      (cursorOffset >= text.length ? unscoredIntervals[unscoredIntervals.length - 1] : undefined) ??
      unscoredIntervals.find((interval) => interval.start > cursorOffset) ??
      unscoredIntervals[0];
    const startLine = lineNumberFromOffset(text, unscoredAtCursor?.start ?? cursorOffset);
    if (hasMore) {
      updateStatus(`Text starting from line ${startLine} has not been analyzed. Select Analyze Next Chunk to continue.`);
    } else {
      updateStatus(`Text starting from line ${startLine} has not been analyzed. Select Analyze Chunk to refresh coverage.`);
    }
    return;
  }

  if (!cursorChunk.metrics) {
    updateStatus('Binoculars analyzed chunk available.');
    return;
  }
  const exactB = Number(cursorChunk.metrics.binoculars_score);
  const staleSuffix = state.stale ? ' | stale (run Analyze; estimate may differ from exact)' : '';
  const estimateState = (() => {
    const estimate = state.forecastEstimate;
    const hasMatchingEstimate =
      !!estimate &&
      estimate.chunkStart === cursorChunk.charStart &&
      estimate.docVersion === editor.document.version &&
      Number.isFinite(estimate.b);
    if (!hasMatchingEstimate) {
      return {
        text: '',
        value: undefined as number | undefined,
      };
    }
    return {
      text: state.forecastPending
        ? `${formatSignedMetric(estimate.b)} (updating...)`
        : `${formatSignedMetric(estimate.b)}`,
      value: Number(estimate.b),
    };
  })();
  const estimateDiffValue =
    typeof estimateState.value === 'number' && Number.isFinite(estimateState.value) && Number.isFinite(exactB)
      ? estimateState.value - exactB
      : undefined;
  const hasNonZeroNumericEstimate =
    typeof estimateDiffValue === 'number' && Number.isFinite(estimateDiffValue)
      ? estimateDiffValue !== 0
      : false;
  const estimateSuffix = (() => {
    if (hasNonZeroNumericEstimate) {
      return ` | B Est.: ${estimateState.text} | B Diff. Est.: ${formatSignedMetric(estimateDiffValue ?? 0)}`;
    }
    if (state.stale && state.forecastPending) {
      return ' | B Est.: estimating... | B Diff. Est.: estimating...';
    }
    if (state.stale) {
      return ' | B Est.: n/a | B Diff. Est.: n/a';
    }
    return '';
  })();
  const priorBSuffix =
    typeof state.priorChunkB === 'number' && Number.isFinite(state.priorChunkB)
      ? ` | Prior B ${formatSignedMetric(state.priorChunkB)}`
      : '';
  const moreSuffix = hasMore ? ` | Analyze Next available (line ${lineNumberFromOffset(text, covered)})` : '';
  const metricCore = `B: ${formatSignedMetric(exactB)}${priorBSuffix}${estimateSuffix} | Obs: ${formatStatusMetric(cursorChunk.metrics.observer_logPPL)} | Cross: ${formatStatusMetric(cursorChunk.metrics.cross_logXPPL)}${staleSuffix}${moreSuffix}`;
  if (orderedChunks.length > 1) {
    const chunkIndex = Math.max(1, orderedChunks.indexOf(cursorChunk) + 1);
    updateStatus(`Binoculars (chunk ${chunkIndex}): ${metricCore}`);
    return;
  }
  updateStatus(metricCore);
}

// Update Status.
function updateStatus(message: string): void {
  lastStatusMessage = message;
  if (!isExtensionEnabled()) {
    controlsProvider?.refresh();
    return;
  }
  const text = message.startsWith('Binoculars ') ? message : `Binoculars: ${message}`;
  statusBar.text = `$(telescope) ${text}`;
  controlsProvider?.refresh();
}

// Show Error.
function showError(message: string): void {
  void vscode.window.showErrorMessage(message);
  updateStatus(message);
}

// Input Label.
function inputLabel(doc: vscode.TextDocument): string {
  return doc.uri.scheme === 'file' ? doc.uri.fsPath : '<unsaved>';
}

// Format Status Metric.
function formatStatusMetric(value: number): string {
  if (!Number.isFinite(value)) {
    return 'n/a';
  }
  return value.toFixed(DISPLAY_DECIMALS);
}

// Format Signed Metric.
function formatSignedMetric(value: number): string {
  if (!Number.isFinite(value)) {
    return 'n/a';
  }
  return `${value >= 0 ? '+' : ''}${value.toFixed(DISPLAY_DECIMALS)}`;
}

// Format Approx.
function formatApprox(approxB?: number, deltaB?: number): string {
  if (typeof approxB !== 'number' || Number.isNaN(approxB)) {
    return 'rewrite option';
  }
  const d = typeof deltaB === 'number' && Number.isFinite(deltaB) ? ` (${formatSignedMetric(deltaB)})` : '';
  return `approx B ${approxB.toFixed(DISPLAY_DECIMALS)}${d}`;
}

// Line Number From Offset.
function lineNumberFromOffset(text: string, offset: number): number {
  const clamped = Math.max(0, Math.min(text.length, Math.trunc(offset)));
  return positionAt(text, clamped).line + 1;
}

// Escape Html.
function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// Create Nonce.
function createNonce(): string {
  return crypto.randomBytes(12).toString('base64');
}

// Render Rewrite Options Html.
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

// Position At.
function positionAt(text: string, offset: number): vscode.Position {
  const bounded = Math.max(0, Math.min(text.length, offset));
  const lines = text.slice(0, bounded).split('\n');
  const line = Math.max(0, lines.length - 1);
  const ch = lines[lines.length - 1]?.length ?? 0;
  return new vscode.Position(line, ch);
}

// Offset At.
function offsetAt(text: string, pos: vscode.Position): number {
  const lines = text.split('\n');
  let offset = 0;
  const line = Math.max(0, Math.min(lines.length - 1, pos.line));
  for (let i = 0; i < line; i += 1) {
    offset += lines[i].length + 1;
  }
  return Math.max(0, Math.min(text.length, offset + pos.character));
}

// Merge Intervals.
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

// Invert Intervals.
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
