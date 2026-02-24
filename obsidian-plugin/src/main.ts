import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import type * as CmStateRuntime from '@codemirror/state';
import type * as CmViewRuntime from '@codemirror/view';
import type { Extension, Range } from '@codemirror/state';
import type { Decoration as DecorationType, DecorationSet, EditorView, Tooltip, ViewUpdate } from '@codemirror/view';
import {
  App,
  addIcon,
  FileSystemAdapter,
  ItemView,
  MarkdownView,
  Modal,
  Plugin,
  PluginSettingTab,
  Setting,
  TAbstractFile,
  TFile,
  WorkspaceLeaf,
} from 'obsidian';
import { BackendClient, BridgeRpcError } from './backendClient';
import { AnalyzeResult, ChunkMetrics, ChunkState, ParagraphRow, RewriteOption } from './types';

interface EditedRange {
  start: number;
  end: number;
}

interface ContentChange {
  rangeOffset: number;
  rangeLength: number;
  text: string;
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
  html: string;
  isMinorContributor: boolean;
  segmentStart: number;
  segmentEnd: number;
  rangeStart: number;
  rangeEnd: number;
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

interface BinocularsSettings {
  enabled: boolean;
  editorIntegrationEnabled: boolean;
  backendPythonPath: string;
  backendBridgeScriptPath: string;
  configPath: string;
  textMaxTokensOverride: number | null;
  topK: number;
  observerGgufPath: string;
  performerGgufPath: string;
  externalLlmEnabled: boolean;
  externalLlmConfigPath: string;
  externalLlmEndpoint: string;
  externalLlmModel: string;
  externalLlmTemperature: number;
  externalLlmMaxTokens: number;
  renderContributionBars: boolean;
  renderColorizeText: boolean;
  diagnosticLogging: boolean;
}

interface RenderArtifacts {
  lineContribution: Map<number, { sign: 'red' | 'green'; mag: number; anchorOffset: number }>;
  maxLineContributionMag: number;
}

interface EditorContext {
  view: MarkdownView;
  file: TFile;
  editor: EditorLike;
  key: string;
  text: string;
  cursorOffset: number;
  cursorLine: number;
  selectionStart: number;
  selectionEnd: number;
}

interface EditorLike {
  getValue(): string;
  getCursor(which?: 'from' | 'to' | string): { line: number; ch: number };
  replaceRange(text: string, from: { line: number; ch: number }, to?: { line: number; ch: number }): void;
  setCursor(pos: { line: number; ch: number }): void;
}

type MarkdownMode = 'source' | 'preview' | 'unknown';

type RuntimeRequireFn = ((moduleName: string) => unknown) & {
  resolve?: (moduleName: string) => string;
};

interface RuntimeRequireCandidate {
  source: string;
  req: RuntimeRequireFn;
}

interface RuntimeModuleLoad<T> {
  module: T | null;
  source: string;
  resolvedPath: string;
  error: string;
}

const DISPLAY_DECIMALS = 5;
const LIVE_ESTIMATE_DEBOUNCE_MS = 900;
const HOVER_TYPING_SUPPRESS_MS = 1300;
const HOVER_SAME_SEGMENT_SUPPRESS_MS = 900;
const HOVER_CONTRIBUTOR_DELAY_MS = 1150;
const ENABLE_TEXT_SEGMENT_HOVER = true;
const GUTTER_LEVELS = 21;
const CONTROLS_VIEW_TYPE = 'binoculars-controls-view';
const BINOCULARS_OWL_ICON_ID = 'binoculars-owl';
const BINOCULARS_OWL_ICON_SVG = `
  <path d="M16.7 37.5a33.3 33.3 0 1 0 66.6 0V16.7l-12.5 8.8L50 16.7 29.2 25.5 16.7 16.7v20.8Z" stroke="currentColor" stroke-width="7.5" stroke-linecap="round" stroke-linejoin="round" fill="none"></path>
  <circle cx="37.5" cy="45.8" r="5.6" fill="currentColor"></circle>
  <circle cx="62.5" cy="45.8" r="5.6" fill="currentColor"></circle>
  <path d="M50 51.3V66.7" stroke="currentColor" stroke-width="7.5" stroke-linecap="round" fill="none"></path>
`;
const STATUS_LOG_FILENAME = 'status.log';
const STATUS_LOG_MAX_BYTES = 2 * 1024 * 1024;
const STATUS_LOG_MAX_LINES = 10000;
const BUSY_BLOCKED_STATUS_MESSAGE = 'An analysis is already in progress... please refresh or return later.';
const SIDECAR_FILE_EXT = '.binoculars';
const LEGACY_SIDECAR_FILE_EXT = '.json';
const ENABLE_STATUS_BAR_ITEM = false;

const DEFAULT_SETTINGS: BinocularsSettings = {
  enabled: true,
  editorIntegrationEnabled: false,
  backendPythonPath: '/home/npepin/Projects/binoculars/venv/bin/python',
  backendBridgeScriptPath: '/home/npepin/Projects/binoculars/vscode-extension/python/binoculars_bridge.py',
  configPath: '/home/npepin/Projects/binoculars/config.llama31.cuda12gb.fast.json',
  textMaxTokensOverride: null,
  topK: 5,
  observerGgufPath:
    '/home/npepin/Projects/binoculars/models/Meta-Llama-3.1-8B-Q5_K_M-GGUF/meta-llama-3.1-8b-q5_k_m.gguf',
  performerGgufPath:
    '/home/npepin/Projects/binoculars/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M-GGUF/meta-llama-3.1-8b-instruct-q5_k_m.gguf',
  externalLlmEnabled: true,
  externalLlmConfigPath: '/home/npepin/Projects/binoculars/config.binoculars.llm.json',
  externalLlmEndpoint: '',
  externalLlmModel: '',
  externalLlmTemperature: 0.7,
  externalLlmMaxTokens: 280,
  renderContributionBars: true,
  renderColorizeText: true,
  diagnosticLogging: true,
};

function formatRequireLoadError(err: unknown): string {
  if (err instanceof Error) {
    return err.message;
  }
  return String(err ?? 'unknown error');
}

function normalizeFsPath(input: string): string {
  return path.resolve(input).split(path.sep).join('/');
}

function collectRequireCandidates(): RuntimeRequireCandidate[] {
  const candidates: RuntimeRequireCandidate[] = [];
  const seen = new Set<unknown>();
  const pushCandidate = (source: string, reqLike: unknown): void => {
    if (typeof reqLike !== 'function' || seen.has(reqLike)) {
      return;
    }
    seen.add(reqLike);
    candidates.push({ source, req: reqLike as RuntimeRequireFn });
  };

  pushCandidate('global-require', (globalThis as Record<string, unknown>).require);
  const win = (globalThis as { window?: unknown }).window;
  if (win && typeof win === 'object') {
    pushCandidate('window-require', (win as Record<string, unknown>).require);
  }
  if (typeof require === 'function') {
    pushCandidate('scoped-require', require as RuntimeRequireFn);
  }
  if (typeof module !== 'undefined' && module && typeof module.require === 'function') {
    pushCandidate('module-require', module.require.bind(module) as RuntimeRequireFn);
  }
  return candidates;
}

function loadRuntimeModule<T>(moduleName: string): RuntimeModuleLoad<T> {
  const attempts: string[] = [];
  const localNodeModulesRoot = `${normalizeFsPath(path.join(__dirname, 'node_modules'))}/`;
  const candidates = collectRequireCandidates();

  for (const candidate of candidates) {
    try {
      const loaded = candidate.req(moduleName) as T;
      let resolvedPath = '';
      if (typeof candidate.req.resolve === 'function') {
        try {
          resolvedPath = String(candidate.req.resolve(moduleName) ?? '');
        } catch (resolveErr) {
          attempts.push(`${candidate.source}.resolve: ${formatRequireLoadError(resolveErr)}`);
        }
      }
      if (resolvedPath) {
        const normalizedResolved = normalizeFsPath(resolvedPath);
        if (
          normalizedResolved.startsWith(localNodeModulesRoot) &&
          normalizedResolved.includes('/node_modules/@codemirror/')
        ) {
          attempts.push(
            `${candidate.source}: resolved to plugin-local codemirror (${resolvedPath}); skipped to avoid duplicate runtime instance`,
          );
          continue;
        }
      }
      return {
        module: loaded,
        source: candidate.source,
        resolvedPath: resolvedPath || `<unresolved:${candidate.source}>`,
        error: '',
      };
    } catch (err) {
      attempts.push(`${candidate.source}: ${formatRequireLoadError(err)}`);
    }
  }

  return {
    module: null,
    source: '',
    resolvedPath: '',
    error: `Unable to resolve module ${moduleName}. Attempts: ${attempts.join(' | ')}`,
  };
}

const loadedCmState = loadRuntimeModule<typeof CmStateRuntime>('@codemirror/state');
const loadedCmView = loadRuntimeModule<typeof CmViewRuntime>('@codemirror/view');
const CM_STATE = loadedCmState.module;
const CM_VIEW = loadedCmView.module;
const cmRuntimeLoadError = [loadedCmState.error, loadedCmView.error].filter(Boolean).join(' || ');

const refreshDecorationsEffect = CM_STATE?.StateEffect.define<null>();
const refreshDecorationsField =
  CM_STATE && refreshDecorationsEffect
    ? CM_STATE.StateField.define<number>({
        create: () => 0,
        update(value, tr) {
          for (const effect of tr.effects) {
            if (effect.is(refreshDecorationsEffect)) {
              return value + 1;
            }
          }
          return value;
        },
      })
    : undefined;

const lowMajorDeco = CM_VIEW?.Decoration.mark({
  class: 'binoculars-low-major',
  attributes: {
    style:
      'color: #ff4d4f !important; background-color: rgba(255, 77, 79, 0.18) !important; border-radius: 2px;',
  },
});
const highMajorDeco = CM_VIEW?.Decoration.mark({
  class: 'binoculars-high-major',
  attributes: {
    style:
      'color: #2fbf71 !important; background-color: rgba(47, 191, 113, 0.18) !important; border-radius: 2px;',
  },
});
const lowMinorDeco = CM_VIEW?.Decoration.mark({
  class: 'binoculars-low-minor',
  attributes: {
    style: 'color: var(--text-normal) !important;',
  },
});
const highMinorDeco = CM_VIEW?.Decoration.mark({
  class: 'binoculars-high-minor',
  attributes: {
    style: 'color: var(--text-normal) !important;',
  },
});
const priorLowDeco = CM_VIEW?.Decoration.mark({
  class: 'binoculars-prior-low',
  attributes: {
    style: 'background-color: rgba(255, 77, 79, 0.14) !important; border-radius: 2px;',
  },
});
const priorHighDeco = CM_VIEW?.Decoration.mark({
  class: 'binoculars-prior-high',
  attributes: {
    style: 'background-color: rgba(47, 191, 113, 0.14) !important; border-radius: 2px;',
  },
});
const unscoredDeco = CM_VIEW?.Decoration.mark({
  class: 'binoculars-unscored',
  attributes: {
    style: 'color: #7d8792 !important; opacity: 0.72;',
  },
});
const editedDeco = CM_VIEW?.Decoration.mark({
  class: 'binoculars-edited',
  attributes: {
    style: 'background-color: rgba(255, 213, 79, 0.26) !important; border-radius: 2px;',
  },
});

class ConfirmModal extends Modal {
  private resolver: ((value: boolean) => void) | undefined;
  private readonly message: string;

  constructor(app: App, message: string) {
    super(app);
    this.message = message;
  }

  public openAndWait(): Promise<boolean> {
    return new Promise<boolean>((resolve) => {
      this.resolver = resolve;
      this.open();
    });
  }

  override onOpen(): void {
    const { contentEl } = this;
    contentEl.empty();
    contentEl.createEl('h3', { text: 'Binoculars' });
    contentEl.createEl('p', { text: this.message });
    const row = contentEl.createDiv({ cls: 'binoculars-modal-actions' });
    const yesBtn = row.createEl('button', { text: 'Yes' });
    const noBtn = row.createEl('button', { text: 'No' });
    yesBtn.addEventListener('click', () => {
      this.resolver?.(true);
      this.resolver = undefined;
      this.close();
    });
    noBtn.addEventListener('click', () => {
      this.resolver?.(false);
      this.resolver = undefined;
      this.close();
    });
  }

  override onClose(): void {
    if (this.resolver) {
      this.resolver(false);
      this.resolver = undefined;
    }
    this.contentEl.empty();
  }
}

class RewritePickerModal extends Modal {
  private readonly source: string;
  private readonly options: RewriteOption[];
  private resolver: ((value: RewriteOption | undefined) => void) | undefined;

  constructor(app: App, source: string, options: RewriteOption[]) {
    super(app);
    this.source = source;
    this.options = options.slice(0, 3);
  }

  public openAndWait(): Promise<RewriteOption | undefined> {
    return new Promise<RewriteOption | undefined>((resolve) => {
      this.resolver = resolve;
      this.open();
    });
  }

  override onOpen(): void {
    const { contentEl } = this;
    contentEl.empty();
    contentEl.createEl('h3', { text: `Binoculars Rewrite (${this.source})` });
    contentEl.createEl('p', { text: 'Choose a rewrite option to apply. Keyboard: 1/2/3 apply, Q/Esc cancel.' });

    this.options.forEach((opt, idx) => {
      const card = contentEl.createDiv({ cls: 'binoculars-rewrite-card' });
      const row = card.createDiv({ cls: 'binoculars-rewrite-head' });
      row.createEl('strong', { text: `[${idx + 1}] ${formatApprox(opt.approx_B, opt.delta_B)}` });
      const applyBtn = row.createEl('button', { text: `Apply ${idx + 1}` });
      applyBtn.addEventListener('click', () => this.finish(opt));
      card.createEl('pre', { text: String(opt.text || '') });
    });

    const cancelRow = contentEl.createDiv({ cls: 'binoculars-modal-actions' });
    const cancelBtn = cancelRow.createEl('button', { text: 'Cancel' });
    cancelBtn.addEventListener('click', () => this.finish(undefined));

    this.scope.register([], 'Escape', () => {
      this.finish(undefined);
      return false;
    });
    this.scope.register([], 'q', () => {
      this.finish(undefined);
      return false;
    });
    this.scope.register([], '1', () => {
      if (this.options[0]) {
        this.finish(this.options[0]);
      }
      return false;
    });
    this.scope.register([], '2', () => {
      if (this.options[1]) {
        this.finish(this.options[1]);
      }
      return false;
    });
    this.scope.register([], '3', () => {
      if (this.options[2]) {
        this.finish(this.options[2]);
      }
      return false;
    });
  }

  override onClose(): void {
    if (this.resolver) {
      this.resolver(undefined);
      this.resolver = undefined;
    }
    this.contentEl.empty();
  }

  private finish(value: RewriteOption | undefined): void {
    if (!this.resolver) {
      return;
    }
    const resolve = this.resolver;
    this.resolver = undefined;
    resolve(value);
    this.close();
  }
}

class StatusLogModal extends Modal {
  private readonly plugin: BinocularsPlugin;

  constructor(app: App, plugin: BinocularsPlugin) {
    super(app);
    this.plugin = plugin;
  }

  override onOpen(): void {
    const { contentEl } = this;
    contentEl.empty();
    contentEl.createEl('h3', { text: 'Binoculars Status Log' });
    const pathLine = contentEl.createEl('p');
    pathLine.createEl('code', { text: this.plugin.getStatusLogPath() ?? '<unavailable>' });
    const text = this.plugin.readStatusLogContents() ?? '<status log is empty>';
    const pre = contentEl.createEl('pre', { text });
    pre.style.maxHeight = '65vh';
    pre.style.overflow = 'auto';

    const actions = contentEl.createDiv({ cls: 'binoculars-modal-actions' });
    const clearBtn = actions.createEl('button', { text: 'Clear Log' });
    const closeBtn = actions.createEl('button', { text: 'Close' });
    clearBtn.addEventListener('click', () => {
      this.plugin.clearStatusLog();
      pre.textContent = this.plugin.readStatusLogContents() ?? '<status log is empty>';
    });
    closeBtn.addEventListener('click', () => this.close());
  }

  override onClose(): void {
    this.contentEl.empty();
  }
}

class BinocularsControlsView extends ItemView {
  private plugin: BinocularsPlugin;

  constructor(leaf: WorkspaceLeaf, plugin: BinocularsPlugin) {
    super(leaf);
    this.plugin = plugin;
  }

  getViewType(): string {
    return CONTROLS_VIEW_TYPE;
  }

  getDisplayText(): string {
    return 'Binoculars';
  }

  getIcon(): string {
    return this.plugin.getUiIconId();
  }

  async onOpen(): Promise<void> {
    this.render();
  }

  async onClose(): Promise<void> {
    this.contentEl.empty();
  }

  public render(): void {
    const root = this.contentEl;
    root.empty();

    const bindSinglePress = (btn: HTMLButtonElement, action: () => void): void => {
      let swallowClick = false;
      let swallowTimer: number | undefined;

      const clearSwallow = (): void => {
        swallowClick = false;
        if (typeof swallowTimer === 'number') {
          window.clearTimeout(swallowTimer);
          swallowTimer = undefined;
        }
      };

      const armSwallow = (): void => {
        swallowClick = true;
        if (typeof swallowTimer === 'number') {
          window.clearTimeout(swallowTimer);
        }
        swallowTimer = window.setTimeout(() => {
          clearSwallow();
        }, 1500);
      };

      btn.addEventListener('pointerdown', (evt: PointerEvent) => {
        if (evt.button !== 0) {
          return;
        }
        armSwallow();
        action();
      });

      btn.addEventListener('click', () => {
        if (swallowClick) {
          clearSwallow();
          return;
        }
        action();
      });
    };

    const title = root.createEl('h3', { text: 'Binoculars' });
    title.addClass('binoculars-controls-title');

    if (!this.plugin.isExtensionEnabled()) {
      const enableBtn = root.createEl('button', { text: 'Enable' });
      bindSinglePress(enableBtn, () => {
        void this.plugin.enableExtension({ openControlsIfMissing: true });
      });
      return;
    }

    const addCmdBtn = (label: string, commandId: string): void => {
      const btn = root.createEl('button', { text: label });
      btn.addClass('binoculars-controls-btn');
      bindSinglePress(btn, () => {
        void this.plugin.executeControlsCommand(commandId);
      });
    };

    const busy = this.plugin.isControlsBlockedByInFlightAnalysis();
    const hasAnalysis = this.plugin.hasAnyAnalysisForPreferredNote();
    const hasPriors = this.plugin.hasAnyPriorsForPreferredNote();
    if (busy) {
      addCmdBtn('Refresh', 'binoculars-obsidian:refresh-controls');
    } else {
      const analyzePrimaryLabel = hasAnalysis ? 'Re-Analyze Chunk' : 'Analyze Chunk';
      addCmdBtn(analyzePrimaryLabel, 'binoculars-obsidian:analyze-chunk');
      if (this.plugin.hasNextChunkAvailable()) {
        addCmdBtn('Analyze Next Chunk', 'binoculars-obsidian:analyze-next-chunk');
        addCmdBtn('Analyze All', 'binoculars-obsidian:analyze-all');
      }
    }
    if (hasAnalysis) {
      addCmdBtn('Rewrite Selection', 'binoculars-obsidian:rewrite-selection-or-line');
    }
    if (hasPriors) {
      addCmdBtn('Clear Priors', 'binoculars-obsidian:clear-priors');
    }
    if (hasAnalysis) {
      addCmdBtn(
        this.plugin.isRuntimeColorizationEnabled() ? 'Hide Colorization' : 'Show Colorization',
        'binoculars-obsidian:toggle-colorization',
      );
    }
    addCmdBtn('Restart Backend', 'binoculars-obsidian:restart-backend');
    addCmdBtn('Show Status Log', 'binoculars-obsidian:show-status-log');
    addCmdBtn('Disable', 'binoculars-obsidian:disable');

    const status = root.createDiv({ cls: 'binoculars-controls-status' });
    status.createEl('strong', { text: 'Status' });
    status.createEl('div', { text: this.plugin.getControlsStatusMessage() });

    const chunkCount = this.plugin.activeChunkCount();
    const chunkRow = root.createDiv({ cls: 'binoculars-controls-status' });
    chunkRow.createEl('strong', { text: 'Analyzed Chunks' });
    const chunkValue = chunkRow.createDiv();
    if (chunkCount <= 0) {
      chunkValue.createEl('em', { text: 'none' });
    } else {
      chunkValue.setText(String(chunkCount));
    }
  }
}

export default class BinocularsPlugin extends Plugin {
  private settings: BinocularsSettings = { ...DEFAULT_SETTINGS };
  private uiIconId = 'telescope';
  private ribbonToggleEl: HTMLElement | undefined;
  private backend: BackendClient | undefined;
  private backendStarted = false;
  private statusLogPath: string | undefined;
  private readonly logSessionId = `${Date.now()}-${Math.floor(Math.random() * 1_000_000)}`;
  private statusBarEl!: HTMLElement;
  private lastStatusMessage = 'Binoculars Ready. Select Analyze Chunk to begin.';
  private runtimeColorizationEnabled = true;
  private foregroundBusyOperationCount = 0;
  private foregroundBusyDocKey: string | undefined;
  private foregroundBusyKind: 'analysis' | 'rewrite' | 'other' | undefined;

  // Document-scoped runtime state, keyed by vault-relative markdown path.
  private readonly docStates = new Map<string, DocumentState>();
  // "<text-hash>:<sidecar-hash>" signatures to avoid redundant sidecar loads.
  private readonly loadedSidecarSignatures = new Map<string, string>();
  // Debounced live-estimate orchestration (timer + epoch + single retry marker).
  private readonly liveEstimateTimers = new Map<string, number>();
  private readonly liveEstimateEpochs = new Map<string, number>();
  private readonly liveEstimateRecoverAttempts = new Map<string, number>();
  // Recent typing and hover dedupe windows for less noisy tooltip behavior.
  private readonly recentTypingActivity = new Map<string, { atMs: number; start: number; end: number }>();
  private readonly hoverSeenSegments = new Map<string, { start: number; end: number; lastSeenMs: number }>();
  // Lightweight counters/signatures for status + diagnostics.
  private readonly docVersions = new Map<string, number>();
  private readonly lastDecorationSummaryByDoc = new Map<string, string>();
  private readonly renderArtifactsByView = new WeakMap<EditorView, RenderArtifacts>();
  private lastMissingContextLogAtMs = 0;
  private lastUnmappedCmViewLogAtMs = 0;
  private statusLogWriteCount = 0;
  private visualsDisabledHintShown = false;
  private lastActiveMarkdownFilePath: string | undefined;
  private lastOpenedMarkdownFilePath: string | undefined;

  private controlsView: BinocularsControlsView | undefined;

  async onload(): Promise<void> {
    await this.loadPluginSettings();
    this.initializeStatusLog();
    this.registerCustomIcons();
    this.logStatus('plugin.onload.begin', {
      enabled: this.settings.enabled,
      vaultRoot: this.vaultBasePath() ?? '',
      pluginDir: this.pluginDirectoryPath() ?? '',
    });
    this.logStatus('cm.runtime.resolve', {
      stateSource: loadedCmState?.source ?? '',
      statePath: loadedCmState?.resolvedPath ?? '',
      viewSource: loadedCmView?.source ?? '',
      viewPath: loadedCmView?.resolvedPath ?? '',
      loadError: cmRuntimeLoadError,
      cwd: process.cwd(),
      entryDir: __dirname,
    });

    this.statusBarEl = this.addStatusBarItem();
    this.statusBarEl.addClass('binoculars-status-bar');
    if (!ENABLE_STATUS_BAR_ITEM) {
      this.statusBarEl.hide();
    }

    this.registerView(CONTROLS_VIEW_TYPE, (leaf) => {
      const view = new BinocularsControlsView(leaf, this);
      this.controlsView = view;
      return view;
    });

    this.ribbonToggleEl = this.addRibbonIcon(this.uiIconId, this.currentRibbonToggleTooltip(), () => {
      void this.togglePluginEnablementFromRibbon();
    });
    this.refreshRibbonToggleTooltip();

    this.addCommand({
      id: 'open-controls',
      name: 'Binoculars: Open Controls',
      callback: () => void this.openControlsView(),
    });

    this.addCommand({
      id: 'enable',
      name: 'Binoculars: Enable',
      callback: () => void this.enableExtension({ openControlsIfMissing: true }),
    });

    this.addCommand({
      id: 'disable',
      name: 'Binoculars: Disable',
      callback: () => void this.disableExtension(),
    });

    this.addCommand({
      id: 'analyze-chunk',
      name: 'Binoculars: Analyze Chunk',
      editorCallback: () => void this.runAnalyze(),
      callback: () => void this.runAnalyze(),
    });

    this.addCommand({
      id: 'analyze-next-chunk',
      name: 'Binoculars: Analyze Next Chunk',
      editorCallback: () => void this.runAnalyzeNext(),
      callback: () => void this.runAnalyzeNext(),
    });

    this.addCommand({
      id: 'analyze-all',
      name: 'Binoculars: Analyze All',
      editorCallback: () => void this.runAnalyzeAll(),
      callback: () => void this.runAnalyzeAll(),
    });

    this.addCommand({
      id: 'rewrite-selection',
      name: 'Binoculars: Rewrite Selection',
      editorCallback: () => void this.runRewriteSelection(),
      callback: () => void this.runRewriteSelection(),
    });

    this.addCommand({
      id: 'rewrite-selection-or-line',
      name: 'Binoculars: Rewrite Selection',
      editorCallback: () => void this.runRewriteSelectionOrLine(),
      callback: () => void this.runRewriteSelectionOrLine(),
    });

    this.addCommand({
      id: 'clear-priors',
      name: 'Binoculars: Clear Priors',
      callback: () => this.clearPriors(),
    });

    this.addCommand({
      id: 'toggle-colorization',
      name: 'Binoculars: Toggle Colorization',
      callback: () => this.toggleColorization(),
    });

    this.addCommand({
      id: 'restart-backend',
      name: 'Binoculars: Restart Backend',
      callback: () => void this.restartBackend(),
    });

    this.addCommand({
      id: 'show-status-log',
      name: 'Binoculars: Show Status Log',
      callback: () => this.showStatusLog(),
    });

    this.addCommand({
      id: 'copy-status-log-path',
      name: 'Binoculars: Copy Status Log Path',
      callback: () => void this.copyStatusLogPath(),
    });

    this.addCommand({
      id: 'refresh-controls',
      name: 'Binoculars: Refresh Controls',
      callback: () => this.refreshControlsAfterBusyCheck(),
    });

    if (this.settings.editorIntegrationEnabled) {
      this.logStatus('editor-integration.register.begin');
      if (!this.isCmRuntimeReady()) {
        this.settings.editorIntegrationEnabled = false;
        this.logStatus('editor-integration.runtime-unavailable', {
          loadError: cmRuntimeLoadError,
        });
        this.updateStatus('Loaded with editor overlays disabled (CodeMirror runtime unavailable).');
      } else {
      try {
        this.registerEditorExtension(this.buildEditorExtension());
        this.logStatus('editor-integration.register.end');
      } catch (err) {
        this.settings.editorIntegrationEnabled = false;
        this.logStatus('editor-integration.register.error', {
          error: formatErrorForLog(err),
        });
        this.updateStatus(
          'Loaded with editor overlays disabled (editor integration init failed). Check status.log for details.',
        );
      }
      }
    } else {
      this.logStatus('editor-integration.disabled');
    }

    this.registerEvent(
      this.app.workspace.on('active-leaf-change', (leaf) => {
        this.safeRun('event.active-leaf-change', () => {
          const leafType = this.describeLeafType(leaf);
          const leafFile = this.describeLeafFile(leaf);
          const leafMode = this.describeLeafMode(leaf);
          this.logStatus('event.active-leaf-change', {
            leafType,
            file: leafFile,
            mode: leafMode,
          });
          if (!this.isExtensionEnabled()) {
            return;
          }
          if (leafType !== 'markdown') {
            this.refreshAllDecorations();
            this.updateStatusForActiveEditor();
            this.refreshControlsView();
            return;
          }
          if (leafFile) {
            this.lastActiveMarkdownFilePath = leafFile;
          }
          const ctx = this.activeContext();
          this.logStatus('event.active-leaf-change.context', {
            hasContext: Boolean(ctx),
            contextFile: ctx?.file.path ?? '',
            contextMode: this.markdownModeForView(ctx?.view),
          });
          if (ctx && this.isMarkdownSidecarEligible(ctx.file)) {
            this.maybeLoadStateSidecar(ctx.file, ctx.text, 'activate');
            if (this.docStates.has(ctx.file.path)) {
              this.refreshDecorationsForFile(ctx.file.path);
            }
            this.refreshAllDecorations();
            this.updateStatusForActiveEditor();
            this.refreshControlsView();
            return;
          }
          // During markdown-leaf initialization, editor context can lag briefly.
          // Re-check once after a short delay to avoid missing sidecar restore.
          window.setTimeout(() => {
            this.safeRun('event.active-leaf-change.deferred', () => {
              if (!this.isExtensionEnabled()) {
                return;
              }
              const deferredCtx = this.activeContext();
              this.logStatus('event.active-leaf-change.deferred.context', {
                hasContext: Boolean(deferredCtx),
                contextFile: deferredCtx?.file.path ?? '',
                contextMode: this.markdownModeForView(deferredCtx?.view),
                leafFile,
              });
              if (deferredCtx && this.isMarkdownSidecarEligible(deferredCtx.file)) {
                this.maybeLoadStateSidecar(deferredCtx.file, deferredCtx.text, 'activate');
                if (this.docStates.has(deferredCtx.file.path)) {
                  this.refreshDecorationsForFile(deferredCtx.file.path);
                }
              } else if (leafFile && this.docStates.has(leafFile)) {
                this.refreshDecorationsForFile(leafFile);
              }
              this.refreshAllDecorations();
              this.updateStatusForActiveEditor();
              this.refreshControlsView();
            });
          }, 80);
        });
      }),
    );

    this.registerEvent(
      this.app.workspace.on('file-open', (file) => {
        this.safeRun('event.file-open', () => {
          this.logStatus('event.file-open', {
            file: file instanceof TFile ? file.path : '<non-file>',
            isMarkdown: file instanceof TFile ? this.isMarkdownSidecarEligible(file) : false,
          });
          if (!this.isExtensionEnabled()) {
            return;
          }
          if (!(file instanceof TFile) || !this.isMarkdownSidecarEligible(file)) {
            return;
          }
          this.lastOpenedMarkdownFilePath = file.path;
          const ctx = this.activeContext();
          this.logStatus('event.file-open.context', {
            hasContext: Boolean(ctx),
            contextFile: ctx?.file.path ?? '',
            openedFile: file.path,
          });
          if (ctx && ctx.file.path === file.path) {
            this.maybeLoadStateSidecar(file, ctx.text, 'open');
            if (this.docStates.has(file.path)) {
              this.refreshDecorationsForFile(file.path);
            }
            this.updateStatusForActiveEditor();
          }
        });
      }),
    );

    this.registerEvent(
      this.app.vault.on('modify', (file) => {
        this.safeRun('event.vault-modify', () => {
          if (!this.isExtensionEnabled()) {
            return;
          }
          if (!(file instanceof TFile) || !this.isMarkdownSidecarEligible(file)) {
            return;
          }
          const ctx = this.activeContext();
          this.logStatus('event.vault-modify.context', {
            hasContext: Boolean(ctx),
            contextFile: ctx?.file.path ?? '',
            modifiedFile: file.path,
          });
          if (ctx && ctx.file.path === file.path) {
            this.autoSaveStateSidecar(file, ctx.text);
          }
        });
      }),
    );

    this.registerEvent(
      this.app.vault.on('delete', (file) => {
        // Keep sidecar lifecycle coupled to markdown lifecycle.
        void this.handleVaultDelete(file);
      }),
    );

    const onWindowError = (evt: ErrorEvent): void => {
      this.logStatus('window.error', {
        message: evt.message,
        filename: evt.filename,
        lineno: evt.lineno,
        colno: evt.colno,
        error: formatErrorForLog(evt.error),
      });
    };
    const onUnhandledRejection = (evt: PromiseRejectionEvent): void => {
      this.logStatus('window.unhandledrejection', {
        reason: formatErrorForLog(evt.reason),
      });
    };
    window.addEventListener('error', onWindowError);
    window.addEventListener('unhandledrejection', onUnhandledRejection);
    this.register(() => {
      window.removeEventListener('error', onWindowError);
      window.removeEventListener('unhandledrejection', onUnhandledRejection);
    });

    this.addSettingTab(new BinocularsSettingTab(this.app, this));
    await this.applyEnablementMode();
    this.logStatus('plugin.onload.end', { status: this.lastStatusMessage });
  }

  public async executeCommandById(commandId: string): Promise<void> {
    this.logStatus('command.dispatch.request', {
      commandId,
      hasActiveContext: Boolean(this.activeContext()),
    });
    if (this.commandPrefersEditorContext(commandId)) {
      await this.ensureAnyMarkdownLeafActive();
    }
    if (this.isPluginOwnedCommand(commandId)) {
      const handledLocally = await this.executeCommandByIdFallback(commandId);
      this.logStatus('command.dispatch.local', {
        commandId,
        handledLocally,
        hasActiveContextAfter: Boolean(this.activeContext()),
      });
      if (handledLocally) {
        return;
      }
    }
    const commandsApi = this.app as unknown as { commands?: { executeCommandById?: (id: string) => unknown } };
    const dispatchResult = commandsApi.commands?.executeCommandById?.(commandId);
    const result = dispatchResult instanceof Promise ? await dispatchResult : dispatchResult;
    this.logStatus('command.dispatch.result', {
      commandId,
      resultType: typeof result,
      resultBool: typeof result === 'boolean' ? result : undefined,
      hasActiveContextAfter: Boolean(this.activeContext()),
    });
    if (result === false) {
      const fallbackApplied = await this.executeCommandByIdFallback(commandId);
      this.logStatus('command.dispatch.fallback', {
        commandId,
        fallbackApplied,
      });
    }
  }

  public async executeControlsCommand(commandId: string): Promise<void> {
    this.logStatus('controls.command.request', {
      commandId,
      hasActiveContext: Boolean(this.activeContext()),
    });
    const immediateStatus = this.controlsImmediateStatus(commandId);
    if (immediateStatus) {
      this.updateStatus(immediateStatus);
    }
    if (this.commandPrefersEditorContext(commandId)) {
      await this.ensureAnyMarkdownLeafActive();
    }
    const handled = await this.executeCommandByIdFallback(commandId);
    this.logStatus('controls.command.result', {
      commandId,
      handled,
      hasActiveContextAfter: Boolean(this.activeContext()),
    });
    if (!handled) {
      await this.executeCommandById(commandId);
    }
  }

  private commandPrefersEditorContext(commandId: string): boolean {
    return (
      commandId === 'binoculars-obsidian:analyze-chunk' ||
      commandId === 'binoculars-obsidian:analyze-next-chunk' ||
      commandId === 'binoculars-obsidian:analyze-all' ||
      commandId === 'binoculars-obsidian:rewrite-selection' ||
      commandId === 'binoculars-obsidian:rewrite-selection-or-line' ||
      commandId === 'binoculars-obsidian:toggle-colorization'
    );
  }

  private controlsImmediateStatus(commandId: string): string | undefined {
    switch (commandId) {
      case 'binoculars-obsidian:rewrite-selection':
      case 'binoculars-obsidian:rewrite-selection-or-line':
        return 'Preparing rewrite request...';
      default:
        return undefined;
    }
  }

  private isPluginOwnedCommand(commandId: string): boolean {
    return commandId.startsWith(`${this.manifest.id}:`);
  }

  private async executeCommandByIdFallback(commandId: string): Promise<boolean> {
    switch (commandId) {
      case 'binoculars-obsidian:analyze-chunk':
        await this.runAnalyze();
        return true;
      case 'binoculars-obsidian:analyze-next-chunk':
        await this.runAnalyzeNext();
        return true;
      case 'binoculars-obsidian:analyze-all':
        await this.runAnalyzeAll();
        return true;
      case 'binoculars-obsidian:rewrite-selection':
        await this.runRewriteSelection();
        return true;
      case 'binoculars-obsidian:rewrite-selection-or-line':
        await this.runRewriteSelectionOrLine();
        return true;
      case 'binoculars-obsidian:clear-priors':
        this.clearPriors();
        return true;
      case 'binoculars-obsidian:toggle-colorization':
        this.toggleColorization();
        return true;
      case 'binoculars-obsidian:restart-backend':
        await this.restartBackend();
        return true;
      case 'binoculars-obsidian:show-status-log':
        this.showStatusLog();
        return true;
      case 'binoculars-obsidian:copy-status-log-path':
        await this.copyStatusLogPath();
        return true;
      case 'binoculars-obsidian:refresh-controls':
        this.refreshControlsAfterBusyCheck();
        return true;
      case 'binoculars-obsidian:disable':
        await this.disableExtension();
        return true;
      case 'binoculars-obsidian:enable':
        await this.enableExtension({ openControlsIfMissing: true });
        return true;
      case 'binoculars-obsidian:open-controls':
        await this.openControlsView();
        return true;
      default:
        return false;
    }
  }

  private async ensureAnyMarkdownLeafActive(): Promise<void> {
    if (this.activeContextQuiet()) {
      return;
    }
    const markdownLeaf = this.preferredMarkdownLeaf();
    if (!markdownLeaf) {
      this.logStatus('command.context.activate-leaf.missing');
      return;
    }
    try {
      await this.app.workspace.setActiveLeaf(markdownLeaf, true, true);
      this.logStatus('command.context.activate-leaf.ok', {
        file: ((markdownLeaf.view as MarkdownView).file?.path ?? ''),
      });
      await this.waitForEditorContextReady(markdownLeaf);
      const readyCtx = this.activeContextQuiet() ?? this.anyMarkdownContextQuiet();
      if (readyCtx && this.isMarkdownSidecarEligible(readyCtx.file)) {
        this.maybeLoadStateSidecar(readyCtx.file, readyCtx.text, 'activate');
      }
    } catch (err) {
      this.logStatus('command.context.activate-leaf.failed', {
        error: formatErrorForLog(err),
      });
    }
  }

  private editorForMarkdownViewQuiet(view: MarkdownView): EditorLike | undefined {
    const maybeEditor = (view as unknown as { editor?: unknown }).editor;
    if (!maybeEditor || typeof maybeEditor !== 'object') {
      return undefined;
    }
    const candidate = maybeEditor as Partial<EditorLike>;
    if (
      typeof candidate.getValue !== 'function' ||
      typeof candidate.getCursor !== 'function' ||
      typeof candidate.replaceRange !== 'function' ||
      typeof candidate.setCursor !== 'function'
    ) {
      return undefined;
    }
    return candidate as EditorLike;
  }

  private contextFromMarkdownViewQuiet(view: MarkdownView): EditorContext | undefined {
    const file = view.file;
    if (!(file instanceof TFile)) {
      return undefined;
    }
    const editor = this.editorForMarkdownViewQuiet(view);
    if (!editor) {
      return undefined;
    }
    const text = editor.getValue();
    const cursor = editor.getCursor();
    const from = editor.getCursor('from');
    const to = editor.getCursor('to');
    return {
      view,
      file,
      editor,
      key: file.path,
      text,
      cursorOffset: offsetAt(text, cursor),
      cursorLine: cursor.line,
      selectionStart: offsetAt(text, from),
      selectionEnd: offsetAt(text, to),
    };
  }

  private activeContextQuiet(): EditorContext | undefined {
    const view = this.app.workspace.getActiveViewOfType(MarkdownView);
    if (!view) {
      return undefined;
    }
    return this.contextFromMarkdownViewQuiet(view);
  }

  private anyMarkdownContextQuiet(): EditorContext | undefined {
    const preferredLeaf = this.preferredMarkdownLeaf();
    if (preferredLeaf?.view instanceof MarkdownView) {
      const preferred = this.contextFromMarkdownViewQuiet(preferredLeaf.view);
      if (preferred) {
        return preferred;
      }
    }
    for (const md of this.markdownViews()) {
      const ctx = this.contextFromMarkdownViewQuiet(md);
      if (ctx) {
        return ctx;
      }
    }
    return undefined;
  }

  private async waitForEditorContextReady(targetLeaf: WorkspaceLeaf, timeoutMs = 550): Promise<void> {
    const targetFile =
      targetLeaf.view instanceof MarkdownView && targetLeaf.view.file instanceof TFile ? targetLeaf.view.file.path : '';

    const started = Date.now();
    while (Date.now() - started <= timeoutMs) {
      const active = this.activeContextQuiet();
      if (active) {
        this.logStatus('command.context.ready', {
          file: active.file.path,
          elapsedMs: Date.now() - started,
          targetFile,
          active: true,
        });
        return;
      }
      const any = this.anyMarkdownContextQuiet();
      if (any) {
        this.logStatus('command.context.ready', {
          file: any.file.path,
          elapsedMs: Date.now() - started,
          targetFile,
          active: false,
        });
        return;
      }

      if (targetLeaf.isDeferred) {
        try {
          await targetLeaf.loadIfDeferred();
        } catch {
          // ignore
        }
      }

      await new Promise<void>((resolve) => {
        window.setTimeout(resolve, 25);
      });
    }

    this.logStatus('command.context.wait.timeout', {
      timeoutMs,
      targetFile,
    });
  }

  public getStatusLogPath(): string | undefined {
    return this.statusLogPath;
  }

  public readStatusLogContents(): string | undefined {
    if (!this.statusLogPath || !fs.existsSync(this.statusLogPath)) {
      return undefined;
    }
    try {
      return fs.readFileSync(this.statusLogPath, 'utf8');
    } catch (err) {
      this.logStatus('status-log.read.failed', { error: formatErrorForLog(err) });
      return undefined;
    }
  }

  public clearStatusLog(): void {
    if (!this.statusLogPath) {
      return;
    }
    try {
      fs.writeFileSync(this.statusLogPath, '', { encoding: 'utf8' });
      this.logStatus('status-log.cleared');
      this.updateStatus('Status log cleared.');
    } catch (err) {
      this.showError(`Failed to clear status log: ${(err as Error).message}`);
    }
  }

  public showStatusLog(): void {
    new StatusLogModal(this.app, this).open();
  }

  private async copyStatusLogPath(): Promise<void> {
    const p = this.statusLogPath;
    if (!p) {
      this.updateStatus('Status log path is unavailable.');
      return;
    }
    try {
      await navigator.clipboard.writeText(p);
      this.updateStatus('Status log path copied to clipboard.');
    } catch {
      this.updateStatus(`Status log path: ${p}`);
    }
  }

  private describeLeafType(leaf: WorkspaceLeaf | null): string {
    if (!leaf) {
      return '<none>';
    }
    return leaf.view?.getViewType?.() ?? '<unknown>';
  }

  private markdownModeForView(view: MarkdownView | null | undefined): MarkdownMode {
    if (!view) {
      return 'unknown';
    }
    try {
      const mode = (view as unknown as { getMode?: () => unknown }).getMode?.();
      if (mode === 'source' || mode === 'preview') {
        return mode;
      }
    } catch {
      // ignore
    }
    return 'unknown';
  }

  private describeLeafMode(leaf: WorkspaceLeaf | null): MarkdownMode {
    if (!leaf || !(leaf.view instanceof MarkdownView)) {
      return 'unknown';
    }
    return this.markdownModeForView(leaf.view);
  }

  private describeLeafFile(leaf: WorkspaceLeaf | null): string {
    if (!leaf) {
      return '';
    }
    const view = leaf.view as unknown as { file?: TFile };
    return view.file?.path ?? '';
  }

  private safeRun(label: string, fn: () => void): void {
    try {
      fn();
    } catch (err) {
      this.logStatus(`${label}.error`, { error: formatErrorForLog(err) });
      this.showError(`Unexpected plugin error (${label}): ${(err as Error).message}`);
    }
  }

  private initializeStatusLog(): void {
    const pluginDir = this.pluginDirectoryPath();
    if (!pluginDir) {
      return;
    }
    try {
      fs.mkdirSync(pluginDir, { recursive: true });
      this.statusLogPath = path.join(pluginDir, STATUS_LOG_FILENAME);
      if (fs.existsSync(this.statusLogPath)) {
        const stat = fs.statSync(this.statusLogPath);
        if (stat.size > STATUS_LOG_MAX_BYTES) {
          const rotated = `${this.statusLogPath}.1`;
          try {
            fs.unlinkSync(rotated);
          } catch {
            // ignore
          }
          fs.renameSync(this.statusLogPath, rotated);
        }
      }
      this.logStatus('status-log.initialized', { path: this.statusLogPath });
      this.trimStatusLogToMaxLines();
    } catch {
      this.statusLogPath = undefined;
    }
  }

  private trimStatusLogToMaxLines(): void {
    if (!this.statusLogPath || !fs.existsSync(this.statusLogPath)) {
      return;
    }
    try {
      const raw = fs.readFileSync(this.statusLogPath, 'utf8');
      if (!raw) {
        return;
      }
      const lines = raw.split(/\r?\n/);
      if (lines.length <= STATUS_LOG_MAX_LINES) {
        return;
      }
      const trimmed = lines.slice(-STATUS_LOG_MAX_LINES).join('\n');
      const normalized = trimmed.endsWith('\n') ? trimmed : `${trimmed}\n`;
      fs.writeFileSync(this.statusLogPath, normalized, { encoding: 'utf8' });
    } catch {
      // best effort only; never fail plugin behavior on log trimming
    }
  }

  private logStatus(event: string, payload?: Record<string, unknown>): void {
    const line = this.composeLogLine(event, payload);
    console.log(line);
    if (!this.settings.diagnosticLogging || !this.statusLogPath) {
      return;
    }
    try {
      fs.appendFileSync(this.statusLogPath, `${line}\n`, { encoding: 'utf8' });
      this.statusLogWriteCount += 1;
      if (this.statusLogWriteCount >= 64) {
        this.statusLogWriteCount = 0;
        this.trimStatusLogToMaxLines();
      }
    } catch {
      // Avoid recursive logging failure loops.
    }
  }

  private logDecorationSummary(
    docKey: string,
    summary: {
      chunks: number;
      rows: number;
      lowMajor: number;
      highMajor: number;
      lowMinor: number;
      highMinor: number;
      unscored: number;
      priorLow: number;
      priorHigh: number;
      edited: number;
      queued: number;
      emitted: number;
      lineContribution: number;
      colorize: boolean;
      bars: boolean;
      mode?: MarkdownMode;
    },
  ): void {
    if (!this.settings.diagnosticLogging) {
      return;
    }
    const signature = JSON.stringify(summary);
    if (this.lastDecorationSummaryByDoc.get(docKey) === signature) {
      return;
    }
    this.lastDecorationSummaryByDoc.set(docKey, signature);
    this.logStatus('cm.decorations.summary', {
      file: docKey,
      ...summary,
    });
  }

  private composeLogLine(event: string, payload?: Record<string, unknown>): string {
    const base: Record<string, unknown> = {
      ts: new Date().toISOString(),
      session: this.logSessionId,
      event,
    };
    if (payload && Object.keys(payload).length > 0) {
      base.payload = payload;
    }
    return `[binoculars] ${JSON.stringify(base)}`;
  }

  async onunload(): Promise<void> {
    this.logStatus('plugin.onunload.begin');
    for (const key of this.liveEstimateTimers.keys()) {
      this.clearLiveEstimateTimer(key);
    }
    await this.stopBackend({ shutdownDaemon: true });
    this.app.workspace.getLeavesOfType(CONTROLS_VIEW_TYPE).forEach((leaf) => {
      void this.app.workspace.detachLeavesOfType(CONTROLS_VIEW_TYPE);
      leaf.detach();
    });
    this.logStatus('plugin.onunload.end');
  }

  public isExtensionEnabled(): boolean {
    return this.settings.enabled;
  }

  public getLastStatusMessage(): string {
    return this.lastStatusMessage;
  }

  public getControlsStatusMessage(): string {
    if (this.isControlsBlockedByInFlightAnalysis()) {
      return BUSY_BLOCKED_STATUS_MESSAGE;
    }
    return this.lastStatusMessage;
  }

  public isForegroundBusy(): boolean {
    return this.foregroundBusyOperationCount > 0;
  }

  private preferredControlsDocKey(): string | undefined {
    return (this.activeContextQuiet() ?? this.anyMarkdownContextQuiet())?.key;
  }

  private isControlsBusyBlocked(): boolean {
    if (this.foregroundBusyOperationCount <= 0) {
      return false;
    }
    const busyDocKey = this.foregroundBusyDocKey;
    if (!busyDocKey) {
      return false;
    }
    const currentDocKey = this.preferredControlsDocKey();
    if (!currentDocKey) {
      return true;
    }
    return currentDocKey !== busyDocKey;
  }

  public isControlsBlockedByInFlightAnalysis(): boolean {
    return this.isControlsBusyBlocked() && this.foregroundBusyKind === 'analysis';
  }

  private refreshControlsAfterBusyCheck(): void {
    if (!this.isExtensionEnabled()) {
      this.refreshControlsView();
      return;
    }
    if (this.isControlsBlockedByInFlightAnalysis()) {
      this.refreshControlsView();
      return;
    }
    this.updateStatusForActiveEditor();
    this.refreshControlsView();
  }

  public postStatus(message: string): void {
    this.updateStatus(message);
  }

  public getSettings(): BinocularsSettings {
    return this.settings;
  }

  public resetVisualsDisabledHint(): void {
    this.visualsDisabledHintShown = false;
  }

  public isRuntimeColorizationEnabled(): boolean {
    return this.runtimeColorizationEnabled;
  }

  private currentRibbonToggleTooltip(): string {
    return this.isExtensionEnabled() ? 'Disable Binoculars' : 'Enable Binoculars';
  }

  private refreshRibbonToggleTooltip(): void {
    const tooltip = this.currentRibbonToggleTooltip();
    this.ribbonToggleEl?.setAttr('title', tooltip);
    this.ribbonToggleEl?.setAttr('aria-label', tooltip);
  }

  public getUiIconId(): string {
    return this.uiIconId;
  }

  private async togglePluginEnablementFromRibbon(): Promise<void> {
    if (this.isExtensionEnabled()) {
      await this.disableExtension();
    } else {
      await this.enableExtension({ openControlsIfMissing: true, focusControls: true });
    }
  }

  private registerCustomIcons(): void {
    try {
      addIcon(BINOCULARS_OWL_ICON_ID, BINOCULARS_OWL_ICON_SVG);
      this.uiIconId = BINOCULARS_OWL_ICON_ID;
      this.logStatus('icon.registered', { iconId: BINOCULARS_OWL_ICON_ID });
    } catch (err) {
      this.uiIconId = 'telescope';
      this.logStatus('icon.register.failed', {
        iconId: BINOCULARS_OWL_ICON_ID,
        fallbackIconId: this.uiIconId,
        error: formatErrorForLog(err),
      });
    }
  }

  public activeChunkCount(): number {
    const ctx = this.activeContext() ?? this.anyMarkdownContext();
    if (!ctx) {
      return 0;
    }
    return this.docStates.get(ctx.key)?.chunks.length ?? 0;
  }

  public hasAnyAnalysisForPreferredNote(): boolean {
    const ctx = this.activeContext() ?? this.anyMarkdownContext();
    if (!ctx) {
      return false;
    }
    const state = this.docStates.get(ctx.key);
    return Boolean(state && state.chunks.length > 0);
  }

  public hasAnyPriorsForPreferredNote(): boolean {
    const ctx = this.activeContext() ?? this.anyMarkdownContext();
    if (!ctx) {
      return false;
    }
    const state = this.docStates.get(ctx.key);
    if (!state) {
      return false;
    }
    return (state.priorLowRanges?.length ?? 0) > 0 || (state.priorHighRanges?.length ?? 0) > 0;
  }

  public hasNextChunkAvailable(): boolean {
    const ctx = this.activeContext() ?? this.anyMarkdownContext();
    if (!ctx || !this.isExtensionEnabled()) {
      return false;
    }
    const textLen = ctx.text.length;
    if (textLen <= 0) {
      return false;
    }
    const state = this.docStates.get(ctx.key);
    if (!state || state.chunks.length === 0) {
      return false;
    }
    return Math.max(0, state.nextChunkStart) < textLen;
  }

  public async enableExtension(opts?: { openControlsIfMissing?: boolean; focusControls?: boolean }): Promise<void> {
    this.settings.enabled = true;
    await this.savePluginSettings();
    await this.applyEnablementMode();
    if (opts?.focusControls) {
      await this.openControlsView();
      return;
    }
    if (opts?.openControlsIfMissing) {
      const controlsLeaves = this.app.workspace.getLeavesOfType(CONTROLS_VIEW_TYPE);
      if (controlsLeaves.length === 0) {
        await this.openControlsView();
      }
    }
  }

  public async disableExtension(): Promise<void> {
    this.settings.enabled = false;
    await this.savePluginSettings();
    await this.applyEnablementMode();
  }

  private async applyEnablementMode(): Promise<void> {
    this.logStatus('mode.apply.begin', { enabled: this.isExtensionEnabled() });
    if (!this.isExtensionEnabled()) {
      await this.stopBackend({ shutdownDaemon: true });
      for (const key of this.liveEstimateTimers.keys()) {
        this.clearLiveEstimateTimer(key);
      }
      this.liveEstimateEpochs.clear();
      this.liveEstimateRecoverAttempts.clear();
      this.recentTypingActivity.clear();
      this.hoverSeenSegments.clear();
      this.docStates.clear();
      this.loadedSidecarSignatures.clear();
      if (ENABLE_STATUS_BAR_ITEM) {
        this.statusBarEl.hide();
      }
      this.lastStatusMessage = 'Binoculars disabled.';
      this.refreshAllDecorations();
      this.refreshControlsView();
      this.refreshRibbonToggleTooltip();
      this.logStatus('mode.apply.disabled');
      return;
    }

    const ctx = this.activeContext();
    if (ctx && this.isMarkdownSidecarEligible(ctx.file)) {
      this.maybeLoadStateSidecar(ctx.file, ctx.text, 'activate');
    }
    if (ENABLE_STATUS_BAR_ITEM) {
      this.statusBarEl.show();
    }
    if (ctx && this.docStates.has(ctx.file.path)) {
      this.refreshDecorationsForFile(ctx.file.path);
    }
    this.updateStatusForActiveEditor();
    this.refreshControlsView();
    this.refreshRibbonToggleTooltip();
    this.logStatus('mode.apply.enabled');
  }

  private async restartBackend(): Promise<void> {
    if (!this.isExtensionEnabled()) {
      await this.stopBackend({ shutdownDaemon: true });
      return;
    }
    await this.stopBackend({ shutdownDaemon: true });
    await this.ensureBackend();
    this.updateStatus('Binoculars backend restarted.');
  }

  private blockIfForegroundBusy(action: string): boolean {
    if (this.foregroundBusyOperationCount <= 0) {
      return false;
    }
    this.logStatus('command.busy', {
      action,
      busyCount: this.foregroundBusyOperationCount,
    });
    const msg = `Binoculars is busy with another request. Wait for completion, then retry ${action}.`;
    this.updateStatus(msg);
    return true;
  }

  private async runAnalyze(): Promise<void> {
    if (!this.ensureEnabledOrNotify()) {
      return;
    }
    if (this.blockIfForegroundBusy('Analyze')) {
      return;
    }
    this.logStatus('command.analyze.begin');
    let ctx = this.preferredContextForCommand('Analyze');
    if (!ctx) {
      this.logStatus('command.analyze.no-context');
      return;
    }
    ctx = await this.ensureSourceModeForOverlays(ctx, 'Analyze');

    this.clearLiveEstimateTimer(ctx.key);
    const previous = this.docStates.get(ctx.key);
    const activeChunk = previous && previous.chunks.length > 0 ? this.getActiveChunk(ctx.text, ctx.cursorOffset, previous) : undefined;
    const analyzeStart = Math.max(0, Math.min(ctx.text.length, activeChunk?.charStart ?? 0));
    const startLine = lineNumberFromOffset(ctx.text, analyzeStart);
    this.updateStatus(`Analyzing chunk from line ${startLine}...`);

    try {
      const result = await this.runWithBusyNotice(
        `Analyzing chunk from line ${startLine}...`,
        async () => {
          const client = await this.ensureBackend();
          if (previous && previous.chunks.length > 0) {
            return client.analyzeChunk(ctx.text, this.inputLabel(ctx.file), analyzeStart, ctx.text.length);
          }
          return client.analyzeDocument(ctx.text, this.inputLabel(ctx.file));
        },
        { docKey: ctx.key, kind: 'analysis' },
      );
      const incoming = toChunkState(result);
      const topK = Math.max(1, Math.trunc(Number.isFinite(this.settings.topK) ? this.settings.topK : 5));
      const priorRanges = previous ? priorContributorRangesForIncoming(previous, incoming, ctx.text.length, topK) : undefined;

      let state: DocumentState;
      if (previous && previous.chunks.length > 0) {
        previous.priorLowRanges = mergeEditedRanges([...(previous.priorLowRanges ?? []), ...(priorRanges?.low ?? [])]);
        previous.priorHighRanges = mergeEditedRanges([...(previous.priorHighRanges ?? []), ...(priorRanges?.high ?? [])]);
        previous.priorChunkB = priorChunkScoreForIncoming(previous, incoming);
        mergeChunk(previous, incoming);
        previous.nextChunkStart = computeContiguousCoverage(previous.chunks, ctx.text.length);
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

      this.docStates.set(ctx.key, state);
      this.docVersions.set(ctx.key, this.docVersionForKey(ctx.key));
      this.refreshDecorationsForFile(ctx.key);
      this.autoSaveStateSidecar(ctx.file, ctx.text);
      this.updateStatusForActiveEditor();
      this.refreshControlsView();
      this.maybeNotifyVisualsDisabled();
      this.maybeNotifyReadingViewOverlayHidden(ctx.key);
    } catch (err) {
      this.handleOperationFailure('Analyze', err);
    }
  }

  private async runAnalyzeNext(): Promise<void> {
    if (!this.ensureEnabledOrNotify()) {
      return;
    }
    if (this.blockIfForegroundBusy('Analyze Next')) {
      return;
    }
    this.logStatus('command.analyze-next.begin');
    let ctx = this.preferredContextForCommand('Analyze Next');
    if (!ctx) {
      this.logStatus('command.analyze-next.no-context');
      return;
    }
    ctx = await this.ensureSourceModeForOverlays(ctx, 'Analyze Next');

    this.clearLiveEstimateTimer(ctx.key);
    let existing = this.docStates.get(ctx.key);
    if (!existing && this.isMarkdownSidecarEligible(ctx.file)) {
      this.maybeLoadStateSidecar(ctx.file, ctx.text, 'activate');
      existing = this.docStates.get(ctx.key);
    }
    if (!existing) {
      await this.runAnalyze();
      return;
    }

    const start = Math.max(0, existing.nextChunkStart);
    if (start >= ctx.text.length) {
      this.updateStatus('All text already covered by analyzed chunks.');
      return;
    }

    const startLine = lineNumberFromOffset(ctx.text, start);
    this.updateStatus(`Analyzing next chunk from line ${startLine}...`);

    try {
      const result = await this.runWithBusyNotice(
        `Analyzing next chunk from line ${startLine}...`,
        async () => {
          const client = await this.ensureBackend();
          return client.analyzeNextChunk(ctx.text, this.inputLabel(ctx.file), start);
        },
        { docKey: ctx.key, kind: 'analysis' },
      );
      const incoming = toChunkState(result);
      const topK = Math.max(1, Math.trunc(Number.isFinite(this.settings.topK) ? this.settings.topK : 5));
      const priorRanges = priorContributorRangesForIncoming(existing, incoming, ctx.text.length, topK);
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
      this.refreshDecorationsForFile(ctx.key);
      this.autoSaveStateSidecar(ctx.file, ctx.text);
      this.updateStatusForActiveEditor();
      this.refreshControlsView();
      this.maybeNotifyVisualsDisabled();
      this.maybeNotifyReadingViewOverlayHidden(ctx.key);
    } catch (err) {
      this.handleOperationFailure('Analyze Next', err);
    }
  }

  private async runAnalyzeAll(): Promise<void> {
    if (!this.ensureEnabledOrNotify()) {
      return;
    }
    if (this.blockIfForegroundBusy('Analyze All')) {
      return;
    }
    this.logStatus('command.analyze-all.begin');
    let ctx = this.preferredContextForCommand('Analyze All');
    if (!ctx) {
      this.logStatus('command.analyze-all.no-context');
      return;
    }
    ctx = await this.ensureSourceModeForOverlays(ctx, 'Analyze All');

    this.clearLiveEstimateTimer(ctx.key);
    const current = this.docStates.get(ctx.key);
    const start = Math.max(0, current?.nextChunkStart ?? 0);
    if (ctx.text.length <= 0 || start >= ctx.text.length) {
      this.updateStatus('All text already covered by analyzed chunks.');
      return;
    }

    const confirm = await new ConfirmModal(this.app, 'Analyze All may take a while on long documents. Are you sure?').openAndWait();
    if (!confirm) {
      this.updateStatus('Analyze All canceled.');
      return;
    }

    const initialVersion = this.docVersionForKey(ctx.key);

    try {
      await this.runWithBusyNotice(
        'Analyzing all remaining chunks...',
        async () => {
          const client = await this.ensureBackend();
          let state = this.docStates.get(ctx.key);
          const fullText = this.currentTextForKey(ctx.key) ?? ctx.text;

        if (!state || state.chunks.length === 0) {
          this.updateStatus('Analyze All in progress: analyzing initial chunk...');
          const result = await client.analyzeDocument(fullText, this.inputLabel(ctx.file));
          const incoming = toChunkState(result);
          const topK = Math.max(1, Math.trunc(Number.isFinite(this.settings.topK) ? this.settings.topK : 5));
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
          this.docStates.set(ctx.key, state);
          this.refreshDecorationsForFile(ctx.key);
        }

        let safetyCounter = 0;
        while (state && state.nextChunkStart < fullText.length) {
          safetyCounter += 1;
          if (safetyCounter > 512) {
            throw new Error('Analyze All aborted due to unexpected excessive chunk iterations.');
          }
          if (this.docVersionForKey(ctx.key) !== initialVersion) {
            throw new Error('Document changed while Analyze All was running. Re-run Analyze All on current text.');
          }

          const loopStart = Math.max(0, state.nextChunkStart);
          const startLine = lineNumberFromOffset(fullText, loopStart);
          this.updateStatus(`Analyze All in progress: analyzing from line ${startLine}...`);
          const result = await client.analyzeNextChunk(fullText, this.inputLabel(ctx.file), loopStart);
          const incoming = toChunkState(result);
          const topK = Math.max(1, Math.trunc(Number.isFinite(this.settings.topK) ? this.settings.topK : 5));
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
          this.refreshDecorationsForFile(ctx.key);
          if (state.nextChunkStart <= prevNext) {
            throw new Error('Analyze All made no forward progress. Try Analyze Chunk again.');
          }
        }
        },
        { docKey: ctx.key, kind: 'analysis' },
      );
      const latest = this.currentTextForKey(ctx.key) ?? ctx.text;
      this.autoSaveStateSidecar(ctx.file, latest);
      this.updateStatusForActiveEditor();
      this.refreshControlsView();
      this.maybeNotifyVisualsDisabled();
      this.maybeNotifyReadingViewOverlayHidden(ctx.key);
    } catch (err) {
      this.handleOperationFailure('Analyze All', err);
    }
  }

  private async runRewriteSelection(): Promise<void> {
    if (!this.ensureEnabledOrNotify()) {
      return;
    }
    if (this.blockIfForegroundBusy('Rewrite')) {
      return;
    }
    this.updateStatus('Preparing rewrite request...');
    this.logStatus('command.rewrite-selection.begin');
    const ctx = this.preferredContextForCommand('Rewrite Selection');
    if (!ctx) {
      this.logStatus('command.rewrite-selection.no-context');
      return;
    }

    let state = this.docStates.get(ctx.key);
    if (!state && this.isMarkdownSidecarEligible(ctx.file)) {
      this.maybeLoadStateSidecar(ctx.file, ctx.text, 'activate');
      state = this.docStates.get(ctx.key);
    }
    if (!state || state.chunks.length === 0) {
      this.showError('Run Analyze first before requesting rewrite suggestions.');
      return;
    }

    const span = this.resolveRewriteSpan(ctx.text, ctx.selectionStart, ctx.selectionEnd, ctx.cursorOffset, state);
    if (!span) {
      this.showError('No scored span available for rewrite at cursor/selection.');
      return;
    }

    await this.runRewriteForSpan(ctx.key, ctx.file, ctx.text, span.start, span.end, state);
  }

  private async runRewriteSelectionOrLine(): Promise<void> {
    if (!this.ensureEnabledOrNotify()) {
      return;
    }
    if (this.blockIfForegroundBusy('Rewrite')) {
      return;
    }
    this.updateStatus('Preparing rewrite request...');
    this.logStatus('command.rewrite-selection-or-line.begin');
    const ctx = this.preferredContextForCommand('Rewrite Selection');
    if (!ctx) {
      this.logStatus('command.rewrite-selection-or-line.no-context');
      return;
    }

    let state = this.docStates.get(ctx.key);
    if (!state && this.isMarkdownSidecarEligible(ctx.file)) {
      this.maybeLoadStateSidecar(ctx.file, ctx.text, 'activate');
      state = this.docStates.get(ctx.key);
    }
    if (!state || state.chunks.length === 0) {
      this.showError('Run Analyze first before requesting rewrite suggestions.');
      return;
    }

    const span = this.resolveSelectionOrLineSpan(ctx.text, ctx.selectionStart, ctx.selectionEnd, ctx.cursorLine);
    if (!span) {
      this.showError('No line or selection is available for rewrite at cursor position.');
      return;
    }

    await this.runRewriteForSpan(ctx.key, ctx.file, ctx.text, span.start, span.end, state);
  }

  private async runRewriteForSpan(
    stateKey: string,
    file: TFile,
    fullText: string,
    start: number,
    end: number,
    state: DocumentState,
  ): Promise<void> {
    const cursorOffset = this.activeContext()?.cursorOffset ?? start;
    const activeChunk = this.getActiveChunk(fullText, cursorOffset, state);
    const baseMetrics = activeChunk?.metrics;
    const initialVersion = this.docVersionForKey(stateKey);

    try {
      const result = await this.runWithBusyNotice(
        'Generating rewrite options...',
        async () => {
          const client = await this.ensureBackend();
          return client.rewriteSpan(
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
        },
        { docKey: stateKey, kind: 'rewrite' },
      );

      const options = (result.rewrites || []).slice(0, 3);
      if (options.length === 0) {
        this.showError('No rewrite options returned.');
        return;
      }

      this.updateStatus('Reviewing rewrite options...');
      const selected = await new RewritePickerModal(this.app, result.source, options).openAndWait();
      if (!selected || !selected.text || !selected.text.trim()) {
        this.updateStatus('Rewrite selection canceled.');
        return;
      }

      if (this.docVersionForKey(stateKey) !== initialVersion) {
        this.showError('Document changed while rewrite options were open. Re-run rewrite for current text.');
        return;
      }

      const ctx = this.activeContext();
      if (!ctx || ctx.key !== stateKey) {
        this.showError('Active editor changed before rewrite could be applied.');
        return;
      }

      const fromPos = positionAt(fullText, start);
      const toPos = positionAt(fullText, end);
      ctx.editor.replaceRange(selected.text, fromPos, toPos);

      const next = this.docStates.get(stateKey);
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
        const exactApproxB = Number(selected.approx_B ?? Number.NaN);
        const rewrittenText = this.currentTextForKey(stateKey) ?? ctx.editor.getValue();
        const rewrittenCursor = Math.max(0, Math.min(rewrittenText.length, start + selected.text.length));
        const rewrittenChunk = this.getActiveChunk(rewrittenText, rewrittenCursor, next);
        if (rewrittenChunk && Number.isFinite(exactApproxB)) {
          next.forecastEstimate = {
            chunkStart: rewrittenChunk.charStart,
            docVersion: this.docVersionForKey(stateKey),
            b: exactApproxB,
          };
        }
        next.forecastPending = true;
      }

      const newText = ctx.editor.getValue();
      ctx.editor.setCursor(positionAt(newText, start + selected.text.length));
      this.refreshDecorationsForFile(stateKey);
      this.scheduleLiveEstimate(stateKey);
      this.updateStatusForActiveEditor();
      this.updateStatus('Rewrite applied. Analyze to confirm exact B.');
      this.autoSaveStateSidecar(file, newText);
    } catch (err) {
      this.handleOperationFailure('Rewrite', err);
    }
  }

  private clearPriors(): void {
    if (!this.ensureEnabledOrNotify()) {
      return;
    }
    const ctx = this.preferredContextForCommand('Clear Priors');
    if (!ctx) {
      return;
    }
    const state = this.docStates.get(ctx.key);
    if (!state) {
      return;
    }
    state.priorLowRanges = [];
    state.priorHighRanges = [];
    this.refreshDecorationsForFile(ctx.key);
    this.updateStatus('Cleared prior highlights.');
  }

  private toggleColorization(): void {
    if (!this.ensureEnabledOrNotify()) {
      return;
    }
    if (this.blockIfForegroundBusy('Toggle Colorization')) {
      return;
    }
    this.runtimeColorizationEnabled = !this.runtimeColorizationEnabled;
    this.refreshAllDecorations();
    this.updateStatusForActiveEditor();
    this.refreshControlsView();
    const visualSuffix = this.visualizationStatusSuffix();
    this.updateStatus(`Colorization ${this.runtimeColorizationEnabled ? 'ON' : 'OFF'}${visualSuffix}.`);
  }

  private async ensureBackend(): Promise<BackendClient> {
    if (!this.isExtensionEnabled()) {
      throw new Error('Binoculars is disabled. Run "Binoculars: Enable" to re-enable.');
    }
    if (!this.backend) {
      this.backend = new BackendClient((line: string) => {
        this.logStatus('bridge.message', { line });
      });
    }

    const pythonPath = this.resolvePathSetting(this.settings.backendPythonPath);
    const bridgeScriptPath = this.resolvePathSetting(this.settings.backendBridgeScriptPath);
    const configPath = this.resolvePathSetting(this.settings.configPath);
    const topK = Math.max(1, Math.trunc(Number.isFinite(this.settings.topK) ? this.settings.topK : 5));
    const textMaxTokensOverride =
      this.settings.textMaxTokensOverride === null || typeof this.settings.textMaxTokensOverride === 'undefined'
        ? null
        : Math.max(0, Math.trunc(this.settings.textMaxTokensOverride));
    const observerModelPath = this.resolvePathSetting(this.settings.observerGgufPath);
    const performerModelPath = this.resolvePathSetting(this.settings.performerGgufPath);
    const rewriteLlmConfigPath = this.resolveRewriteLlmConfigPath();

    const cwd = this.vaultBasePath();

    this.logStatus('backend.ensure.start', {
      pythonPath,
      bridgeScriptPath,
      configPath,
      topK,
      textMaxTokensOverride,
      observerModelPath,
      performerModelPath,
      rewriteLlmConfigPath,
      cwd: cwd ?? '',
    });

    await this.backend.start(pythonPath, bridgeScriptPath, cwd);
    this.backendStarted = true;

    await this.backend.initialize(
      configPath,
      topK,
      textMaxTokensOverride,
      observerModelPath,
      performerModelPath,
      rewriteLlmConfigPath,
    );

    this.logStatus('backend.ensure.ready');

    return this.backend;
  }

  private async stopBackend(opts?: { shutdownDaemon?: boolean }): Promise<void> {
    const shutdownDaemon = opts?.shutdownDaemon === true;
    this.logStatus('backend.stop.begin', { shutdownDaemon, hadBackend: Boolean(this.backend) });
    if (shutdownDaemon) {
      try {
        if (this.backend) {
          await this.backend.shutdownDaemon();
        } else {
          const probe = new BackendClient(() => undefined);
          await probe.shutdownDaemon();
          probe.dispose();
        }
      } catch {
        // ignore
      }
    }
    try {
      await this.backend?.shutdown();
    } catch {
      // ignore
    }
    this.backend = undefined;
    this.backendStarted = false;
    this.logStatus('backend.stop.end');
  }

  private resolveRewriteLlmConfigPath(): string {
    const enabled = this.settings.externalLlmEnabled;
    const configuredPath = this.resolvePathSetting(this.settings.externalLlmConfigPath);
    const endpoint = String(this.settings.externalLlmEndpoint ?? '').trim();
    const model = String(this.settings.externalLlmModel ?? '').trim();
    const temperature = Number(this.settings.externalLlmTemperature);
    const maxTokens = Math.max(1, Math.trunc(Number(this.settings.externalLlmMaxTokens)));

    const hasInlineOverrides = endpoint.length > 0 || model.length > 0;
    if (enabled && !hasInlineOverrides) {
      return configuredPath;
    }

    const pluginDir = this.pluginDirectoryPath();
    if (!pluginDir) {
      return configuredPath;
    }

    try {
      fs.mkdirSync(pluginDir, { recursive: true });
    } catch {
      return configuredPath;
    }

    const runtimePath = path.join(pluginDir, 'rewrite-llm.runtime.json');
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

    const llmObj = payload.llm && typeof payload.llm === 'object' ? { ...(payload.llm as Record<string, unknown>) } : {};
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

  private async openControlsView(): Promise<void> {
    const leaves = this.app.workspace.getLeavesOfType(CONTROLS_VIEW_TYPE);
    if (leaves.length > 0) {
      await this.app.workspace.revealLeaf(leaves[0]);
      this.controlsView?.render();
      return;
    }
    const leaf = this.app.workspace.getRightLeaf(false);
    if (!leaf) {
      return;
    }
    await leaf.setViewState({ type: CONTROLS_VIEW_TYPE, active: true });
    await this.app.workspace.revealLeaf(leaf);
    this.controlsView?.render();
  }

  private refreshControlsView(): void {
    this.controlsView?.render();
  }

  private updateStatus(message: string): void {
    this.lastStatusMessage = message;
    if (!this.isExtensionEnabled()) {
      this.refreshControlsView();
      return;
    }
    if (ENABLE_STATUS_BAR_ITEM) {
      const text = message.startsWith('Binoculars ') ? message : `Binoculars: ${message}`;
      this.statusBarEl.setText(text);
    }
    this.refreshControlsView();
  }

  private visualizationStatusSuffix(): string {
    if (!this.settings.editorIntegrationEnabled) {
      return ' | overlays disabled (enable Editor integration + reload)';
    }
    const activeView = this.app.workspace.getActiveViewOfType(MarkdownView);
    if (this.markdownModeForView(activeView) === 'preview') {
      return ' | overlays hidden in Reading view';
    }
    return '';
  }

  private maybeNotifyVisualsDisabled(): void {
    if (this.settings.editorIntegrationEnabled || this.visualsDisabledHintShown) {
      return;
    }
    this.visualsDisabledHintShown = true;
    this.logStatus('editor-integration.visuals-disabled-hint');
    this.updateStatus(
      'Analyze completed. Visual overlays are disabled. Enable Editor integration and reload plugin to see colorization/gutter bars.',
    );
  }

  private looksLikeVramOomError(err: unknown): boolean {
    const text = (() => {
      if (err instanceof BridgeRpcError) {
        const detailText =
          err.details && typeof err.details === 'object'
            ? JSON.stringify(err.details)
            : '';
        return `${err.message} ${detailText}`;
      }
      if (err instanceof Error) {
        return err.message;
      }
      return String(err ?? '');
    })()
      .toLowerCase()
      .trim();
    if (!text) {
      return false;
    }
    const tokens = [
      'oom_vram',
      'out of memory',
      'cuda out of memory',
      'cuda error out of memory',
      'failed to allocate',
      'allocation failed',
      'cublas_status_alloc_failed',
      'cudamalloc',
      'ggml_backend_cuda',
      'hiperroroutofmemory',
      'not enough memory',
      'insufficient memory',
      'vram',
      'std::bad_alloc',
    ];
    return tokens.some((token) => text.includes(token));
  }

  private looksLikeBridgeTimeout(err: unknown): boolean {
    const text = (err instanceof Error ? err.message : String(err ?? ''))
      .toLowerCase()
      .trim();
    if (!text) {
      return false;
    }
    return (
      text.includes('bridge request timed out') ||
      text.includes('timed out waiting for daemon ready event')
    );
  }

  private formatOperationErrorMessage(action: string, err: unknown): string {
    const bridgeCode = err instanceof BridgeRpcError ? err.code : '';
    if (bridgeCode === 'oom_vram' || this.looksLikeVramOomError(err)) {
      return `${action} failed: GPU VRAM appears exhausted. Try a smaller chunk, lower text.max_tokens override, and close other GPU-heavy apps (Obsidian/VS Code/other model sessions) before retrying.`;
    }
    if (this.looksLikeBridgeTimeout(err)) {
      return `${action} failed: request timed out while waiting on the shared Binoculars daemon. It may be busy with another long-running analysis (Obsidian or VS Code). Wait, then retry, or use Restart Backend if it appears stuck.`;
    }
    const base = err instanceof Error ? err.message : String(err ?? 'unknown error');
    return `${action} failed: ${base}`;
  }

  private handleOperationFailure(action: string, err: unknown): void {
    this.logStatus('operation.failed', {
      action,
      error: formatErrorForLog(err),
      bridgeCode: err instanceof BridgeRpcError ? err.code : '',
      bridgeMethod: err instanceof BridgeRpcError ? err.method : '',
    });
    this.showError(this.formatOperationErrorMessage(action, err));
  }

  private showError(message: string): void {
    this.logStatus('ui.error', { message });
    this.updateStatus(message);
  }

  private isCmRuntimeReady(): boolean {
    return Boolean(
      CM_STATE &&
        CM_VIEW &&
        refreshDecorationsEffect &&
        refreshDecorationsField &&
        lowMajorDeco &&
        highMajorDeco &&
        lowMinorDeco &&
        highMinorDeco &&
        priorLowDeco &&
        priorHighDeco &&
        unscoredDeco &&
        editedDeco,
    );
  }

  private updateStatusForActiveEditor(): void {
    if (!this.isExtensionEnabled()) {
      return;
    }
    if (this.foregroundBusyOperationCount > 0) {
      // Preserve explicit in-flight status (e.g., "Analyzing...") until the
      // foreground operation completes.
      return;
    }
    const visualSuffix = this.visualizationStatusSuffix();
    // Prefer active markdown context when available, but fall back to any open
    // markdown editor so controls-view commands still report useful status.
    const activeMarkdown = this.app.workspace.getActiveViewOfType(MarkdownView);
    const ctx = activeMarkdown ? this.contextFromMarkdownView(activeMarkdown, 'active') : this.anyMarkdownContext();
    if (!ctx) {
      this.updateStatus('Binoculars Ready. Select Analyze Chunk to begin.');
      return;
    }

    const state = this.docStates.get(ctx.key);
    if (!state || state.chunks.length === 0) {
      this.updateStatus('Binoculars Ready. Select Analyze Chunk to begin.');
      return;
    }

    const text = ctx.text;
    const orderedChunks = [...state.chunks].sort((a, b) => a.charStart - b.charStart);
    const cursorOffset = ctx.cursorOffset;
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
        this.updateStatus(`Text starting from line ${startLine} has not been analyzed. Select Analyze Next Chunk to continue.${visualSuffix}`);
      } else {
        this.updateStatus(`Text starting from line ${startLine} has not been analyzed. Select Analyze Chunk to refresh coverage.${visualSuffix}`);
      }
      return;
    }

    if (!cursorChunk.metrics) {
      this.updateStatus(`Binoculars analyzed chunk available.${visualSuffix}`);
      return;
    }

    const exactB = Number(cursorChunk.metrics.binoculars_score);
    const staleSuffix = state.stale ? ' | stale (run Analyze; estimate may differ from exact)' : '';

    const estimateState = (() => {
      const estimate = state.forecastEstimate;
      const hasMatchingEstimate =
        !!estimate &&
        estimate.chunkStart === cursorChunk.charStart &&
        estimate.docVersion === this.docVersionForKey(ctx.key) &&
        Number.isFinite(estimate.b);
      if (!hasMatchingEstimate) {
        return {
          text: '',
          value: undefined as number | undefined,
        };
      }
      return {
        text: state.forecastPending ? `${formatSignedMetric(estimate.b)} (updating...)` : `${formatSignedMetric(estimate.b)}`,
        value: Number(estimate.b),
      };
    })();

    const estimateDiffValue =
      typeof estimateState.value === 'number' && Number.isFinite(estimateState.value) && Number.isFinite(exactB)
        ? estimateState.value - exactB
        : undefined;

    const hasNonZeroNumericEstimate =
      typeof estimateDiffValue === 'number' && Number.isFinite(estimateDiffValue) ? estimateDiffValue !== 0 : false;

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
    const metricCore = `B: ${formatSignedMetric(exactB)}${priorBSuffix}${estimateSuffix} | Obs: ${formatStatusMetric(cursorChunk.metrics.observer_logPPL)} | Cross: ${formatStatusMetric(cursorChunk.metrics.cross_logXPPL)}${staleSuffix}${moreSuffix}${visualSuffix}`;
    if (orderedChunks.length > 1) {
      const chunkIndex = Math.max(1, orderedChunks.indexOf(cursorChunk) + 1);
      this.updateStatus(`Binoculars (chunk ${chunkIndex}): ${metricCore}`);
      return;
    }
    this.updateStatus(metricCore);
  }

  private ensureEnabledOrNotify(): boolean {
    if (this.isExtensionEnabled()) {
      return true;
    }
    this.updateStatus('Binoculars is disabled. Run "Binoculars: Enable" to re-enable.');
    return false;
  }

  private resolvePathSetting(raw: string): string {
    const base = this.vaultBasePath();
    if (!raw) {
      return '';
    }
    if (raw.includes('${workspaceFolder}') && base) {
      return raw.split('${workspaceFolder}').join(base);
    }
    if (raw.includes('${vaultRoot}') && base) {
      return raw.split('${vaultRoot}').join(base);
    }
    return path.isAbsolute(raw) ? raw : base ? path.join(base, raw) : raw;
  }

  private inputLabel(file: TFile): string {
    const abs = this.absolutePathForFile(file);
    return abs ?? file.path;
  }

  private vaultBasePath(): string | undefined {
    const adapter = this.app.vault.adapter;
    if (adapter instanceof FileSystemAdapter) {
      return adapter.getBasePath();
    }
    const maybe = adapter as unknown as { getBasePath?: () => string };
    if (typeof maybe.getBasePath === 'function') {
      return maybe.getBasePath();
    }
    return undefined;
  }

  private pluginDirectoryPath(): string | undefined {
    const base = this.vaultBasePath();
    if (!base) {
      return undefined;
    }
    return path.join(base, '.obsidian', 'plugins', this.manifest.id);
  }

  private absolutePathForFile(file: TFile): string | undefined {
    const base = this.vaultBasePath();
    if (!base) {
      return undefined;
    }
    return path.join(base, file.path);
  }

  private sidecarStatePathForFile(file: TFile, extension = SIDECAR_FILE_EXT, hidden = true): string | undefined {
    const abs = this.absolutePathForFile(file);
    if (!abs) {
      return undefined;
    }
    const parsed = path.parse(path.resolve(abs));
    const prefix = hidden ? `.${parsed.name}` : parsed.name;
    return path.join(parsed.dir, `${prefix}${extension}`);
  }

  private sidecarStateCandidatePathsForFile(file: TFile): string[] {
    // Preferred-first lookup order:
    // 1) hidden .<name>.binoculars (current format)
    // 2) legacy visible <name>.binoculars
    // 3) legacy visible <name>.json
    const candidates = [
      this.sidecarStatePathForFile(file, SIDECAR_FILE_EXT, true),
      this.sidecarStatePathForFile(file, SIDECAR_FILE_EXT, false),
      this.sidecarStatePathForFile(file, LEGACY_SIDECAR_FILE_EXT, false),
    ].filter((value): value is string => Boolean(value));
    if (candidates.length === 0) {
      return [];
    }
    const deduped: string[] = [];
    const seen = new Set<string>();
    for (const candidate of candidates) {
      const key = process.platform === 'win32' ? path.resolve(candidate).toLowerCase() : path.resolve(candidate);
      if (seen.has(key)) {
        continue;
      }
      seen.add(key);
      deduped.push(candidate);
    }
    return deduped;
  }

  private vaultRelativePathForAbsolutePath(absPath: string): string | undefined {
    const base = this.vaultBasePath();
    if (!base) {
      return undefined;
    }
    const resolvedBase = path.resolve(base);
    const resolvedAbs = path.resolve(absPath);
    const rel = path.relative(resolvedBase, resolvedAbs);
    if (!rel || rel === '.' || rel.startsWith('..') || path.isAbsolute(rel)) {
      return undefined;
    }
    return rel.split(path.sep).join('/');
  }

  private sidecarFilesForMarkdownFile(file: TFile): TFile[] {
    // Convert on-disk candidate sidecar absolute paths back into vault-relative
    // paths so Obsidian can operate on file objects (trash, metadata, events).
    const files: TFile[] = [];
    const seen = new Set<string>();
    for (const sidecarAbsPath of this.sidecarStateCandidatePathsForFile(file)) {
      const sidecarVaultPath = this.vaultRelativePathForAbsolutePath(sidecarAbsPath);
      if (!sidecarVaultPath) {
        continue;
      }
      const abstractFile = this.app.vault.getAbstractFileByPath(sidecarVaultPath);
      if (!(abstractFile instanceof TFile)) {
        continue;
      }
      if (seen.has(abstractFile.path)) {
        continue;
      }
      seen.add(abstractFile.path);
      files.push(abstractFile);
    }
    return files;
  }

  private clearDocumentRuntimeState(docKey: string): void {
    // Best-effort cleanup for all per-document caches/timers to prevent stale
    // overlays and timer callbacks after a note has been removed.
    this.clearLiveEstimateTimer(docKey);
    this.docStates.delete(docKey);
    this.loadedSidecarSignatures.delete(docKey);
    this.docVersions.delete(docKey);
    this.liveEstimateEpochs.delete(docKey);
    this.liveEstimateRecoverAttempts.delete(docKey);
    this.recentTypingActivity.delete(docKey);
    this.hoverSeenSegments.delete(docKey);
    this.lastDecorationSummaryByDoc.delete(docKey);
  }

  private async handleVaultDelete(file: TAbstractFile): Promise<void> {
    try {
      const filePath = file?.path ?? '<unknown>';
      const isMarkdown = file instanceof TFile ? this.isMarkdownSidecarEligible(file) : false;
      this.logStatus('event.vault-delete', { file: filePath, isMarkdown });

      if (!(file instanceof TFile)) {
        return;
      }

      // Remove in-memory state regardless of file type so status/decorations do
      // not keep references to deleted documents.
      this.clearDocumentRuntimeState(file.path);
      this.refreshDecorationsForFile(file.path);
      this.updateStatusForActiveEditor();
      this.refreshControlsView();

      if (!isMarkdown) {
        return;
      }

      const sidecars = this.sidecarFilesForMarkdownFile(file);
      if (sidecars.length === 0) {
        this.logStatus('sidecar.delete.none', { file: file.path });
        return;
      }

      for (const sidecar of sidecars) {
        try {
          // Respect user trash settings (system trash vs Obsidian local trash).
          await this.app.fileManager.trashFile(sidecar);
          this.logStatus('sidecar.delete.trashed', {
            file: file.path,
            sidecar: sidecar.path,
          });
        } catch (err) {
          this.logStatus('sidecar.delete.failed', {
            file: file.path,
            sidecar: sidecar.path,
            error: formatErrorForLog(err),
          });
        }
      }
    } catch (err) {
      this.logStatus('event.vault-delete.error', {
        error: formatErrorForLog(err),
      });
    }
  }

  private isMarkdownSidecarEligible(file: TFile): boolean {
    return path.extname(file.path).toLowerCase() === '.md';
  }

  private maybeLoadStateSidecar(file: TFile, textSnapshot: string, reason: 'open' | 'activate'): void {
    if (!this.isMarkdownSidecarEligible(file)) {
      return;
    }
    const key = file.path;
    const textHash = sha256Hex(textSnapshot);
    const sidecarCandidates = this.sidecarStateCandidatePathsForFile(file);
    let sidecarPath = sidecarCandidates[0];
    for (const candidatePath of sidecarCandidates) {
      if (fs.existsSync(candidatePath)) {
        sidecarPath = candidatePath;
        break;
      }
    }
    if (!sidecarPath || !fs.existsSync(sidecarPath)) {
      return;
    }

    let sidecarRaw: string;
    try {
      sidecarRaw = fs.readFileSync(sidecarPath, 'utf8');
    } catch {
      this.logStatus('sidecar.read.failed', { reason, sidecarPath, file: key });
      return;
    }

    // Signature guards against repeated parse/apply for unchanged text+sidecar.
    const signature = `${textHash}:${sha256Hex(sidecarRaw)}`;
    if (this.loadedSidecarSignatures.get(key) === signature) {
      return;
    }

    let payloadUnknown: unknown;
    try {
      payloadUnknown = JSON.parse(sidecarRaw);
    } catch {
      this.logStatus('sidecar.parse.failed', { reason, sidecarPath, file: key });
      this.loadedSidecarSignatures.set(key, signature);
      return;
    }

    const payload = asRecord(payloadUnknown);
    if (!payload || payload.binoculars_gui_state !== true) {
      this.logStatus('sidecar.ignored.unrecognized', { reason, sidecarPath, file: key });
      this.loadedSidecarSignatures.set(key, signature);
      return;
    }

    const expectedHash = String(payload.text_sha256 ?? '').trim();
    if (expectedHash && expectedHash !== textHash) {
      this.logStatus('sidecar.ignored.hash-mismatch', { reason, sidecarPath, file: key });
      this.loadedSidecarSignatures.set(key, signature);
      return;
    }

    const rawState = asRecord(payload.state);
    if (!rawState) {
      this.logStatus('sidecar.ignored.missing-state', { reason, sidecarPath, file: key });
      this.loadedSidecarSignatures.set(key, signature);
      return;
    }

    const persistedState = documentStateFromPersisted(rawState, textSnapshot.length);
    this.loadedSidecarSignatures.set(key, signature);
    if (!persistedState) {
      this.logStatus('sidecar.ignored.invalid-state', { reason, sidecarPath, file: key });
      return;
    }

    this.docStates.set(key, persistedState);
    this.refreshDecorationsForFile(key);
    this.updateStatusForActiveEditor();
    this.refreshControlsView();
    this.logStatus('sidecar.loaded', { reason, sidecarPath, file: key });
  }

  private autoSaveStateSidecar(file: TFile, textSnapshot: string): void {
    if (!this.isMarkdownSidecarEligible(file)) {
      return;
    }
    const key = file.path;
    const state = this.docStates.get(key);
    const sidecarPath = this.sidecarStatePathForFile(file, SIDECAR_FILE_EXT);
    const absPath = this.absolutePathForFile(file);
    if (!sidecarPath || !absPath) {
      return;
    }
    // Persist even when analysis is absent so metadata captures "clean" state.
    const payload = buildPersistedSidecarPayload(absPath, textSnapshot, state);
    const serialized = JSON.stringify(payload, null, 2);
    try {
      fs.writeFileSync(sidecarPath, serialized, { encoding: 'utf8' });
      this.loadedSidecarSignatures.set(key, `${sha256Hex(textSnapshot)}:${sha256Hex(serialized)}`);
    } catch (err) {
      const message = `State sidecar save failed: ${(err as Error).message}`;
      this.logStatus('sidecar.save.failed', { message, sidecarPath, file: key });
      this.updateStatus(message);
    }
  }

  private buildEditorExtension(): Extension {
    if (!this.isCmRuntimeReady()) {
      this.logStatus('editor-integration.build.skipped', {
        loadError: cmRuntimeLoadError,
      });
      return [];
    }
    const cmView = CM_VIEW as NonNullable<typeof CM_VIEW>;
    const refreshEffect = refreshDecorationsEffect as NonNullable<typeof refreshDecorationsEffect>;
    const refreshField = refreshDecorationsField as NonNullable<typeof refreshDecorationsField>;

    const plugin = this;

    const decorationsPlugin = cmView.ViewPlugin.fromClass(
      class {
        public decorations: DecorationSet;

        constructor(private readonly view: EditorView) {
          try {
            this.decorations = plugin.buildDecorationsForView(this.view);
          } catch (err) {
            plugin.logStatus('cm.decorations.constructor.error', {
              error: formatErrorForLog(err),
            });
            this.decorations = cmView.Decoration.none;
          }
        }

        update(update: ViewUpdate): void {
          try {
            plugin.handleEditorViewUpdate(this.view, update);
            const refreshTriggered = update.transactions.some((tr) =>
              tr.effects.some((e) => e.is(refreshEffect)),
            );
            if (update.docChanged || update.selectionSet || update.viewportChanged || refreshTriggered) {
              this.decorations = plugin.buildDecorationsForView(this.view);
            }
          } catch (err) {
            plugin.logStatus('cm.decorations.update.error', {
              error: formatErrorForLog(err),
            });
            this.decorations = cmView.Decoration.none;
          }
        }
      },
      {
        decorations: (v) => v.decorations,
      },
    );

    class ContributionMarker extends cmView.GutterMarker {
      constructor(
        private readonly sign: 'red' | 'green',
        private readonly widthPx: number,
      ) {
        super();
      }

      override eq(other: ContributionMarker): boolean {
        return this.sign === other.sign && this.widthPx === other.widthPx;
      }

      override toDOM(): HTMLElement {
        const el = document.createElement('span');
        el.className = 'binoculars-gutter-bar';
        const safeWidth = Math.max(2, Math.round(Number.isFinite(this.widthPx) ? this.widthPx : 2));
        const barColor = this.sign === 'red' ? '#ff4d4f' : '#2fbf71';
        el.style.setProperty('--binoculars-bar-width', `${safeWidth}px`);
        el.style.setProperty('--binoculars-bar-color', barColor);
        // Inline fallback styles ensure visibility even when plugin CSS rules are not applied.
        el.style.setProperty('display', 'inline-block', 'important');
        el.style.setProperty('height', '16px', 'important');
        el.style.marginTop = '1px';
        el.style.borderRadius = '1px';
        el.style.setProperty('width', `${Math.max(4, safeWidth)}px`, 'important');
        el.style.setProperty('background-color', barColor, 'important');
        el.style.verticalAlign = 'top';
        return el;
      }
    }

    class ForcedLineNumberMarker extends cmView.GutterMarker {
      constructor(private readonly lineNo: number) {
        super();
      }

      override eq(other: ForcedLineNumberMarker): boolean {
        return this.lineNo === other.lineNo;
      }

      override toDOM(): HTMLElement {
        const el = document.createElement('span');
        el.className = 'binoculars-line-number';
        el.textContent = String(this.lineNo);
        return el;
      }
    }

    const forcedLineNumberGutter = cmView.gutter({
      class: 'binoculars-line-number-gutter',
      lineMarker(view, line) {
        try {
          if (!plugin.shouldRenderForcedLineNumbers(view)) {
            return null;
          }
          const lineNo = view.state.doc.lineAt(line.from).number;
          return new ForcedLineNumberMarker(lineNo);
        } catch (err) {
          plugin.logStatus('cm.line-numbers.lineMarker.error', {
            error: formatErrorForLog(err),
          });
          return null;
        }
      },
      initialSpacer(view) {
        try {
          const lineCount = Math.max(1, view.state.doc.lines);
          return new ForcedLineNumberMarker(lineCount);
        } catch (err) {
          plugin.logStatus('cm.line-numbers.initialSpacer.error', {
            error: formatErrorForLog(err),
          });
          return new ForcedLineNumberMarker(1);
        }
      },
    });

    const contributionGutter = cmView.gutter({
      class: 'binoculars-contribution-gutter',
      lineMarker(view, line) {
        try {
          if (!plugin.isExtensionEnabled()) {
            return null;
          }
          if (!plugin.settings.renderContributionBars) {
            return null;
          }
          const artifacts = plugin.renderArtifactsByView.get(view);
          if (!artifacts) {
            return null;
          }
          const lineNo = view.state.doc.lineAt(line.from).number - 1;
          const info = artifacts.lineContribution.get(lineNo);
          if (!info) {
            return null;
          }
          const level =
            artifacts.maxLineContributionMag > 0
              ? Math.min(
                  GUTTER_LEVELS - 1,
                  Math.round((info.mag / artifacts.maxLineContributionMag) * (GUTTER_LEVELS - 1)),
                )
              : 0;
          const width = Math.max(2, Math.round(((level + 1) / GUTTER_LEVELS) * 18));
          const visibleWidth = Math.max(4, width);
          return new ContributionMarker(info.sign, visibleWidth);
        } catch (err) {
          plugin.logStatus('cm.gutter.lineMarker.error', {
            error: formatErrorForLog(err),
          });
          return null;
        }
      },
    });

    const hover = cmView.hoverTooltip(
      (view, pos): Tooltip | null => {
        try {
          if (!plugin.isExtensionEnabled() || !ENABLE_TEXT_SEGMENT_HOVER) {
            return null;
          }
          const docKey = plugin.docKeyForEditorView(view);
          if (!docKey) {
            return null;
          }
          if (plugin.shouldSuppressHoverDuringTyping(docKey, pos)) {
            return null;
          }

          const lastSeen = plugin.hoverSeenSegments.get(docKey);
          if (lastSeen) {
            if (pos >= lastSeen.start && pos < lastSeen.end) {
              const now = Date.now();
              if (now - lastSeen.lastSeenMs <= HOVER_SAME_SEGMENT_SUPPRESS_MS) {
                return null;
              }
            } else {
              plugin.hoverSeenSegments.delete(docKey);
            }
          }

          const text = view.state.doc.toString();
          const state = plugin.docStates.get(docKey);
          if (!state || state.chunks.length === 0) {
            return null;
          }
          const decision = plugin.hoverForOffset(docKey, text, pos, state);
          if (!decision) {
            return null;
          }

          if (decision.segmentEnd > decision.segmentStart) {
            plugin.hoverSeenSegments.set(docKey, {
              start: decision.segmentStart,
              end: decision.segmentEnd,
              lastSeenMs: Date.now(),
            });
          }

          return {
            pos: decision.rangeStart,
            end: decision.rangeEnd,
            above: false,
            create(): { dom: HTMLElement } {
              const dom = document.createElement('div');
              dom.className = decision.isMinorContributor
                ? 'binoculars-hover-tooltip binoculars-hover-minor'
                : 'binoculars-hover-tooltip';
              dom.innerHTML = decision.html;
              return { dom };
            },
          };
        } catch (err) {
          plugin.logStatus('cm.hover.error', {
            error: formatErrorForLog(err),
          });
          return null;
        }
      },
      {
        hoverTime: HOVER_CONTRIBUTOR_DELAY_MS,
      },
    );

    return [refreshField, decorationsPlugin, forcedLineNumberGutter, contributionGutter, hover];
  }

  private shouldRenderForcedLineNumbers(view: EditorView): boolean {
    if (!this.isExtensionEnabled()) {
      return false;
    }
    const md = this.markdownViewForEditorView(view);
    if (!md || this.markdownModeForView(md) !== 'source') {
      return false;
    }
    const docKey = this.docKeyForEditorView(view);
    if (!docKey) {
      return false;
    }
    const state = this.docStates.get(docKey);
    if (!state || state.chunks.length === 0) {
      return false;
    }
    const preferredCtx = this.activeContextQuiet() ?? this.anyMarkdownContextQuiet();
    if (!preferredCtx || preferredCtx.key !== docKey) {
      return false;
    }
    const host = view.dom;
    if (host.querySelector('.cm-gutters .cm-lineNumbers')) {
      return false;
    }
    return true;
  }

  private handleEditorViewUpdate(view: EditorView, update: ViewUpdate): void {
    if (!this.isExtensionEnabled()) {
      return;
    }
    const docKey = this.docKeyForEditorView(view);
    if (!docKey) {
      return;
    }

    if (update.docChanged) {
      const changes: ContentChange[] = [];
      update.changes.iterChanges((fromA, toA, _fromB, _toB, inserted) => {
        changes.push({
          rangeOffset: fromA,
          rangeLength: toA - fromA,
          text: inserted.toString(),
        });
      });
      if (changes.length > 0) {
        this.noteTypingActivity(docKey, changes, update.state.doc.length);
        this.hoverSeenSegments.delete(docKey);
      }

      const state = this.docStates.get(docKey);
      if (state && changes.length > 0) {
        state.stale = true;
        state.priorChunkB = undefined;
        state.forecastPending = state.chunks.length > 0;
        shiftChunkStateForContentChanges(state, changes, update.state.doc.length);
        state.editedRanges = applyContentChangesToEditedRanges(state.editedRanges, changes);
        state.rewriteRanges = applyContentChangesToRewriteRanges(state.rewriteRanges, changes);
        state.priorLowRanges = applyContentChangesToPriorRanges(state.priorLowRanges, changes);
        state.priorHighRanges = applyContentChangesToPriorRanges(state.priorHighRanges, changes);
      }

      this.docVersions.set(docKey, this.docVersionForKey(docKey) + 1);
      this.scheduleLiveEstimate(docKey);
      this.refreshControlsView();
      this.updateStatusForActiveEditor();
    }

    if (update.selectionSet) {
      this.updateStatusForActiveEditor();
      this.refreshControlsView();
    }
  }

  private buildDecorationsForView(view: EditorView): DecorationSet {
    if (!this.isCmRuntimeReady()) {
      return [] as unknown as DecorationSet;
    }
    const cmView = CM_VIEW as NonNullable<typeof CM_VIEW>;
    const lowMajor = lowMajorDeco as DecorationType;
    const highMajor = highMajorDeco as DecorationType;
    const priorLow = priorLowDeco as DecorationType;
    const priorHigh = priorHighDeco as DecorationType;
    const unscoredMarker = unscoredDeco as DecorationType;
    const edited = editedDeco as DecorationType;
    const hostMarkdownView = this.markdownViewForEditorView(view);
    const hostMode = this.markdownModeForView(hostMarkdownView);
    const docKey = this.docKeyForEditorView(view);
    if (!docKey) {
      this.renderArtifactsByView.set(view, { lineContribution: new Map(), maxLineContributionMag: 0 });
      return cmView.Decoration.none;
    }
    const state = this.docStates.get(docKey);
    const text = view.state.doc.toString();
    if (!this.isExtensionEnabled() || !state || state.chunks.length === 0) {
      this.renderArtifactsByView.set(view, { lineContribution: new Map(), maxLineContributionMag: 0 });
      return cmView.Decoration.none;
    }

    const topK = Math.max(1, Math.trunc(Number.isFinite(this.settings.topK) ? this.settings.topK : 5));
    const enableColorize = this.isTextOverlayColorizationEnabled();

    const lowRanges: EditedRange[] = [];
    const highRanges: EditedRange[] = [];
    const lowMinorRanges: EditedRange[] = [];
    const highMinorRanges: EditedRange[] = [];
    const rowEntries: Array<{
      range: EditedRange;
      delta: number;
    }> = [];

    const lineContribution = new Map<number, { sign: 'red' | 'green'; mag: number; anchorOffset: number }>();

    for (const chunk of state.chunks) {
      for (const row of chunk.rows) {
        const start = Math.max(0, Math.min(text.length, row.char_start));
        const end = Math.max(start, Math.min(text.length, row.char_end));
        if (end <= start) {
          continue;
        }
        const delta = Number(row.delta_doc_logPPL_if_removed ?? 0.0);
        rowEntries.push({ range: { start, end }, delta });

        const startLine = lineNumberFromOffset(text, start) - 1;
        const endLine = lineNumberFromOffset(text, Math.max(start, end - 1)) - 1;
        const mag = Math.abs(delta);
        const sign: 'red' | 'green' = delta >= 0 ? 'red' : 'green';
        for (let line = startLine; line <= endLine; line += 1) {
          const lineStartOffset = offsetAt(text, { line, ch: 0 });
          const anchorOffset = Math.max(start, Math.min(end - 1, Math.min(text.length - 1, lineStartOffset)));
          const prev = lineContribution.get(line);
          if (!prev || mag > prev.mag) {
            lineContribution.set(line, { sign, mag, anchorOffset });
          }
        }
      }
    }

    // Rank contributors globally per document by delta magnitude/sign and keep
    // only top-K LOW and top-K HIGH in major colorized buckets.
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

    const ranges: Array<Range<DecorationType>> = [];
    let queuedCount = 0;
    let unscoredCount = 0;
    let priorLowCount = 0;
    let priorHighCount = 0;
    let editedCount = 0;

    if (enableColorize) {
      const queued: Array<{ start: number; end: number; deco: DecorationType; priority: number }> = [];
      const queue = (range: EditedRange, deco: DecorationType, priority: number): void => {
        if (range.end <= range.start) {
          return;
        }
        queued.push({
          start: range.start,
          end: range.end,
          deco,
          priority,
        });
      };

      for (const r of lowRanges) {
        queue(r, lowMajor, 10);
      }
      for (const r of highRanges) {
        queue(r, highMajor, 11);
      }
      // Minor contributors are intentionally not colorized to match VS Code behavior:
      // only top-K LOW/HIGH contributors get red/green text overlays.

      // Unscored overlay is computed from scored-interval complement.
      const scoredIntervals = mergeIntervals(
        state.chunks
          .map((c) => ({ start: c.charStart, end: c.analyzedCharEnd }))
          .filter((x) => x.end > x.start),
      );
      const unscored = invertIntervals(scoredIntervals, text.length);
      unscoredCount = unscored.length;
      for (const iv of unscored) {
        queue(iv, unscoredMarker, 30);
      }

      const mergedPriorLow = mergeEditedRanges(state.priorLowRanges);
      priorLowCount = mergedPriorLow.length;
      for (const r of mergedPriorLow) {
        queue(r, priorLow, 40);
      }
      const mergedPriorHigh = mergeEditedRanges(state.priorHighRanges);
      priorHighCount = mergedPriorHigh.length;
      for (const r of mergedPriorHigh) {
        queue(r, priorHigh, 41);
      }
      const mergedEdited = mergeEditedRanges(state.editedRanges);
      editedCount = mergedEdited.length;
      for (const r of mergedEdited) {
        queue(r, edited, 50);
      }

      queued.sort((a, b) => {
        if (a.start !== b.start) {
          return a.start - b.start;
        }
        if (a.end !== b.end) {
          return a.end - b.end;
        }
        return a.priority - b.priority;
      });

      for (const q of queued) {
        ranges.push(q.deco.range(q.start, q.end));
      }
      queuedCount = queued.length;
    }

    const mags = [...lineContribution.values()].map((x) => x.mag);
    const maxMag = mags.length > 0 ? Math.max(...mags) : 0;
    this.renderArtifactsByView.set(view, {
      lineContribution,
      maxLineContributionMag: maxMag,
    });

    this.logDecorationSummary(docKey, {
      chunks: state.chunks.length,
      rows: rowEntries.length,
      lowMajor: lowRanges.length,
      highMajor: highRanges.length,
      lowMinor: lowMinorRanges.length,
      highMinor: highMinorRanges.length,
      unscored: unscoredCount,
      priorLow: priorLowCount,
      priorHigh: priorHighCount,
      edited: editedCount,
      queued: queuedCount,
      emitted: ranges.length,
      lineContribution: lineContribution.size,
      colorize: enableColorize,
      bars: Boolean(this.settings.renderContributionBars),
      mode: hostMode,
    });

    return cmView.Decoration.set(ranges, true);
  }

  private hoverForOffset(docKey: string, text: string, charOffset: number, state: DocumentState): HoverDecision | undefined {
    const lineStart = Math.max(0, text.lastIndexOf('\n', Math.max(0, charOffset - 1)) + 1);
    const rawLineEnd = text.indexOf('\n', Math.max(0, charOffset));
    const lineEnd = rawLineEnd >= 0 ? rawLineEnd : text.length;

    const topK = Math.max(1, Math.trunc(Number.isFinite(this.settings.topK) ? this.settings.topK : 5));
    const bestMatch = findBestRowMatch(state, charOffset) ?? findStrongestRowMatchForLine(state, lineStart, lineEnd);
    const majorRows = computeMajorContributorRows(state, topK);
    const isMinorContributor = bestMatch ? !majorRows.has(bestMatch.row) : false;

    const rewriteRange = smallestContainingRange(state.rewriteRanges, charOffset);
    if (rewriteRange) {
      const html = '<strong>Segment rewritten</strong><br/>Select Analyze to determine new score.';
      return {
        html,
        isMinorContributor,
        segmentStart: rewriteRange.start,
        segmentEnd: rewriteRange.end,
        rangeStart: rewriteRange.start,
        rangeEnd: rewriteRange.end,
      };
    }

    const manuallyEditedRange = smallestContainingRange(state.editedRanges, charOffset);
    if (!bestMatch) {
      if (manuallyEditedRange) {
        const html =
          '<em>Note: some text has been manually changed which may impact score. Select Analyze again to obtain accurate statistics.</em>';
        return {
          html,
          isMinorContributor: false,
          segmentStart: manuallyEditedRange.start,
          segmentEnd: manuallyEditedRange.end,
          rangeStart: manuallyEditedRange.start,
          rangeEnd: manuallyEditedRange.end,
        };
      }
      return undefined;
    }

    const delta = Number(bestMatch.row.delta_doc_logPPL_if_removed ?? Number.NaN);
    const rowStart = Math.max(0, Number(bestMatch.row.char_start ?? 0));
    const rowEnd = Math.max(rowStart, Number(bestMatch.row.char_end ?? rowStart));
    const rowHasManualEdits = hasRangeOverlap(state.editedRanges, rowStart, rowEnd);
    const chunkB = Number(bestMatch.chunk.metrics?.binoculars_score ?? Number.NaN);
    const nextChunkB = Number.isFinite(delta) && Number.isFinite(chunkB) ? chunkB + delta : Number.NaN;
    const paragraphLogPPL = Number(bestMatch.row.logPPL ?? Number.NaN);
    const label = Number.isFinite(delta) ? (delta >= 0 ? 'LOW' : 'HIGH') : 'UNKNOWN';

    const lines: string[] = [
      `<strong>Binoculars ${label} Perplexity Segment</strong>`,
      `Delta if removed: <code>${escapeHtml(formatSignedMetric(delta))}</code> (Chunk changes to <code>${escapeHtml(formatSignedMetric(nextChunkB))}</code>)`,
      `Paragraph LogPPL: <code>${escapeHtml(formatStatusMetric(paragraphLogPPL))}</code>`,
    ];
    if (manuallyEditedRange || rowHasManualEdits) {
      lines.push(
        '<em>Note: some text has been manually changed which may impact score. Select Analyze again to obtain accurate statistics.</em>',
      );
    }

    const html = lines.join('<br/>');
    const rangeStart = lineStart;
    const rangeEnd = lineEnd > lineStart ? lineEnd : rowEnd;
    return {
      html,
      isMinorContributor,
      segmentStart: rowStart,
      segmentEnd: rowEnd,
      rangeStart,
      rangeEnd,
    };
  }

  private noteTypingActivity(docKey: string, changes: readonly ContentChange[], docLen: number): void {
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
    this.recentTypingActivity.set(docKey, {
      atMs: Date.now(),
      start,
      end: Math.max(start, end),
    });
  }

  private shouldSuppressHoverDuringTyping(docKey: string, charOffset: number): boolean {
    const activity = this.recentTypingActivity.get(docKey);
    if (!activity) {
      return false;
    }
    const ageMs = Date.now() - activity.atMs;
    if (ageMs > HOVER_TYPING_SUPPRESS_MS) {
      this.recentTypingActivity.delete(docKey);
      return false;
    }
    const padding = 120;
    return charOffset >= Math.max(0, activity.start - padding) && charOffset <= activity.end + padding;
  }

  private scheduleLiveEstimate(key: string): void {
    if (!this.isExtensionEnabled() || this.foregroundBusyOperationCount > 0) {
      return;
    }

    const state = this.docStates.get(key);
    if (!state || !state.stale || state.chunks.length === 0) {
      return;
    }

    this.clearLiveEstimateTimer(key);
    const nextEpoch = (this.liveEstimateEpochs.get(key) ?? 0) + 1;
    this.liveEstimateEpochs.set(key, nextEpoch);
    state.forecastPending = true;
    this.updateStatusForActiveEditor();

    const timer = window.setTimeout(() => {
      void this.computeLiveEstimate(key, nextEpoch);
    }, LIVE_ESTIMATE_DEBOUNCE_MS);
    this.liveEstimateTimers.set(key, timer);
  }

  private clearLiveEstimateTimer(key: string): void {
    const t = this.liveEstimateTimers.get(key);
    if (typeof t === 'number') {
      window.clearTimeout(t);
      this.liveEstimateTimers.delete(key);
    }
  }

  private async computeLiveEstimate(key: string, epoch: number): Promise<void> {
    if (!this.isExtensionEnabled() || this.foregroundBusyOperationCount > 0) {
      return;
    }
    if (this.liveEstimateEpochs.get(key) !== epoch) {
      return;
    }

    this.clearLiveEstimateTimer(key);
    const ctx = this.activeContext();
    const state = this.docStates.get(key);
    if (!ctx || ctx.key !== key || !state || !state.stale || state.chunks.length === 0) {
      return;
    }

    const fullText = ctx.text;
    const cursorOffset = ctx.cursorOffset;
    const orderedChunks = [...state.chunks].sort((a, b) => a.charStart - b.charStart);
    const cursorChunk = orderedChunks.find((chunk) => cursorOffset >= chunk.charStart && cursorOffset < chunk.analyzedCharEnd);
    const activeChunk = cursorChunk ?? this.getActiveChunk(fullText, cursorOffset, state);
    if (!activeChunk || cursorOffset < activeChunk.charStart || cursorOffset >= activeChunk.analyzedCharEnd) {
      state.forecastPending = false;
      this.updateStatusForActiveEditor();
      return;
    }

    const chunkStart = Math.max(0, activeChunk.charStart);
    const chunkEnd = Math.max(chunkStart, Math.min(fullText.length, activeChunk.analyzedCharEnd));
    const baseCross = Number(activeChunk.metrics?.cross_logXPPL ?? Number.NaN);
    if (!Number.isFinite(baseCross) || baseCross === 0) {
      state.forecastPending = false;
      state.forecastEstimate = undefined;
      this.updateStatusForActiveEditor();
      return;
    }

    const version = this.docVersionForKey(key);
    try {
      const client = await this.ensureBackend();
      const result = await client.estimateLiveB(fullText, this.inputLabel(ctx.file), chunkStart, chunkEnd, baseCross);
      if (this.liveEstimateEpochs.get(key) !== epoch) {
        return;
      }
      const liveState = this.docStates.get(key);
      const liveCtx = this.activeContext();
      if (!liveState || !liveCtx || liveCtx.key !== key) {
        return;
      }
      if (this.docVersionForKey(key) !== version) {
        return;
      }

      const approxB = Number(result.approx_b ?? Number.NaN);
      const observerLogPpl = Number((result as { observer_logPPL?: number }).observer_logPPL ?? Number.NaN);
      const estimatedB =
        Number.isFinite(approxB)
          ? approxB
          : Number.isFinite(observerLogPpl) && Number.isFinite(baseCross) && baseCross !== 0
            ? observerLogPpl / baseCross
            : Number.NaN;

      liveState.forecastPending = false;
      if (Number.isFinite(estimatedB)) {
        liveState.forecastEstimate = {
          chunkStart,
          docVersion: version,
          b: estimatedB,
        };
        this.liveEstimateRecoverAttempts.delete(key);
      } else {
        const attempt = this.liveEstimateRecoverAttempts.get(key) ?? 0;
        this.logStatus('estimate.non-finite', {
          key,
          attempt,
          approx_b: result.approx_b,
          observer_logPPL: (result as { observer_logPPL?: number }).observer_logPPL,
          baseCross,
        });
        if (attempt < 1) {
          this.liveEstimateRecoverAttempts.set(key, attempt + 1);
          await this.stopBackend({ shutdownDaemon: true });
          this.scheduleLiveEstimate(key);
          return;
        }
      }
      this.updateStatusForActiveEditor();
    } catch (err) {
      if (this.liveEstimateEpochs.get(key) !== epoch) {
        return;
      }
      const liveState = this.docStates.get(key);
      if (liveState) {
        liveState.forecastPending = false;
      }
      const attempt = this.liveEstimateRecoverAttempts.get(key) ?? 0;
      if (attempt < 1) {
        this.liveEstimateRecoverAttempts.set(key, attempt + 1);
        try {
          await this.stopBackend({ shutdownDaemon: true });
        } catch {
          // ignore
        }
        this.scheduleLiveEstimate(key);
        return;
      }
      this.updateStatusForActiveEditor();
      this.logStatus('estimate.failed', {
        key,
        error: formatErrorForLog(err),
      });
    }
  }

  private refreshAllDecorations(): void {
    if (!this.settings.editorIntegrationEnabled || !refreshDecorationsEffect) {
      return;
    }
    const refreshEffect = refreshDecorationsEffect;
    for (const md of this.markdownViews()) {
      const cm = this.editorViewForMarkdownView(md);
      if (!cm) {
        continue;
      }
      try {
        cm.dispatch({ effects: refreshEffect.of(null) });
      } catch (err) {
        this.logStatus('cm.refresh-all.dispatch.error', {
          file: md.file?.path ?? '',
          error: formatErrorForLog(err),
        });
      }
    }
  }

  private refreshDecorationsForFile(docKey: string): void {
    if (!this.settings.editorIntegrationEnabled || !refreshDecorationsEffect) {
      return;
    }
    const refreshEffect = refreshDecorationsEffect;
    for (const md of this.markdownViews()) {
      if (!md.file || md.file.path !== docKey) {
        continue;
      }
      const cm = this.editorViewForMarkdownView(md);
      if (!cm) {
        continue;
      }
      try {
        cm.dispatch({ effects: refreshEffect.of(null) });
      } catch (err) {
        this.logStatus('cm.refresh-file.dispatch.error', {
          file: docKey,
          error: formatErrorForLog(err),
        });
      }
    }
  }

  private markdownViews(): MarkdownView[] {
    return this.app.workspace
      .getLeavesOfType('markdown')
      .map((leaf) => leaf.view)
      .filter((v): v is MarkdownView => v instanceof MarkdownView);
  }

  private markdownLeavesWithFiles(): Array<{ leaf: WorkspaceLeaf; view: MarkdownView }> {
    return this.app.workspace
      .getLeavesOfType('markdown')
      .map((leaf) => {
        if (leaf.view instanceof MarkdownView && leaf.view.file instanceof TFile) {
          return { leaf, view: leaf.view };
        }
        return undefined;
      })
      .filter((entry): entry is { leaf: WorkspaceLeaf; view: MarkdownView } => Boolean(entry));
  }

  private preferredMarkdownLeaf(): WorkspaceLeaf | undefined {
    const leaves = this.markdownLeavesWithFiles();
    if (leaves.length === 0) {
      return undefined;
    }

    const byPath = (targetPath: string | undefined): WorkspaceLeaf | undefined => {
      if (!targetPath) {
        return undefined;
      }
      return leaves.find((entry) => entry.view.file?.path === targetPath)?.leaf;
    };

    const activeMarkdown = this.app.workspace.getActiveViewOfType(MarkdownView);
    if (activeMarkdown?.file?.path) {
      const activeLeaf = byPath(activeMarkdown.file.path);
      if (activeLeaf) {
        return activeLeaf;
      }
    }

    const activeFilePath = this.app.workspace.getActiveFile()?.path;
    const byActiveFile = byPath(activeFilePath);
    if (byActiveFile) {
      return byActiveFile;
    }

    const byLastActive = byPath(this.lastActiveMarkdownFilePath);
    if (byLastActive) {
      return byLastActive;
    }

    const byLastOpened = byPath(this.lastOpenedMarkdownFilePath);
    if (byLastOpened) {
      return byLastOpened;
    }

    const mostRecentLeaf = this.app.workspace.getMostRecentLeaf();
    if (mostRecentLeaf?.view instanceof MarkdownView && mostRecentLeaf.view.file instanceof TFile) {
      return mostRecentLeaf;
    }

    const withCmEditor = leaves.find((entry) => Boolean(this.editorViewForMarkdownView(entry.view)));
    if (withCmEditor) {
      return withCmEditor.leaf;
    }

    return leaves[0].leaf;
  }

  private editorViewForMarkdownView(view: MarkdownView): EditorView | undefined {
    const editor = this.editorForMarkdownView(view) as unknown as { cm?: EditorView } | undefined;
    return editor?.cm;
  }

  private markdownViewForEditorView(view: EditorView): MarkdownView | undefined {
    for (const md of this.markdownViews()) {
      const cm = this.editorViewForMarkdownView(md);
      if (cm === view) {
        return md;
      }
    }
    return undefined;
  }

  private docKeyForEditorView(view: EditorView): string | undefined {
    const md = this.markdownViewForEditorView(view);
    if (md?.file) {
      return md.file.path;
    }
    const now = Date.now();
    if (now - this.lastUnmappedCmViewLogAtMs > 1500) {
      this.lastUnmappedCmViewLogAtMs = now;
      this.logStatus('cm.view.unmapped');
    }
    return undefined;
  }

  private currentTextForKey(key: string): string | undefined {
    for (const md of this.markdownViews()) {
      if (md.file?.path === key) {
        const editor = this.editorForMarkdownView(md);
        if (editor) {
          return editor.getValue();
        }
      }
    }
    return undefined;
  }

  private docVersionForKey(key: string): number {
    return this.docVersions.get(key) ?? 0;
  }

  private activeContext(): EditorContext | undefined {
    const view = this.app.workspace.getActiveViewOfType(MarkdownView);
    if (!view) {
      this.logMissingContext('no-active-markdown-view');
      return undefined;
    }
    return this.contextFromMarkdownView(view, 'active');
  }

  private contextFromMarkdownView(view: MarkdownView, source: 'active' | 'fallback'): EditorContext | undefined {
    const file = view.file;
    if (!file) {
      this.logMissingContext('markdown-view-missing-file', { source });
      return undefined;
    }
    const editor = this.editorForMarkdownView(view);
    if (!editor) {
      this.logMissingContext('markdown-view-missing-editor', { file: file.path, source });
      return undefined;
    }
    const text = editor.getValue();
    const cursor = editor.getCursor();
    const from = editor.getCursor('from');
    const to = editor.getCursor('to');

    return {
      view,
      file,
      editor,
      key: file.path,
      text,
      cursorOffset: offsetAt(text, cursor),
      cursorLine: cursor.line,
      selectionStart: offsetAt(text, from),
      selectionEnd: offsetAt(text, to),
    };
  }

  private anyMarkdownContext(): EditorContext | undefined {
    const preferredLeaf = this.preferredMarkdownLeaf();
    if (preferredLeaf?.view instanceof MarkdownView) {
      const preferredContext = this.contextFromMarkdownView(preferredLeaf.view, 'fallback');
      if (preferredContext) {
        return preferredContext;
      }
    }
    for (const md of this.markdownViews()) {
      const ctx = this.contextFromMarkdownView(md, 'fallback');
      if (ctx) {
        return ctx;
      }
    }
    return undefined;
  }

  private preferredContextForCommand(action: string): EditorContext | undefined {
    const active = this.activeContext();
    if (active) {
      return active;
    }
    const fallback = this.anyMarkdownContext();
    if (fallback) {
      this.logStatus('command.context.fallback-used', {
        action,
        file: fallback.file.path,
      });
      return fallback;
    }
    this.showError(`${action} requires an open markdown editor.`);
    return undefined;
  }

  private async ensureSourceModeForOverlays(ctx: EditorContext, action: string): Promise<EditorContext> {
    if (!this.settings.editorIntegrationEnabled) {
      return ctx;
    }
    const modeBefore = this.markdownModeForView(ctx.view);
    if (modeBefore !== 'preview') {
      return ctx;
    }

    this.logStatus('editor.mode.ensure-source.begin', {
      action,
      file: ctx.file.path,
      modeBefore,
    });
    this.updateStatus('Switching note to Source mode so overlays can render...');

    try {
      const leaf = ctx.view.leaf;
      const currentState = leaf.getViewState();
      if (currentState?.type === 'markdown') {
        const nextState = {
          ...currentState,
          state: {
            ...(currentState.state ?? {}),
            mode: 'source',
          },
          active: true,
        };
        await leaf.setViewState(nextState);
        await this.app.workspace.setActiveLeaf(leaf, true, true);
      } else {
        await this.app.workspace.setActiveLeaf(leaf, true, true);
      }
    } catch (err) {
      this.logStatus('editor.mode.ensure-source.error', {
        action,
        file: ctx.file.path,
        error: formatErrorForLog(err),
      });
    }

    const refreshedView = ctx.view.leaf?.view instanceof MarkdownView ? ctx.view.leaf.view : ctx.view;
    const refreshed = this.contextFromMarkdownView(refreshedView, 'fallback') ?? ctx;
    const modeAfter = this.markdownModeForView(refreshed.view);
    this.logStatus('editor.mode.ensure-source.end', {
      action,
      file: refreshed.file.path,
      modeBefore,
      modeAfter,
    });
    if (modeAfter === 'preview') {
      this.updateStatus(
        'Overlays are hidden in Reading view. Switch to Live Preview or Source mode to see colorization and gutter bars.',
      );
    }
    return refreshed;
  }

  private logMissingContext(reason: string, payload?: Record<string, unknown>): void {
    const now = Date.now();
    if (now - this.lastMissingContextLogAtMs < 1500) {
      return;
    }
    this.lastMissingContextLogAtMs = now;
    this.logStatus('context.missing', {
      reason,
      ...(payload ?? {}),
    });
  }

  private editorForMarkdownView(view: MarkdownView): EditorLike | undefined {
    const maybeEditor = (view as unknown as { editor?: unknown }).editor;
    if (!maybeEditor || typeof maybeEditor !== 'object') {
      this.logStatus('editor.missing-object', {
        file: view.file?.path ?? '',
      });
      return undefined;
    }
    const candidate = maybeEditor as Partial<EditorLike>;
    if (
      typeof candidate.getValue !== 'function' ||
      typeof candidate.getCursor !== 'function' ||
      typeof candidate.replaceRange !== 'function' ||
      typeof candidate.setCursor !== 'function'
    ) {
      this.logStatus('editor.missing-methods', {
        file: view.file?.path ?? '',
        hasGetValue: typeof candidate.getValue === 'function',
        hasGetCursor: typeof candidate.getCursor === 'function',
        hasReplaceRange: typeof candidate.replaceRange === 'function',
        hasSetCursor: typeof candidate.setCursor === 'function',
      });
      return undefined;
    }
    return candidate as EditorLike;
  }

  private hasRenderableEditorForFile(docKey: string): boolean {
    for (const md of this.markdownViews()) {
      if (md.file?.path !== docKey) {
        continue;
      }
      if (this.markdownModeForView(md) !== 'source') {
        continue;
      }
      if (this.editorViewForMarkdownView(md)) {
        return true;
      }
    }
    return false;
  }

  private maybeNotifyReadingViewOverlayHidden(docKey: string): void {
    if (!this.isExtensionEnabled() || !this.settings.editorIntegrationEnabled) {
      return;
    }
    if (this.hasRenderableEditorForFile(docKey)) {
      return;
    }
    const modes = [...new Set(this.markdownViews().filter((md) => md.file?.path === docKey).map((md) => this.markdownModeForView(md)))];
    this.logStatus('editor-integration.no-visible-cm-editor', {
      file: docKey,
      modes,
    });
    this.updateStatus(
      'Analyze completed. Overlays are hidden in Reading view. Open the note in Live Preview or Source mode to see colorization and gutter bars.',
    );
  }

  private getActiveChunk(text: string, cursorOffset: number, state: DocumentState): ChunkState | undefined {
    if (state.chunks.length === 0) {
      return undefined;
    }

    const containing = state.chunks.find((c) => cursorOffset >= c.charStart && cursorOffset < c.analyzedCharEnd);
    if (containing) {
      return containing;
    }

    const distanceToChunk = (chunk: ChunkState): number => {
      if (cursorOffset < chunk.charStart) {
        return chunk.charStart - cursorOffset;
      }
      if (cursorOffset > chunk.analyzedCharEnd) {
        return cursorOffset - chunk.analyzedCharEnd;
      }
      return 0;
    };

    const sorted = [...state.chunks].sort((a, b) => {
      const da = distanceToChunk(a);
      const db = distanceToChunk(b);
      if (da !== db) {
        return da - db;
      }
      return a.charStart - b.charStart;
    });
    return sorted[0];
  }

  private resolveRewriteSpan(
    docText: string,
    selectionStart: number,
    selectionEnd: number,
    cursorOffset: number,
    state: DocumentState,
  ): { start: number; end: number } | undefined {
    if (selectionEnd > selectionStart) {
      return {
        start: Math.min(selectionStart, selectionEnd),
        end: Math.max(selectionStart, selectionEnd),
      };
    }

    const active = this.getActiveChunk(docText, cursorOffset, state);
    if (!active) {
      return undefined;
    }

    const row = active.rows
      .filter((r) => Number(r.delta_doc_logPPL_if_removed ?? 0) >= 0)
      .find((r) => cursorOffset >= r.char_start && cursorOffset <= r.char_end);
    if (!row) {
      return undefined;
    }
    return { start: row.char_start, end: row.char_end };
  }

  private resolveSelectionOrLineSpan(
    docText: string,
    selectionStart: number,
    selectionEnd: number,
    cursorLine: number,
  ): { start: number; end: number } | undefined {
    if (selectionEnd > selectionStart) {
      return { start: Math.min(selectionStart, selectionEnd), end: Math.max(selectionStart, selectionEnd) };
    }
    const lineStart = offsetAt(docText, { line: cursorLine, ch: 0 });
    const nextLineStart = offsetAt(docText, { line: cursorLine + 1, ch: 0 });
    const end = cursorLine + 1 < lineCount(docText) ? Math.max(lineStart, nextLineStart - 1) : docText.length;
    if (end <= lineStart) {
      return undefined;
    }
    return {
      start: lineStart,
      end,
    };
  }

  private isTextOverlayColorizationEnabled(): boolean {
    if (!this.isExtensionEnabled()) {
      return false;
    }
    return this.settings.renderColorizeText && this.runtimeColorizationEnabled;
  }

  private async runWithBusyNotice<T>(
    message: string,
    action: () => Promise<T>,
    opts?: { docKey?: string; kind?: 'analysis' | 'rewrite' | 'other' },
  ): Promise<T> {
    // Busy counters are used by controls view to block analysis actions when a
    // different note already has an in-flight analysis request.
    this.updateStatus(message);
    const nextBusyCount = this.foregroundBusyOperationCount + 1;
    this.foregroundBusyOperationCount = nextBusyCount;
    if (nextBusyCount === 1) {
      this.foregroundBusyDocKey = opts?.docKey;
      this.foregroundBusyKind = opts?.kind ?? 'other';
    }
    try {
      return await action();
    } finally {
      this.foregroundBusyOperationCount = Math.max(0, this.foregroundBusyOperationCount - 1);
      if (this.foregroundBusyOperationCount <= 0) {
        this.foregroundBusyDocKey = undefined;
        this.foregroundBusyKind = undefined;
      }
    }
  }

  public async loadPluginSettings(): Promise<void> {
    const loaded = (await this.loadData()) as Partial<BinocularsSettings> | null;
    this.settings = {
      ...DEFAULT_SETTINGS,
      ...(loaded ?? {}),
    };
  }

  public async savePluginSettings(): Promise<void> {
    await this.saveData(this.settings);
    if (this.isExtensionEnabled()) {
      await this.restartBackend();
      this.refreshAllDecorations();
      this.updateStatusForActiveEditor();
      this.refreshControlsView();
    }
  }
}

class BinocularsSettingTab extends PluginSettingTab {
  private plugin: BinocularsPlugin;

  constructor(app: App, plugin: BinocularsPlugin) {
    super(app, plugin);
    this.plugin = plugin;
  }

  display(): void {
    const { containerEl } = this;
    containerEl.empty();

    containerEl.createEl('h2', { text: 'Binoculars' });

    const settings = this.plugin.getSettings();

    new Setting(containerEl)
      .setName('Enabled')
      .setDesc('Enable/disable Binoculars commands and backend globally.')
      .addToggle((toggle) =>
        toggle.setValue(settings.enabled).onChange(async (value) => {
          settings.enabled = value;
          await this.plugin.savePluginSettings();
        }),
      );

    new Setting(containerEl)
      .setName('Editor integration (CM6 overlays/hover/gutter)')
      .setDesc('Enable CodeMirror-based colorization, hover diagnostics, and contribution bars. Disable for open-note stability diagnostics. Requires plugin reload to take effect.')
      .addToggle((toggle) =>
        toggle.setValue(settings.editorIntegrationEnabled).onChange(async (value) => {
          settings.editorIntegrationEnabled = value;
          this.plugin.resetVisualsDisabledHint();
          await this.plugin.savePluginSettings();
          this.plugin.postStatus('Editor integration setting saved. Disable/enable the plugin (or restart Obsidian) to apply.');
        }),
      );

    new Setting(containerEl)
      .setName('Python path')
      .setDesc('Python executable used to launch the persistent bridge backend.')
      .addText((text) =>
        text.setValue(settings.backendPythonPath).onChange(async (value) => {
          settings.backendPythonPath = value.trim();
          await this.plugin.savePluginSettings();
        }),
      );

    new Setting(containerEl)
      .setName('Bridge script path')
      .setDesc('Path to binoculars_bridge.py. Supports ${vaultRoot} and ${workspaceFolder}.')
      .addText((text) =>
        text.setValue(settings.backendBridgeScriptPath).onChange(async (value) => {
          settings.backendBridgeScriptPath = value.trim();
          await this.plugin.savePluginSettings();
        }),
      );

    new Setting(containerEl)
      .setName('Config path')
      .setDesc('Path to Binoculars scoring config JSON.')
      .addText((text) =>
        text.setValue(settings.configPath).onChange(async (value) => {
          settings.configPath = value.trim();
          await this.plugin.savePluginSettings();
        }),
      );

    new Setting(containerEl)
      .setName('Top K')
      .setDesc('Top-k size used for hotspot diagnostics.')
      .addSlider((slider) =>
        slider
          .setLimits(1, 30, 1)
          .setValue(settings.topK)
          .setDynamicTooltip()
          .onChange(async (value) => {
            settings.topK = value;
            await this.plugin.savePluginSettings();
          }),
      );

    new Setting(containerEl)
      .setName('Text max tokens override')
      .setDesc('Optional override for text.max_tokens. Leave blank for profile default.')
      .addText((text) =>
        text
          .setPlaceholder('null')
          .setValue(settings.textMaxTokensOverride === null ? '' : String(settings.textMaxTokensOverride))
          .onChange(async (value) => {
            const v = value.trim();
            if (!v) {
              settings.textMaxTokensOverride = null;
            } else {
              const n = Number(v);
              settings.textMaxTokensOverride = Number.isFinite(n) ? Math.max(0, Math.trunc(n)) : null;
            }
            await this.plugin.savePluginSettings();
          }),
      );

    new Setting(containerEl)
      .setName('Observer GGUF path')
      .setDesc('Optional observer model path override.')
      .addText((text) =>
        text.setValue(settings.observerGgufPath).onChange(async (value) => {
          settings.observerGgufPath = value.trim();
          await this.plugin.savePluginSettings();
        }),
      );

    new Setting(containerEl)
      .setName('Performer GGUF path')
      .setDesc('Optional performer model path override.')
      .addText((text) =>
        text.setValue(settings.performerGgufPath).onChange(async (value) => {
          settings.performerGgufPath = value.trim();
          await this.plugin.savePluginSettings();
        }),
      );

    new Setting(containerEl)
      .setName('External rewrite LLM enabled')
      .setDesc('Whether rewrite generation may use external OpenAI-compatible LLM when configured.')
      .addToggle((toggle) =>
        toggle.setValue(settings.externalLlmEnabled).onChange(async (value) => {
          settings.externalLlmEnabled = value;
          await this.plugin.savePluginSettings();
        }),
      );

    new Setting(containerEl)
      .setName('External rewrite LLM config path')
      .setDesc('Optional explicit path to rewrite LLM config JSON.')
      .addText((text) =>
        text.setValue(settings.externalLlmConfigPath).onChange(async (value) => {
          settings.externalLlmConfigPath = value.trim();
          await this.plugin.savePluginSettings();
        }),
      );

    new Setting(containerEl)
      .setName('External rewrite endpoint')
      .setDesc('Optional endpoint URL for OpenAI-compatible rewrite model.')
      .addText((text) =>
        text.setValue(settings.externalLlmEndpoint).onChange(async (value) => {
          settings.externalLlmEndpoint = value.trim();
          await this.plugin.savePluginSettings();
        }),
      );

    new Setting(containerEl)
      .setName('External rewrite model')
      .setDesc('Optional model name for external rewrite LLM.')
      .addText((text) =>
        text.setValue(settings.externalLlmModel).onChange(async (value) => {
          settings.externalLlmModel = value.trim();
          await this.plugin.savePluginSettings();
        }),
      );

    new Setting(containerEl)
      .setName('External rewrite temperature')
      .setDesc('Optional rewrite temperature override.')
      .addSlider((slider) =>
        slider
          .setLimits(0, 2, 0.1)
          .setValue(settings.externalLlmTemperature)
          .setDynamicTooltip()
          .onChange(async (value) => {
            settings.externalLlmTemperature = value;
            await this.plugin.savePluginSettings();
          }),
      );

    new Setting(containerEl)
      .setName('External rewrite max tokens')
      .setDesc('Optional external rewrite max_tokens override.')
      .addText((text) =>
        text.setValue(String(settings.externalLlmMaxTokens)).onChange(async (value) => {
          const n = Number(value.trim());
          if (Number.isFinite(n) && n > 0) {
            settings.externalLlmMaxTokens = Math.max(1, Math.trunc(n));
            await this.plugin.savePluginSettings();
          }
        }),
      );

    new Setting(containerEl)
      .setName('Render contribution bars')
      .setDesc('Render per-line contribution bars in editor gutter.')
      .addToggle((toggle) =>
        toggle.setValue(settings.renderContributionBars).onChange(async (value) => {
          settings.renderContributionBars = value;
          await this.plugin.savePluginSettings();
        }),
      );

    new Setting(containerEl)
      .setName('Render colorized text')
      .setDesc('Render LOW/HIGH contribution colorization overlays in editor text.')
      .addToggle((toggle) =>
        toggle.setValue(settings.renderColorizeText).onChange(async (value) => {
          settings.renderColorizeText = value;
          await this.plugin.savePluginSettings();
        }),
      );

    new Setting(containerEl)
      .setName('Diagnostic logging')
      .setDesc('Write detailed status logs to the plugin status.log file for troubleshooting.')
      .addToggle((toggle) =>
        toggle.setValue(settings.diagnosticLogging).onChange(async (value) => {
          settings.diagnosticLogging = value;
          await this.plugin.savePluginSettings();
        }),
      );
  }
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

  const parseRanges = (raw: unknown): EditedRange[] =>
    (Array.isArray(raw) ? raw : [])
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
    editedRanges: parseRanges(rawState.edited_ranges),
    rewriteRanges: parseRanges(rawState.rewrite_ranges),
    priorLowRanges: parseRanges(rawState.prior_low_ranges),
    priorHighRanges: parseRanges(rawState.prior_high_ranges),
    priorChunkB: undefined,
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

function buildPersistedSidecarPayload(docPath: string, textSnapshot: string, state: DocumentState | undefined): PersistedSidecarPayload {
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
    edited_ranges: (state?.editedRanges ?? []).map((r) => ({ start: r.start, end: r.end })),
    rewrite_ranges: (state?.rewriteRanges ?? []).map((r) => ({ start: r.start, end: r.end })),
    prior_low_ranges: (state?.priorLowRanges ?? []).map((r) => ({ start: r.start, end: r.end })),
    prior_high_ranges: (state?.priorHighRanges ?? []).map((r) => ({ start: r.start, end: r.end })),
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

function normalizedRangesWithoutTouchMerge(ranges: EditedRange[]): EditedRange[] {
  return [...ranges]
    .filter((r) => Number.isFinite(r.start) && Number.isFinite(r.end) && r.end > r.start)
    .sort((a, b) => a.start - b.start || a.end - b.end);
}

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

function shiftChunkStateForContentChanges(state: DocumentState, changes: readonly ContentChange[], textLength: number): void {
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
      const remappedAnalyzed = remapSpanForSplice(chunk.charStart, chunk.analyzedCharEnd, spliceStart, spliceEnd, insertedLen);
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

function applyContentChangesToRanges(prevRanges: EditedRange[], changes: readonly ContentChange[], includeInsertedText: boolean): EditedRange[] {
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

function applyContentChangesToEditedRanges(prevRanges: EditedRange[], changes: readonly ContentChange[]): EditedRange[] {
  return applyContentChangesToRanges(prevRanges, changes, true);
}

function applyContentChangesToRewriteRanges(prevRanges: EditedRange[], changes: readonly ContentChange[]): EditedRange[] {
  return applyContentChangesToRanges(prevRanges, changes, false);
}

function applyContentChangesToPriorRanges(prevRanges: EditedRange[], changes: readonly ContentChange[]): EditedRange[] {
  return applyContentChangesToRanges(prevRanges, changes, false);
}

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

function findStrongestRowMatchForLine(
  state: DocumentState,
  lineStart: number,
  lineEnd: number,
): { chunk: ChunkState; row: ParagraphRow } | undefined {
  let bestMatch:
    | {
        chunk: ChunkState;
        row: ParagraphRow;
      }
    | undefined;
  let bestMag = -1;
  for (const chunk of state.chunks) {
    for (const row of chunk.rows) {
      const rowStart = Math.max(0, Number(row.char_start ?? 0));
      const rowEnd = Math.max(rowStart, Number(row.char_end ?? rowStart));
      if (rowEnd <= lineStart || rowStart >= lineEnd) {
        continue;
      }
      const mag = Math.abs(Number(row.delta_doc_logPPL_if_removed ?? 0));
      if (!bestMatch || mag > bestMag) {
        bestMatch = { chunk, row };
        bestMag = mag;
      }
    }
  }
  return bestMatch;
}

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

function sha256Hex(value: string): string {
  return crypto.createHash('sha256').update(value, 'utf8').digest('hex');
}

function formatErrorForLog(err: unknown): Record<string, unknown> | string {
  if (err instanceof BridgeRpcError) {
    return {
      name: err.name,
      message: err.message,
      code: err.code ?? '',
      method: err.method ?? '',
      details: err.details ?? {},
      stack: err.stack ?? '',
    };
  }
  if (err instanceof Error) {
    return {
      name: err.name,
      message: err.message,
      stack: err.stack ?? '',
    };
  }
  if (typeof err === 'string') {
    return err;
  }
  try {
    return JSON.stringify(err);
  } catch {
    return String(err);
  }
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

function lineNumberFromOffset(text: string, offset: number): number {
  const clamped = Math.max(0, Math.min(text.length, Math.trunc(offset)));
  return positionAt(text, clamped).line + 1;
}

function lineCount(text: string): number {
  if (!text) {
    return 1;
  }
  let count = 1;
  for (let i = 0; i < text.length; i += 1) {
    if (text.charCodeAt(i) === 10) {
      count += 1;
    }
  }
  return count;
}

function positionAt(text: string, offset: number): { line: number; ch: number } {
  const bounded = Math.max(0, Math.min(text.length, offset));
  const lines = text.slice(0, bounded).split('\n');
  const line = Math.max(0, lines.length - 1);
  const ch = lines[lines.length - 1]?.length ?? 0;
  return { line, ch };
}

function offsetAt(text: string, pos: { line: number; ch: number }): number {
  const lines = text.split('\n');
  let offset = 0;
  const line = Math.max(0, Math.min(lines.length - 1, pos.line));
  for (let i = 0; i < line; i += 1) {
    offset += lines[i].length + 1;
  }
  return Math.max(0, Math.min(text.length, offset + Math.max(0, pos.ch)));
}
