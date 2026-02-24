import * as cp from 'child_process';
import * as fs from 'fs';
import * as net from 'net';
import * as os from 'os';
import * as path from 'path';
import * as readline from 'readline';
import { AnalyzeResult, BridgeRequest, BridgeResponse, RewriteResult } from './types';

// Per-method ceilings are intentionally generous because model load/analyze/rewrite
// operations may span many seconds on large chunks.
const BRIDGE_TIMEOUT_MS_DEFAULT = 180000;
const BRIDGE_TIMEOUT_MS_INITIALIZE = 600000;
const BRIDGE_TIMEOUT_MS_ANALYZE = 900000;
const BRIDGE_TIMEOUT_MS_REWRITE = 600000;

export class BridgeRpcError extends Error {
  public readonly code?: string;
  public readonly details?: Record<string, unknown>;
  public readonly method?: string;

  constructor(message: string, opts?: { code?: string; details?: Record<string, unknown>; method?: string }) {
    super(message);
    this.name = 'BridgeRpcError';
    this.code = opts?.code;
    this.details = opts?.details;
    this.method = opts?.method;
  }
}

export class BackendClient {
  private socket: net.Socket | undefined;
  private rl: readline.Interface | undefined;
  private requestSeq = 1;
  private pending = new Map<
    number,
    { resolve: (value: unknown) => void; reject: (err: Error) => void; timer: NodeJS.Timeout; method: string }
  >();
  private readonly output: (line: string) => void;
  private readyResolver: (() => void) | undefined;
  private readySeen = false;
  private startPromise: Promise<void> | undefined;
  private socketPath = '';
  private lastInitializeSignature = '';

  constructor(output: (line: string) => void) {
    this.output = output;
  }

  public async start(pythonPath: string, bridgeScriptPath: string, cwd?: string): Promise<void> {
    // Reuse the existing live socket when possible. If a startup is already in
    // progress, callers share the same promise to avoid duplicate daemon spawns.
    if (this.socket && !this.socket.destroyed) {
      return;
    }
    if (this.startPromise) {
      return this.startPromise;
    }
    this.startPromise = this.startInternal(pythonPath, bridgeScriptPath, cwd);
    return this.startPromise;
  }

  public async initialize(
    cfgPath: string,
    topK: number,
    textMaxTokensOverride: number | null,
    observerModelPath: string,
    performerModelPath: string,
    rewriteLlmConfigPath: string,
  ): Promise<void> {
    // Skip redundant initialize RPC calls when effective config/override values
    // have not changed for this live socket session.
    const params = {
      cfg_path: cfgPath,
      top_k: topK,
      text_max_tokens_override: textMaxTokensOverride,
      observer_model_path: observerModelPath,
      performer_model_path: performerModelPath,
      rewrite_llm_config_path: rewriteLlmConfigPath,
    };
    const signature = JSON.stringify(params);
    if (signature === this.lastInitializeSignature && this.socket && !this.socket.destroyed) {
      return;
    }
    await this.request('initialize', params, BRIDGE_TIMEOUT_MS_INITIALIZE);
    this.lastInitializeSignature = signature;
  }

  public analyzeDocument(text: string, inputLabel: string): Promise<AnalyzeResult> {
    return this.request<AnalyzeResult>('analyze_document', {
      text,
      input_label: inputLabel,
    }, BRIDGE_TIMEOUT_MS_ANALYZE);
  }

  public analyzeChunk(text: string, inputLabel: string, startChar: number, endChar: number): Promise<AnalyzeResult> {
    return this.request<AnalyzeResult>('analyze_chunk', {
      text,
      input_label: inputLabel,
      start_char: startChar,
      end_char: endChar,
    }, BRIDGE_TIMEOUT_MS_ANALYZE);
  }

  public analyzeNextChunk(text: string, inputLabel: string, startChar?: number): Promise<AnalyzeResult> {
    return this.request<AnalyzeResult>('analyze_next_chunk', {
      text,
      input_label: inputLabel,
      start_char: startChar,
    }, BRIDGE_TIMEOUT_MS_ANALYZE);
  }

  public estimateLiveB(
    text: string,
    inputLabel: string,
    startChar: number,
    endChar: number,
    baseCrossLogXppl: number,
  ): Promise<{
    ok: boolean;
    approx_b: number;
    observer_logPPL: number;
    transitions: number;
    analyzed_char_end: number;
    truncated_by_limit: boolean;
  }> {
    return this.request(
      'estimate_live_b',
      {
        text,
        input_label: inputLabel,
        start_char: startChar,
        end_char: endChar,
        base_cross_logxppl: baseCrossLogXppl,
      },
      600000,
    );
  }

  public rewriteSpan(
    text: string,
    spanStart: number,
    spanEnd: number,
    baseMetrics: Record<string, number> | undefined,
    optionCount = 3,
  ): Promise<RewriteResult> {
    return this.request<RewriteResult>('rewrite_span', {
      text,
      span_start: spanStart,
      span_end: spanEnd,
      option_count: optionCount,
      base_metrics: baseMetrics,
    }, BRIDGE_TIMEOUT_MS_REWRITE);
  }

  public async shutdown(): Promise<void> {
    if (!this.socket || this.socket.destroyed) {
      this.cleanupConnectionState();
      return;
    }
    try {
      await this.request('shutdown', {}, 2000);
    } catch {
      // ignore
    }
    this.socket.end();
    this.socket.destroy();
    this.cleanupConnectionState();
  }

  public async shutdownDaemon(): Promise<void> {
    if (this.socket && !this.socket.destroyed) {
      try {
        await this.request('shutdown_daemon', {}, 3000);
      } catch {
        // ignore
      }
      this.socket.end();
      this.socket.destroy();
      this.cleanupConnectionState();
      return;
    }

    // No live socket yet (or already closed): send a one-shot daemon shutdown
    // request through a temporary connection.
    const socketPath = this.socketPath || BackendClient.defaultSocketPath();
    const tmp = new net.Socket();
    await new Promise<void>((resolve, reject) => {
      let done = false;
      const finish = (err?: Error) => {
        if (done) {
          return;
        }
        done = true;
        clearTimeout(timer);
        tmp.removeAllListeners();
        if (err) {
          reject(err);
        } else {
          resolve();
        }
      };
      const timer = setTimeout(() => finish(new Error('Timed out connecting to Binoculars daemon.')), 1200);
      tmp.once('error', (err) => finish(err as Error));
      tmp.connect(socketPath, () => {
        const req: BridgeRequest = { id: 1, method: 'shutdown_daemon', params: {} };
        tmp.write(JSON.stringify(req) + '\n', 'utf8', () => {
          tmp.end();
          finish();
        });
      });
    }).catch(() => undefined);
  }

  public dispose(): void {
    void this.shutdown();
  }

  private async startInternal(pythonPath: string, bridgeScriptPath: string, cwd?: string): Promise<void> {
    const socketPath = BackendClient.defaultSocketPath();
    this.socketPath = socketPath;

    // First try attaching to an already-running shared daemon.
    const connected = await this.tryConnect(socketPath, 1200);
    if (!connected) {
      try {
        if (fs.existsSync(socketPath)) {
          fs.unlinkSync(socketPath);
        }
      } catch {
        // ignore
      }

      // Stale socket paths are common after hard exits; unlink and relaunch.
      this.spawnDaemonProcess(pythonPath, bridgeScriptPath, socketPath, cwd);
      const ready = await this.waitForDaemon(socketPath, 12000);
      if (!ready) {
        this.startPromise = undefined;
        throw new Error(`Timed out waiting for Binoculars daemon socket at ${socketPath}`);
      }

      const retryConnected = await this.waitForConnect(socketPath, 12000, 1200);
      if (!retryConnected) {
        try {
          if (fs.existsSync(socketPath)) {
            fs.unlinkSync(socketPath);
          }
        } catch {
          // ignore
        }
        // One additional spawn/connect cycle before surfacing startup failure.
        this.spawnDaemonProcess(pythonPath, bridgeScriptPath, socketPath, cwd);
        const readyAgain = await this.waitForDaemon(socketPath, 12000);
        const retryConnectedAgain = readyAgain ? await this.waitForConnect(socketPath, 12000, 1200) : false;
        if (!retryConnectedAgain) {
          this.startPromise = undefined;
          throw new Error(`Unable to connect to Binoculars daemon at ${socketPath}`);
        }
      }
    }

    await this.waitForReady();
  }

  private static defaultSocketPath(): string {
    const uid = typeof process.getuid === 'function' ? String(process.getuid()) : String(process.env.USER ?? 'user');
    return path.join(os.tmpdir(), `binoculars-vscode-${uid}.sock`);
  }

  private spawnDaemonProcess(pythonPath: string, bridgeScriptPath: string, socketPath: string, cwd?: string): void {
    try {
      const socketDir = path.dirname(socketPath);
      fs.mkdirSync(socketDir, { recursive: true });
    } catch {
      // ignore
    }

    // Detached daemon: extension host restarts should not terminate the bridge.
    const child = cp.spawn(pythonPath, [bridgeScriptPath, '--daemon', '--socket-path', socketPath], {
      stdio: 'ignore',
      detached: true,
      cwd,
      env: process.env,
    });
    child.unref();
  }

  private async waitForDaemon(socketPath: string, timeoutMs: number): Promise<boolean> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      if (fs.existsSync(socketPath)) {
        return true;
      }
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    return fs.existsSync(socketPath);
  }

  private async waitForConnect(socketPath: string, totalTimeoutMs: number, perAttemptMs: number): Promise<boolean> {
    const deadline = Date.now() + totalTimeoutMs;
    while (Date.now() < deadline) {
      const ok = await this.tryConnect(socketPath, perAttemptMs);
      if (ok) {
        return true;
      }
      await new Promise((resolve) => setTimeout(resolve, 150));
    }
    return this.tryConnect(socketPath, perAttemptMs);
  }

  private async tryConnect(socketPath: string, timeoutMs: number): Promise<boolean> {
    // Each connection attempt starts from a clean request/pending state.
    this.cleanupConnectionState();
    this.readySeen = false;

    return new Promise<boolean>((resolve) => {
      const sock = net.createConnection(socketPath);
      let settled = false;
      const finish = (ok: boolean) => {
        if (settled) {
          return;
        }
        settled = true;
        clearTimeout(timer);
        sock.removeAllListeners('error');
        sock.removeAllListeners('connect');
        resolve(ok);
      };
      const timer = setTimeout(() => {
        sock.destroy();
        finish(false);
      }, timeoutMs);

      sock.once('error', () => {
        sock.destroy();
        finish(false);
      });

      sock.once('connect', () => {
        this.attachSocket(sock);
        finish(true);
      });
    });
  }

  private attachSocket(sock: net.Socket): void {
    this.socket = sock;
    this.rl = readline.createInterface({ input: sock });
    this.rl.on('line', (line: string) => this.onLine(line));
    sock.on('close', () => {
      this.output('Binoculars daemon connection closed.');
      this.cleanupPending(new Error('Binoculars daemon connection closed.'));
      this.cleanupConnectionState();
    });
    sock.on('error', (err) => {
      this.output(`Binoculars daemon socket error: ${(err as Error).message}`);
      this.cleanupPending(new Error('Binoculars daemon socket error.'));
      this.cleanupConnectionState();
    });
  }

  private onLine(line: string): void {
    const trimmed = line.trim();
    if (!trimmed) {
      return;
    }

    let msg: BridgeResponse;
    try {
      msg = JSON.parse(trimmed) as BridgeResponse;
    } catch {
      this.output(`[bridge:raw] ${trimmed}`);
      this.cleanupPending(new Error('Invalid JSON from Binoculars daemon.'));
      return;
    }

    if (msg.event) {
      if (msg.event === 'ready') {
        // Ready event is emitted by the daemon once per connection.
        this.readySeen = true;
        this.readyResolver?.();
        this.readyResolver = undefined;
        return;
      }
      this.output(`[bridge:event:${msg.event}] ${JSON.stringify(msg.payload ?? {})}`);
      return;
    }

    const id = msg.id;
    if (typeof id !== 'number') {
      this.output(`[bridge:unmatched] ${trimmed}`);
      return;
    }

    const pending = this.pending.get(id);
    if (!pending) {
      return;
    }
    clearTimeout(pending.timer);
    this.pending.delete(id);

    if (msg.error) {
      pending.reject(
        new BridgeRpcError(msg.error.message, {
          code: msg.error.code,
          details: msg.error.details,
          method: pending.method,
        }),
      );
      return;
    }
    pending.resolve(msg.result);
  }

  private waitForReady(timeoutMs = 10000): Promise<void> {
    if (this.readySeen) {
      return Promise.resolve();
    }
    return new Promise((resolve, reject) => {
      this.readyResolver = resolve;
      setTimeout(() => {
        if (this.readyResolver) {
          this.readyResolver = undefined;
          reject(new Error('Timed out waiting for daemon ready event.'));
        }
      }, timeoutMs);
    });
  }

  private request<T = unknown>(method: string, params: Record<string, unknown>, timeoutMs = BRIDGE_TIMEOUT_MS_DEFAULT): Promise<T> {
    if (!this.socket || this.socket.destroyed) {
      return Promise.reject(new Error('Binoculars daemon is not connected.'));
    }

    // Monotonic request IDs let us map asynchronous responses back to callers.
    const id = this.requestSeq++;
    const req: BridgeRequest = { id, method, params };

    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Bridge request timed out: ${method} (${Math.round(timeoutMs / 1000)}s)`));
      }, timeoutMs);
      this.pending.set(id, { resolve: (v) => resolve(v as T), reject, timer, method });

      this.socket?.write(JSON.stringify(req) + '\n', 'utf8', (err) => {
        if (err) {
          clearTimeout(timer);
          this.pending.delete(id);
          reject(err);
        }
      });
    });
  }

  private cleanupConnectionState(): void {
    this.rl?.close();
    this.rl = undefined;
    this.socket = undefined;
    this.startPromise = undefined;
    this.readyResolver = undefined;
    this.readySeen = false;
    this.lastInitializeSignature = '';
  }

  private cleanupPending(err: Error): void {
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(err);
    }
    this.pending.clear();
  }
}
