import * as cp from 'node:child_process';
import * as fs from 'node:fs';
import * as net from 'node:net';
import * as os from 'node:os';
import * as path from 'node:path';
import * as readline from 'node:readline';
import * as vscode from 'vscode';
import { AnalyzeResult, BridgeRequest, BridgeResponse, RewriteResult } from './types';

export class BackendClient implements vscode.Disposable {
  private socket: net.Socket | undefined;
  private rl: readline.Interface | undefined;
  private requestSeq = 1;
  private pending = new Map<number, { resolve: (value: unknown) => void; reject: (err: Error) => void; timer: NodeJS.Timeout }>();
  private readonly output: vscode.OutputChannel;
  private readyResolver: (() => void) | undefined;
  private readyRejecter: ((err: Error) => void) | undefined;
  private readySeen = false;
  private startPromise: Promise<void> | undefined;
  private socketPath = '';

  constructor(output: vscode.OutputChannel) {
    this.output = output;
  }

  public async start(pythonPath: string, bridgeScriptPath: string): Promise<void> {
    if (this.socket && !this.socket.destroyed) {
      return;
    }
    if (this.startPromise) {
      return this.startPromise;
    }
    this.startPromise = this.startInternal(pythonPath, bridgeScriptPath);
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
    await this.request('initialize', {
      cfg_path: cfgPath,
      top_k: topK,
      text_max_tokens_override: textMaxTokensOverride,
      observer_model_path: observerModelPath,
      performer_model_path: performerModelPath,
      rewrite_llm_config_path: rewriteLlmConfigPath,
    });
  }

  public analyzeDocument(text: string, inputLabel: string): Promise<AnalyzeResult> {
    return this.request<AnalyzeResult>('analyze_document', {
      text,
      input_label: inputLabel,
    });
  }

  public analyzeChunk(text: string, inputLabel: string, startChar: number, endChar: number): Promise<AnalyzeResult> {
    return this.request<AnalyzeResult>('analyze_chunk', {
      text,
      input_label: inputLabel,
      start_char: startChar,
      end_char: endChar,
    });
  }

  public analyzeNextChunk(text: string, inputLabel: string, startChar?: number): Promise<AnalyzeResult> {
    return this.request<AnalyzeResult>('analyze_next_chunk', {
      text,
      input_label: inputLabel,
      start_char: startChar,
    });
  }

  public estimateLiveB(
    text: string,
    inputLabel: string,
    startChar: number,
    baseCrossLogXppl: number,
  ): Promise<{
    ok: boolean;
    approx_b: number;
    observer_logPPL: number;
    transitions: number;
    analyzed_char_end: number;
    truncated_by_limit: boolean;
  }> {
    return this.request('estimate_live_b', {
      text,
      input_label: inputLabel,
      start_char: startChar,
      base_cross_logxppl: baseCrossLogXppl,
    });
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
    });
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

  private async startInternal(pythonPath: string, bridgeScriptPath: string): Promise<void> {
    const socketPath = BackendClient.defaultSocketPath();
    this.socketPath = socketPath;

    const connected = await this.tryConnect(socketPath, 1200);
    if (!connected) {
      this.spawnDaemonProcess(pythonPath, bridgeScriptPath, socketPath);
      const ready = await this.waitForDaemon(socketPath, 12000);
      if (!ready) {
        this.startPromise = undefined;
        throw new Error(`Timed out waiting for Binoculars daemon socket at ${socketPath}`);
      }
      const retryConnected = await this.tryConnect(socketPath, 1500);
      if (!retryConnected) {
        this.startPromise = undefined;
        throw new Error(`Unable to connect to Binoculars daemon at ${socketPath}`);
      }
    }

    await this.waitForReady();
  }

  private static defaultSocketPath(): string {
    const uid = typeof process.getuid === 'function' ? String(process.getuid()) : String(process.env.USER ?? 'user');
    return path.join(os.tmpdir(), `binoculars-vscode-${uid}.sock`);
  }

  private spawnDaemonProcess(pythonPath: string, bridgeScriptPath: string, socketPath: string): void {
    try {
      const socketDir = path.dirname(socketPath);
      fs.mkdirSync(socketDir, { recursive: true });
    } catch {
      // ignore; python side will fail with actionable error if unusable
    }

    const child = cp.spawn(pythonPath, [bridgeScriptPath, '--daemon', '--socket-path', socketPath], {
      stdio: 'ignore',
      detached: true,
      cwd: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath,
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

  private async tryConnect(socketPath: string, timeoutMs: number): Promise<boolean> {
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
      this.output.appendLine('Binoculars daemon connection closed.');
      this.cleanupPending(new Error('Binoculars daemon connection closed.'));
      this.cleanupConnectionState();
    });
    sock.on('error', (err) => {
      this.output.appendLine(`Binoculars daemon socket error: ${(err as Error).message}`);
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
      this.output.appendLine(`[bridge:raw] ${trimmed}`);
      return;
    }

    if (msg.event) {
      if (msg.event === 'ready') {
        this.readySeen = true;
        this.readyResolver?.();
        this.readyResolver = undefined;
        this.readyRejecter = undefined;
        return;
      }
      this.output.appendLine(`[bridge:event:${msg.event}] ${JSON.stringify(msg.payload ?? {})}`);
      return;
    }

    const id = msg.id;
    if (typeof id !== 'number') {
      this.output.appendLine(`[bridge:unmatched] ${trimmed}`);
      return;
    }

    const pending = this.pending.get(id);
    if (!pending) {
      return;
    }
    clearTimeout(pending.timer);
    this.pending.delete(id);

    if (msg.error) {
      pending.reject(new Error(msg.error.message));
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
      this.readyRejecter = reject;
      setTimeout(() => {
        if (this.readyResolver) {
          this.readyResolver = undefined;
          this.readyRejecter = undefined;
          reject(new Error('Timed out waiting for daemon ready event.'));
        }
      }, timeoutMs);
    });
  }

  private request<T = unknown>(method: string, params: Record<string, unknown>, timeoutMs = 180000): Promise<T> {
    if (!this.socket || this.socket.destroyed) {
      return Promise.reject(new Error('Binoculars daemon is not connected.'));
    }

    const id = this.requestSeq++;
    const req: BridgeRequest = { id, method, params };

    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Bridge request timed out: ${method}`));
      }, timeoutMs);
      this.pending.set(id, { resolve: (v) => resolve(v as T), reject, timer });

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
    this.readyRejecter = undefined;
    this.readySeen = false;
  }

  private cleanupPending(err: Error): void {
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(err);
    }
    this.pending.clear();
  }
}
