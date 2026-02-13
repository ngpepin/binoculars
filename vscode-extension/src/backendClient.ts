import * as cp from 'node:child_process';
import * as readline from 'node:readline';
import * as vscode from 'vscode';
import { AnalyzeResult, BridgeRequest, BridgeResponse, RewriteResult } from './types';

export class BackendClient implements vscode.Disposable {
  private proc: cp.ChildProcessWithoutNullStreams | undefined;
  private rl: readline.Interface | undefined;
  private requestSeq = 1;
  private pending = new Map<number, { resolve: (value: unknown) => void; reject: (err: Error) => void; timer: NodeJS.Timeout }>();
  private readonly output: vscode.OutputChannel;
  private readyResolver: (() => void) | undefined;
  private readyRejecter: ((err: Error) => void) | undefined;
  private startPromise: Promise<void> | undefined;

  constructor(output: vscode.OutputChannel) {
    this.output = output;
  }

  public async start(pythonPath: string, bridgeScriptPath: string): Promise<void> {
    if (this.proc) {
      return;
    }
    if (this.startPromise) {
      return this.startPromise;
    }

    this.proc = cp.spawn(pythonPath, [bridgeScriptPath], {
      stdio: ['pipe', 'pipe', 'pipe'],
      cwd: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath,
      env: process.env,
    });

    this.proc.stderr.on('data', (buf: Buffer) => {
      this.output.appendLine(`[bridge:stderr] ${buf.toString('utf8').trimEnd()}`);
    });

    this.proc.on('exit', (code, signal) => {
      this.output.appendLine(`Binoculars bridge exited (code=${code}, signal=${signal ?? 'none'})`);
      this.cleanupPending(new Error('Binoculars bridge terminated.'));
      this.readyRejecter?.(new Error('Binoculars bridge terminated before ready.'));
      this.proc = undefined;
      this.rl?.close();
      this.rl = undefined;
      this.startPromise = undefined;
    });

    this.rl = readline.createInterface({ input: this.proc.stdout });
    this.rl.on('line', (line: string) => this.onLine(line));

    this.startPromise = this.waitForReady();
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
    if (!this.proc) {
      return;
    }
    try {
      await this.request('shutdown', {}, 2000);
    } catch {
      // ignore
    }
    this.proc.kill();
  }

  public dispose(): void {
    void this.shutdown();
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
        this.readyResolver?.();
        this.readyResolver = undefined;
        this.readyRejecter = undefined;
        return;
      }
      if (msg.event !== 'ready') {
        this.output.appendLine(`[bridge:event:${msg.event}] ${JSON.stringify(msg.payload ?? {})}`);
      }
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
    return new Promise((resolve, reject) => {
      this.readyResolver = resolve;
      this.readyRejecter = reject;
      setTimeout(() => {
        if (this.readyResolver) {
          this.readyResolver = undefined;
          this.readyRejecter = undefined;
          reject(new Error('Timed out waiting for bridge ready event.'));
        }
      }, timeoutMs);
    });
  }

  private request<T = unknown>(method: string, params: Record<string, unknown>, timeoutMs = 180000): Promise<T> {
    if (!this.proc) {
      return Promise.reject(new Error('Bridge is not started.'));
    }

    const id = this.requestSeq++;
    const req: BridgeRequest = { id, method, params };

    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Bridge request timed out: ${method}`));
      }, timeoutMs);
      this.pending.set(id, { resolve: (v) => resolve(v as T), reject, timer });

      this.proc?.stdin.write(JSON.stringify(req) + '\n', 'utf8', (err) => {
        if (err) {
          clearTimeout(timer);
          this.pending.delete(id);
          reject(err);
        }
      });
    });
  }

  private cleanupPending(err: Error): void {
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(err);
    }
    this.pending.clear();
  }
}
