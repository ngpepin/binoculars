# Binoculars Project Agent Guide

## 1) Project Purpose

This repository implements a local, likelihood-based AI text forensics tool inspired by Binoculars.

Primary goal:
- Score markdown text using two related llama.cpp models (observer + performer) and compute:
  - `logPPL` for observer
  - `logXPPL` cross-entropy term
  - `B = logPPL / logXPPL` (Binoculars ratio)

Why this exists:
- `initial-scoping.md` documents that API-only approaches (OpenAI/Ollama top-N logprobs) are approximate and fragile.
- The chosen direction is faithful local scoring with full logits via `llama-cpp-python`.

## 2) Current Project State

Status:
- Early but functional prototype.
- One script, two tuned configs, and local model assets present.
- Regression tests for release line `v1.1.x` now exist under `tests/`.
- No threshold calibration/classification pipeline yet (scoring only).

Latest known commit:
- `cf60f09` - initial implementation + configs.

## 3) Repository Map

Key files:
- `binoculars.py`: main scoring CLI.
- `binoculars.sh`: venv-activating wrapper that can be run from any directory.
- `config.binoculars.json`: master profile map (`fast`/`long`) + default profile.
- `tests/test_regression_v1_1_x.py`: regression checks for heatmap formatting and profile resolution behavior.
- `tests/fixtures/Athens.md`: stable fixture copy used by regression tests.
- `config.llama31.cuda12gb.fast.json`: default config for <= ~4096 tokens.
- `config.llama31.cuda12gb.long.json`: safer long-doc config for <= ~8192 tokens.
- `initial-scoping.md`: conversation-derived technical scoping, rationale, and tuning lessons.
- `.gitignore`: ignores `models/`, `venv/`, and `initial-scoping.md`.

Local assets (present on this machine):
- `models/Meta-Llama-3.1-8B-Q5_K_M-GGUF/meta-llama-3.1-8b-q5_k_m.gguf` (~5.4G)
- `models/Meta-Llama-3.1-8B-Instruct-Q5_K_M-GGUF/meta-llama-3.1-8b-instruct-q5_k_m.gguf` (~5.4G)

## 4) How the Implementation Works

High-level flow in `binoculars.py`:
1. Load JSON config and validate required sections (`observer`, `performer`).
2. Read input markdown from file or stdin.
3. Tokenize text with each model in `vocab_only=True` mode.
4. Enforce exact tokenization match (hard fail if mismatch).
5. Infer `n_ctx` (auto when configured as `0`).
6. Load observer model with `logits_all=True`, run `eval(tokens)`.
7. Compute observer `logPPL`, save observer logits to disk memmap (`observer_logits.dat`).
8. Unload observer (VRAM reduction).
9. Load performer model with `logits_all=True`, run `eval(tokens)`.
10. Compute performer `logPPL` (informational) and cross `logXPPL` using observer memmap + performer logits.
11. Compute `B = logPPL(observer) / logXPPL(observer, performer)`.
12. Emit text or JSON output.
13. Remove cache unless `cache.keep=true`.

Design choice:
- Models are loaded sequentially (never concurrently) to contain VRAM usage.

## 5) Config Profiles and Intent

`config.llama31.cuda12gb.fast.json`:
- `max_tokens: 4096`
- `offload_kqv: true`
- `n_batch: 1024`
- Faster default on 12 GB VRAM when docs are moderate length.

`config.llama31.cuda12gb.long.json`:
- `max_tokens: 8192`
- `offload_kqv: false`
- `n_batch: 512`
- Safer for longer docs; shifts KV cache pressure to system RAM.

Both configs:
- Use Llama 3.1 8B base + instruct Q5_K_M sibling models.
- Use `n_ctx: 0` (auto = token count).
- Use cache dtype `float16` to reduce disk/RAM pressure.

## 6) Environment and Bootstrap

Observed local state:
- Python in repo venv: `3.10.12`.
- Required packages are currently missing in `venv` (`numpy`, `llama_cpp` not installed).

Install baseline dependencies:
```bash
venv/bin/pip install numpy llama-cpp-python
```

Run example:
```bash
venv/bin/python binoculars.py --config fast your_doc.md --json
```

## 7) Output Contract

JSON output includes:
- `input` metadata (chars, tokens, transitions)
- `observer` (`logPPL`, `PPL`)
- `performer` (`logPPL`, `PPL`)
- `cross` (`logXPPL`, `XPPL`)
- `binoculars.score` (`B`)
- `cache` details

Important:
- The script returns scores only. No built-in thresholding/classification labels.

Heatmap markdown output (`--heatmap`) includes:
- In-text note indices that link to the `Notes Table` section.
- A consolidated `Notes Table` with columns in this order:
  `Index | Label | % contribution | Paragraph | logPPL | delta_vs_doc | delta_if_removed | Transitions | Chars | Tokens`.
- Console heatmap output is terminal-oriented (ANSI colors + plain `[N]` note markers + line-drawing table), strips markdown hard-break `\` markers, collapses excessive blank lines, and wraps content to ~85% of terminal width.

## 8) Lessons Learned / Gotchas

From `initial-scoping.md` + implementation behavior:

1. Full-logit requirement is non-negotiable for faithful Binoculars.
- API top-k logprob approaches are approximate and biased.

2. Tokenizer/vocab alignment is critical.
- Script fails if tokenization differs between observer and performer.
- Use same-family base/instruct siblings.

3. Memory pressure is dominated by `(tokens * vocab)` with `logits_all=True`.
- Llama 3.1 vocab (~128k) is expensive for long inputs.
- `text.max_tokens` is the main safety valve.

4. `n_ctx: 0` auto-sizing helps avoid unnecessary logits buffer size.
- Manual low `n_ctx` causes hard failure if below needed tokens.

5. Long docs can be operationally expensive even when VRAM is fine.
- RAM/disk cache growth can become bottlenecks.

6. Markdown is treated as plain text.
- Formatting is preserved in input but not semantically specialized.

7. No threshold calibration yet.
- `B` is not directly actionable without evaluation data and calibration.

## 9) Known Gaps / Next Development Priorities

Priority backlog for next agent:
1. Add calibration pipeline:
- Build dataset runner + threshold selection + FPR/TPR reporting.

2. Add sliding-window scoring mode:
- Score long documents without truncation and without forcing full-doc logits memory.

3. Add optional tokenizer mismatch fallback:
- Common-prefix mode as optional non-default behavior.

4. Add tests:
- Unit tests for perplexity/cross-perplexity math on synthetic logits.
- Smoke test for config parsing and error paths.

5. Add dependency pinning:
- Add `requirements.txt` or `pyproject.toml`.

6. Add reproducible benchmark script:
- Throughput/memory measurements across configs and input lengths.

## 10) Agent Operating Notes

When resuming work:
1. Read `initial-scoping.md` sections near:
- model-pair decisions,
- CUDA 12 GB tuning,
- FAST vs LONG config rationale,
- OOM mitigation guidance.

2. Verify environment before coding:
- dependency availability,
- model paths in config,
- writable cache directory.

3. Avoid silent behavior changes in core scoring math.
- Any changes to token alignment, cross-entropy math, or truncation strategy require explicit documentation.

4. Preserve sequential model loading unless intentionally redesigning memory strategy.

5. If adding classification labels, keep raw numeric outputs and expose calibration metadata.

## 11) Non-Goals (Current)

Not currently in scope:
- Claims of definitive authorship detection.
- Remote API-based approximate detector implementation.
- Production web service deployment.
- UI/visualization layer.
