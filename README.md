# Binoculars (Local Text Forensics with llama.cpp)

`binoculars` is a local, likelihood-based AI text forensics tool inspired by the Binoculars approach.  
It scores markdown/plain text with two related language models (observer + performer), computes Binoculars-style metrics, and can generate a paragraph-level perplexity heatmap.

This repository is designed for:
- Local/offline scoring
- Reproducible config profiles (`fast`, `long`)
- GPU-constrained workflows (models loaded sequentially)

## Background

The approach is based on the Binoculars paper and related scoping work:
- Paper PDF included in this repo under `background/2401.12070v3.pdf`
- Additional local project scoping notes in `initial-scoping.md`

## Theory

Let a tokenized document be:

\[
x_1, x_2, \dots, x_T
\]

Use two aligned causal LMs (same tokenizer/vocab):
- Observer model \(M_o\)
- Performer model \(M_p\)

### 1) Observer log-perplexity

\[
\log PPL_{M_o}(x) = -\frac{1}{T-1}\sum_{t=1}^{T-1}\log p_{M_o}(x_{t+1}\mid x_{\le t})
\]

This is the mean next-token negative log-likelihood under the observer.

### 2) Cross log-perplexity (distributional cross-entropy)

At each position \(t\), let:
- \(p_o(\cdot \mid x_{\le t})\): observer next-token distribution
- \(p_p(\cdot \mid x_{\le t})\): performer next-token distribution

\[
\log XPPL_{M_o,M_p}(x)=
-\frac{1}{T-1}\sum_{t=1}^{T-1}
\sum_{v\in V}
p_o(v\mid x_{\le t})\log p_p(v\mid x_{\le t})
\]

### 3) Binoculars ratio

\[
B(x)=\frac{\log PPL_{M_o}(x)}{\log XPPL_{M_o,M_p}(x)}
\]

The script reports:
- `observer.logPPL`, `observer.PPL`
- `performer.logPPL`, `performer.PPL` (informational)
- `cross.logXPPL`, `cross.XPPL`
- `binoculars.score` (the ratio \(B\))

## Interpretation Notes

- Lower observer perplexity usually means text is more predictable to the observer LM.
- In many settings, that can correlate with “more LM-like” text.
- This is **not** a proof of authorship.
- Reliable operational decisions require calibration (thresholds, ROC/FPR/TPR on your own data).

## Implementation Design

`binoculars.py` uses full logits from `llama-cpp-python` and evaluates models sequentially:

1. Tokenize input with each model in `vocab_only` mode.
2. Enforce exact tokenization match (hard fail on mismatch).
3. Run observer with `logits_all=True`; compute observer metrics.
4. Persist observer logits to memmap on disk.
5. Unload observer; load performer.
6. Compute cross-perplexity using observer memmap + performer logits.
7. Compute Binoculars ratio.

Why sequential loading:
- Keeps VRAM lower by never holding both models fully resident at once.

## Perplexity Heatmap

`--heatmap` produces markdown with colored paragraph spans and a notes table:
- **Red** = lowest paragraph logPPL (more predictable to observer)
- **Green** = highest paragraph logPPL (less predictable to observer)

Each highlighted span has a note index that links to the `Notes Table` section.  
The notes table includes:
- Index
- Label (`LOW`/`HIGH`)
- `% contribution` (signed effect on overall observer logPPL)
- Paragraph id
- `logPPL`
- `delta_vs_doc`
- `delta_if_removed`
- Transition count
- Character/token ranges

Output behavior:
- Printed to console in terminal-friendly format:
  - ANSI color highlights (red/green)
  - plain note markers like `[1]` (non-clickable)
  - notes table rendered with line-drawing characters
  - wrapped to ~85% of terminal width with normalized spacing
- Written to `<input_stem>_heatmap.md` in the same source directory
- Previous file auto-backed up as `<name>.<timestamp>.bak`

## GUI Mode

`--gui <file>` opens an interactive editor/analyzer window with:
- Buttons: `Analyze`, `Save`, `Quit`
- A one-line status bar starting with: `Binocular score B (high is more human-like): ...`
- Paragraph heatmap coloring after each analysis:
  - **Red** = lower paragraph logPPL (more AI-like)
  - **Green** = higher paragraph logPPL (more human-like)
- Hover tooltips on colored segments with the same stats shown in the notes table format
- Edit tracking in **yellow** for changed text since last analysis

Behavior:
- `Analyze` scores the full current editor text, refreshes coloring/tooltips, clears yellow edit tags, and preserves cursor position.
- `Save` writes `<stem>_edited_YYYYMMDDHHMM.md` in the source file directory.
- `Quit` closes the window.

## Repository Layout

- `binoculars.py`: main scoring CLI
- `binoculars.sh`: venv wrapper launcher
- `binocular.sh`: convenience alias wrapper
- `config.binoculars.json`: master profile map + default profile
- `config.llama31.cuda12gb.fast.json`: tuned fast profile
- `config.llama31.cuda12gb.long.json`: tuned long-doc profile
- `background/2401.12070v3.pdf`: reference paper
- `samples/`: example inputs

## Requirements

- Linux/macOS shell environment
- Python 3.10+ (project currently tested on Python 3.10)
- `llama-cpp-python`
- `numpy`
- GGUF model files referenced by your profile configs

Install dependencies (in venv):

```bash
venv/bin/pip install numpy llama-cpp-python
```

If using NVIDIA CUDA and you need GPU-enabled wheels, install `llama-cpp-python` from the appropriate CUDA wheel index for your system.

## Configuration Model

Master config (`config.binoculars.json`) maps profile labels to concrete config files:

```json
{
  "default": "fast",
  "profiles": {
    "fast": "/home/npepin/Projects/binoculars/config.llama31.cuda12gb.fast.json",
    "long": "/home/npepin/Projects/binoculars/config.llama31.cuda12gb.long.json"
  }
}
```

Notes:
- `default` is used when `--config` is omitted.
- `--config` takes the profile label (`fast` or `long`), not a JSON filepath.
- If you move the repo, update absolute paths in `config.binoculars.json`.

## Usage

Show help:

```bash
./binoculars.sh --help
```

Basic scoring (default profile from master config):

```bash
./binoculars.sh samples/Athens.md --json
```

Select profile explicitly:

```bash
./binoculars.sh --config fast samples/Athens.md --json
./binoculars.sh --config=long samples/Athens.md --json
```

Using explicit `--input`:

```bash
./binoculars.sh --config fast --input samples/Athens.md --json
```

Heatmap:

```bash
./binoculars.sh --config fast --input samples/Athens.md --heatmap --diagnose-top-k 8
```

GUI editor:

```bash
./binoculars.sh --config fast --gui samples/Athens.md
```

Run from any directory:

```bash
cd ~
/home/npepin/Projects/binoculars/binoculars.sh --config long /tmp/myfile.md --json
# alias wrapper also works:
/home/npepin/Projects/binoculars/binocular.sh --config=fast /tmp/myfile.md
```

## CLI Notes

- Input can be provided either as:
  - positional `INPUT`, or
  - `--input INPUT`
- Do not provide both at once.
- If no input is provided, stdin is used (`-` behavior).
- `--gui` is mutually exclusive with `--input`, positional `INPUT`, `--output`, `--json`, and `--heatmap`.

## Regression Tests (v1.1.x)

Run the regression suite:

```bash
./venv/bin/python -m unittest -v tests/test_regression_v1_1_x.py
```

Fixtures:
- `tests/fixtures/Athens.md` (copied from `samples/Athens.md` for test stability)

## Output Contract (JSON)

Top-level keys:
- `input`
- `observer`
- `performer`
- `cross`
- `binoculars`
- `cache`

Optional:
- `diagnostics.low_perplexity_spans` (when paragraph diagnostics enabled)

## Operational Caveats

- Tokenizer/vocab mismatch between models is a hard failure by design.
- Long contexts can be expensive in memory due to `logits_all=True` and large vocab.
- `text.max_tokens` in profile configs is the main control for context/memory pressure.
- Use detector outputs as risk signals, not standalone judgments.
