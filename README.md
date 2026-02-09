# Binoculars

Local, likelihood-based AI text forensics using two `llama.cpp` models (observer + performer), inspired by the Binoculars approach.

See paper (also included in repo under background/) : https://arxiv.org/abs/2401.12070

This project focuses on faithful local scoring with full logits, not API approximations.

## What This Does

Given input text, `binoculars` computes:

- Observer log-perplexity: `logPPL`
- Cross log-perplexity: `logXPPL` (observer distribution scored against performer distribution)
- Binoculars ratio: `B = logPPL / logXPPL`

It can also generate paragraph-level diagnostics and heatmaps.

## Why Local / Full Logits

The Binoculars-style cross-entropy term depends on full next-token distributions. In practice, this means:

- `logits_all=True` is required
- top-k API logprobs are not sufficient for faithful reconstruction
- observer/performer tokenizer alignment must be exact

Reference paper is included at:

- `background/2401.12070v3.pdf`

Additional local design notes:

- `initial-scoping.md` (local, gitignored)

## Theory

Let a tokenized document be:

$$
x_1,\, x_2,\, \dots,\, x_T
$$

with observer model $M_o$ and performer model $M_p$.

**Observer log-perplexity:**

$$
\log \mathrm{PPL}_{M_o}(x) = -\frac{1}{T-1} \sum_{t=1}^{T-1} \log p_{M_o}(x_{t+1} \mid x_{\leq t})
$$

**Cross log-perplexity:**

$$
\log \mathrm{XPPL}_{M_o, M_p}(x) = -\frac{1}{T-1} \sum_{t=1}^{T-1} \sum_{v \in V} p_{M_o}(v \mid x_{\leq t}) \log p_{M_p}(v \mid x_{\leq t})
$$

**Binoculars score:**

$$
B(x) = \frac{\log \mathrm{PPL}_{M_o}(x)}{\log \mathrm{XPPL}_{M_o, M_p}(x)}
$$

Current UI/CLI interpretation used by this repo:

- Higher `B` is treated as more human-like
- Lower paragraph `logPPL` is treated as more AI-like for heatmap coloring

Important: this is a scoring signal, not proof of authorship.

## Repository Layout

- `binoculars.py`: main CLI + GUI application
- `binoculars.sh`: wrapper that activates venv, auto-cleans old instances, forwards Ctrl-C
- `binocular.sh`: alias wrapper (`exec binoculars.sh`)
- `config.binoculars.json`: master profile selector (`default` + `profiles`)
- `config.llama31.cuda12gb.fast.json`: fast profile (currently `text.max_tokens=4096`)
- `config.llama31.cuda12gb.long.json`: long profile (currently `text.max_tokens=12288`)
- `background/2401.12070v3.pdf`: background paper
- `samples/`: sample markdown inputs
- `tests/test_regression_v1_1_x.py`: regression suite
- `tests/fixtures/`: fixture docs used by regression tests

## Requirements

- Linux or macOS shell
- Python 3.10+
- `numpy`
- `llama-cpp-python`
- GGUF models on local disk

Install into repo venv:

```bash
venv/bin/pip install numpy llama-cpp-python
```

## Model Files

Configs in this repo point to local model paths under `models/`, for example:

- Base observer: Llama 3.1 8B Q5_K_M
- Instruct performer: Llama 3.1 8B Instruct Q5_K_M

You can use different models if tokenizer/vocab alignment is preserved.

## Configuration

### 1) Master profile config

`config.binoculars.json` selects profile by label:

```json
{
  "default": "long",
  "profiles": {
    "fast": "/abs/path/config.llama31.cuda12gb.fast.json",
    "long": "/abs/path/config.llama31.cuda12gb.long.json"
  }
}
```

`profiles` entries can be either:

- string path (current repo default), or
- object form:

```json
{
  "path": "/abs/path/config.json",
  "max_tokens": 8192
}
```

`max_tokens` in object form (if present) overrides `text.max_tokens` in the concrete profile.

### 2) Concrete observer/performer profile

Each profile must define:

- `observer`
- `performer`

Optional blocks:

- `text` (`add_bos`, `special_tokens`, `max_tokens`)
- `cache` (`dir`, `dtype`, `keep`)

Notes:

- `n_ctx: 0` means auto (`n_ctx = analyzed token count`)
- `text.max_tokens > 0` truncates input token window
- `cache.dtype` may be `float16` or `float32`

## Execution Model

`binoculars.py` loads models sequentially:

1. Tokenize with observer/performer in `vocab_only=True`
2. Hard-fail if tokenization differs
3. Run observer with full logits
4. Persist observer logits to memmap
5. Unload observer, load performer
6. Compute cross-entropy term from observer distribution vs performer logits
7. Emit metrics, optional diagnostics/heatmap

This keeps VRAM lower than concurrent dual-model loading.

## CLI Usage

Show help:

```bash
./binoculars.sh --help
```

Basic:

```bash
./binoculars.sh samples/Athens.md
```

JSON:

```bash
./binoculars.sh --config long samples/Athens.md --json
```

Heatmap:

```bash
./binoculars.sh --config fast --input samples/Athens.md --heatmap --diagnose-top-k 10
```

Diagnostics:

```bash
./binoculars.sh --diagnose-paragraphs --diagnose-top-k 10 samples/Athens.md
./binoculars.sh --diagnose-paragraphs --diagnose-print-text samples/Athens.md
```

Run from any directory:

```bash
cd ~
/home/npepin/Projects/binoculars/binoculars.sh --config long /tmp/doc.md --json
```

Alias wrapper:

```bash
/home/npepin/Projects/binoculars/binocular.sh --config fast /tmp/doc.md
```

### Input rules

- Provide input either as:
  - positional `INPUT`, or
  - `--input INPUT`
- If both are provided, command errors
- If multiple positional paths are provided, only the first is used (warning emitted)
- If no input is given, stdin (`-`) is used

### CLI options

- `--master-config FILE`: master profile mapping file
- `--config PROFILE`: profile label (`fast`, `long`, etc.)
- `--input FILE|-`: explicit input
- `--output FILE`: write text output
- `--json`: emit JSON result object
- `--diagnose-paragraphs`: rank low-perplexity hotspot paragraphs
- `--diagnose-top-k N`: hotspot count (also used by heatmap selection)
- `--diagnose-print-text`: print full hotspot text segments
- `--heatmap`: emit console + markdown heatmap output
- `--gui FILE`: launch interactive GUI editor/analyzer

`--heatmap` cannot be combined with `--json`.

`--gui` is mutually exclusive with:

- `--input`
- positional `INPUT`
- `--output`
- `--json`
- `--heatmap`

## Heatmap Mode (`--heatmap`)

When enabled:

- console output shows:
  - red/green highlights (ANSI)
  - simple note markers like `[1]`
  - line-drawing notes table
  - wrapped layout (about 85% terminal width)
- file output writes markdown to:
  - `<input_stem>_heatmap.md` in the same directory as source input
- existing heatmap file is backed up first:
  - `<name>.YYYYMMDD_HHMMSS.bak` (timestamp format may vary by implementation helper)

Heatmap semantics:

- Red sections: lowest paragraph `logPPL`
- Green sections: highest paragraph `logPPL`
- Note table columns:
  - `Index`
  - `Label`
  - `% contribution`
  - `Paragraph`
  - `logPPL`
  - `delta_vs_doc`
  - `delta_if_removed`
  - `Transitions`
  - `Chars`
  - `Tokens`

## GUI Mode (`--gui <file>`)

Launches an editor/analyzer with:

- Left pane: editable source text
- Left gutter:
  - logical line numbers
  - red/green contribution bars per line
- Right pane: live markdown preview
- Controls:
  - `Analyze`
  - `Save`
  - `Clear Priors`
  - `Quit`
- Status bar:
  - `Binocular score B (high is more human-like): ...`
  - includes prior score and `Last Line`

### GUI behavior

- `Analyze`:
  - scores full current document
  - preserves cursor and top-view position
  - updates red/green foreground highlights
  - updates hover tooltips
  - updates status metrics
- Edits since last analysis show in yellow
- On next `Analyze`, previous highlights/edits become faint prior backgrounds
- `Clear Priors` removes faint prior backgrounds only
- `Save` writes:
  - `<stem>_edited_YYYYMMDDHHMM.md`
  - same directory as source
- Always-on English spell checking:
  - misspellings marked with red underline

### Preview sync + debug controls

Environment variables:

- `BINOCULARS_GUI_DEBUG=1`
  - starts with debug overlay enabled
  - toggle in-app with `F9`
- `BINOCULARS_PREVIEW_VIEW_OFFSET_LINES=-3`
  - vertical view calibration for right pane
  - changes preview viewport position only (not line mapping)

## Wrapper behavior (`binoculars.sh`)

`binoculars.sh`:

- activates repo venv
- runs `binoculars.py`
- deactivates venv on exit
- forwards Ctrl-C to child process
- terminates prior running instances by default to free GPU/VRAM

Disable auto-kill if needed:

```bash
BINOCULARS_DISABLE_AUTO_KILL=1 ./binoculars.sh ...
```

## Output Contract (JSON)

Top-level keys:

- `input`
- `observer`
- `performer`
- `cross`
- `binoculars`
- `cache`

Optional:

- `diagnostics.low_perplexity_spans` (when `--diagnose-paragraphs` enabled)

## Performance and Tuning Notes

- Main memory pressure comes from full logits (`tokens x vocab`)
- Long contexts are expensive even if VRAM appears available
- `text.max_tokens` is the primary cap for runtime/memory safety
- `n_ctx: 0` is usually best (auto-size to analyzed tokens)
- Observer/performer are loaded sequentially by design

Current shipped profile token limits:

- `fast`: 4096
- `long`: 12288

Adjust these in profile JSONs based on your machine.

## Troubleshooting

Tokenizer mismatch error:

- Use same-family model pair (base + instruct sibling)
- Ensure both configs reference compatible tokenizer/vocab models

Missing file errors:

- Validate `config.binoculars.json` profile paths
- Validate model paths in concrete config JSONs

GUI unavailable:

- Ensure Tkinter is installed for your Python environment

Unexpected GPU memory contention:

- Close other LLM processes, or rely on wrapper auto-kill
- reduce `text.max_tokens`
- reduce `n_batch` if needed

## Tests

Run regression suite:

```bash
./venv/bin/python -m unittest -v tests/test_regression_v1_1_x.py
```

## Limitations

- No built-in calibrated classifier thresholds yet
- No claim of definitive authorship attribution
- Markdown is analyzed as text; no semantic markdown parsing
- Long documents may require truncation due full-logit cost

## Safety / Responsible Use

Use outputs as probabilistic signals in a broader review workflow.  
Do not use this tool as a sole basis for punitive or high-stakes decisions.
