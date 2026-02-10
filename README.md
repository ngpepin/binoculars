# Binoculars

`Binoculars` is a local AI-text forensics and humanization workflow tool. It uses two related `llama.cpp` models (observer + performer) to compute faithful Binoculars-style scores from full logits, then helps you iteratively rewrite high-risk spans toward more human-like text in the GUI.

If you need to evaluate or revise long-form text without handing documents to a hosted detector, this repo gives you a practical workflow:

- Keep data and models local by default.
- Compute inspectable signals (`logPPL`, `logXPPL`, `B`) instead of opaque labels.
- See exactly where score pressure comes from with paragraph-level heatmaps.
- Generate 3 rewrite options for a flagged line or selected block and rank them by approximate B impact for fast humanization passes.
- Re-run full Analyze only when needed for exact checkpoint scores.
- Optionally use an OpenAI-compatible rewrite backend, with automatic fallback to internal local generation.

Reference paper (also in `background/`): https://arxiv.org/abs/2401.12070

<p align="center">
  <img src="media/screenshot.png" width="900">
  <br/>
  <em>Screenshot of the GUI</em>
</p>

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
- `config.binoculars.llm.json`: optional OpenAI-compatible rewrite backend config for GUI rewrite suggestions
- `config.llama31.cuda12gb.fast.json`: fast profile (currently `text.max_tokens=4096`)
- `config.llama31.cuda12gb.long.json`: long profile (currently `text.max_tokens=12288`)
- `USERGUIDE-GUI.md`: detailed GUI user guide and iterative workflow guidance
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

### 3) Optional rewrite LLM config (GUI)

If `config.binoculars.llm.json` is present and enabled, GUI rewrite suggestions can use an external OpenAI-compatible endpoint.
If missing or disabled, internal performer-model generation is used.
If present but unreachable/invalid at runtime, GUI rewrites automatically fall back to internal generation.

Supported fields include:

- `llm.enabled`
- `llm.endpoint_url`
- `llm.request_path` (default `/chat/completions`)
- `llm.model`
- `llm.api_key` or `llm.api_key_env` (also supports `OPENAI_API_KEY` when enabled)
- `llm.api_key_header` / `llm.api_key_prefix`
- `llm.timeout_s`
- `llm.max_tokens`
- `llm.temperature`
- `llm.top_p`
- `llm.context_chars_each_side`
- `llm.context_paragraphs_each_side`
- `llm.context_window_max_chars`
- `llm.extra_headers`
- `llm.extra_body`

Example:

```json
{
  "llm": {
    "enabled": true,
    "endpoint_url": "http://localhost:4141/v1",
    "model": "gpt-4.1",
    "request_path": "/chat/completions",
    "api_key_env": "OPENAI_API_KEY",
    "max_tokens": 220,
    "temperature": 0.78,
    "top_p": 0.95,
    "context_chars_each_side": 1800,
    "context_paragraphs_each_side": 2,
    "context_window_max_chars": 5200
  }
}
```

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

For a detailed workflow-oriented guide, see:

- `USERGUIDE-GUI.md`

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
- Rewrite suggestions:
  - right-click a red (`LOW`) paragraph segment to open 3 rewrite options
  - or highlight a block (multi-line allowed) and right-click to request block rewrites
  - highlighted-block rewrites round selection to full lines
  - if selection extends into unscored text, only scored overlap is rewritten
  - popup shows approximate B impact per option (exact B requires `Analyze`)
  - options are sorted by expected B increase (more human-like first)
  - choose option with mouse or keyboard `1`/`2`/`3`, or `Quit`
  - chosen rewrite is inserted as an edit (yellow), with prior backgrounds preserved by prior line status
  - B score is intentionally not auto-recomputed; status marks it stale until next `Analyze`
- Preview selection mirroring:
  - when a block is selected in the left pane, right preview mirrors the same line range
  - selected preview lines show LOW/HIGH/neutral background styles

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

## Planned Next Major Feature

The next major planned feature is chunk-aware GUI analysis for very large files that cannot be analyzed in one pass.

Planned behavior (not yet implemented):

- First `Analyze` computes the first analyzable chunk (bounded by current token/memory limits).
- After the first successful chunk analysis, an `Analyze Next` button appears when unscored text remains.
- `Analyze Next` analyzes the next chunk and updates the last covered line.
- `Analyze Next` remains visible until all chunks are analyzed.
- Status-bar `B` should reflect the active chunk, selected by this priority:
  - chunk containing the current selected line,
  - chunk containing the current selected text block,
  - otherwise chunk containing most lines in the visible window.
- Pressing `Analyze` should analyze the active chunk (not always chunk 1).
- Rewrite suggestions for a red line or highlighted block should compute approximate B-impact for that requested block in the active chunk context.

## Safety / Responsible Use

Use outputs as probabilistic signals in a broader review workflow.  
Do not use this tool as a sole basis for punitive or high-stakes decisions.

## Appendix: GPTZero vs Binoculars (What Is Publicly Known)

This appendix summarizes publicly available information about GPTZero and compares it to the Binoculars method implemented in this repo.

### 1) What is known about how GPTZero works

Public GPTZero materials describe an evolving detector stack:

- GPTZero’s initial release (January 2023) emphasized `perplexity` and `burstiness` as core signals.
- Current GPTZero docs describe a probabilistic, sentence-level and document-level deep-learning detector that does not rely only on perplexity/burstiness.
- GPTZero publicly states its production system combines multiple components and outputs trinary sentence labels (`human`, `mixed`, `AI`) with confidence/uncertainty handling.

What is not public in full detail:

- exact model architecture(s),
- exact training data composition,
- post-processing and thresholding internals,
- adversarial hardening implementation details.

So, GPTZero is partially documented publicly, but the complete detector internals are proprietary.

### 2) How Binoculars differs

Binoculars is much more mechanistically explicit:

- It is defined by transparent equations over two related language models:
  - observer `logPPL`,
  - observer-vs-performer `logXPPL`,
  - ratio `B = logPPL / logXPPL`.
- It is a zero-shot detection approach (no target-model-specific training required for the detector itself).
- The paper reports strong low-FPR performance, including detection of over 90% generated samples at 0.01% FPR across studied settings.
- The paper also reports head-to-head comparisons where Binoculars outperformed GPTZero in their 2023-timeframe evaluation setup.

### 3) Why Binoculars can be in the same league as GPTZero

From currently available evidence, it is reasonable to say Binoculars can be in the same competitive tier, with caveats:

- The published Binoculars results show strong discrimination at very low false-positive rates, which is a key deployment criterion.
- The mechanism is model-agnostic in the zero-shot sense and can generalize to unseen generators when assumptions hold.
- The approach is inspectable and reproducible (equations + open implementation path), which helps calibration and operational trust.

However, a careful statement is still required:

- “As robust as GPTZero” is context-dependent and should be validated on your own domain, document lengths, and attack/perturbation conditions.
- The Binoculars paper itself notes important limits (for example, degraded recall in some low-resource language settings, and no guarantee against motivated adversarial evasion).
- Independent benchmark work also indicates that many detectors can degrade under perturbation, so robustness claims should always be treated as empirical and ongoing.

Practical conclusion:

- Binoculars is credibly “same-league” with commercial detectors in several reported settings, especially when low-FPR behavior is prioritized.
- You should still run periodic, domain-specific benchmark checks (including perturbed/paraphrased text) before making strong operational claims.

### 4) Sources

- Binoculars paper (arXiv): https://arxiv.org/abs/2401.12070
- Binoculars paper in repo: `background/2401.12070v3.pdf`
- GPTZero technology page: https://gptzero.me/technology
- GPTZero FAQ (method overview): https://gptzero.me/faqs/how-does-ai-detection-work
- GPTZero original launch note (perplexity/burstiness framing): https://gptzero.me/news/first-release-of-gptzero-for-educators-january-3-2023
- RAID benchmark (robustness context): https://arxiv.org/abs/2406.07958
