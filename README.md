# FineTuning Regression Tester

**See what your model gained. See what it forgot.**

FineTuning Regression Tester compares a local language model before and after fine-tuning on the same held-out examples. It reports target-task gains, preserved behavior, pass-to-fail regressions, and exact cases that changed. It is an independent evaluation layer around MLX-LM, not another trainer.

## What it does

- Generates deterministic synthetic product-extraction train, validation, and held-out sets, plus 80 general capability, 12 format, 20 missing-data, 10 unsupported-capability, and 10 regression-sentinel cases.
- Evaluates strict JSON, schema, field correctness, arithmetic, instruction following, transformations, extraction, classification, and abstention with deterministic graders.
- Compares paired outcomes as preserved success, regression, improvement, preserved failure, or inconclusive.
- Exports machine-readable JSON/JSONL, CSV, Markdown, and offline HTML, with an optional local Streamlit/Plotly dashboard.
- Integrates optional MLX-LM base and LoRA adapter inference and bounded training on Apple Silicon.
- Runs data generation, validation, comparison, report generation, and explicit fixture demo without MLX.

## Requirements and installation

Python 3.11+ is required. Offline analysis runs on macOS, Linux, and Windows. Real inference/training requires macOS on Apple Silicon and MLX-LM. The recommended model is [`mlx-community/Qwen2.5-1.5B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-4bit). Model weights are not included in this repository. Evaluation and training use a local snapshot; fetch it explicitly with `finetuning-regression-tester model download` first.

```bash
git clone https://github.com/ranajahanzaib/finetuning-regression-tester.git
cd finetuning-regression-tester
uv sync --extra dev --extra dashboard --extra models
# On Apple Silicon, add MLX support:
uv sync --extra dev --extra dashboard --extra models --extra mlx
```

Install the optional model helper and explicitly prefetch the configured snapshot before inference or training:

```bash
finetuning-regression-tester model download --config configs/product-extraction.yaml
```

## Quick start: offline demo

```bash
finetuning-regression-tester doctor
finetuning-regression-tester data generate --output data/product
finetuning-regression-tester data validate --dataset data/product
finetuning-regression-tester demo --output demo-results
finetuning-regression-tester dashboard --comparison demo-results/comparison.json --demo
```

The persistent **DEMO DATA** banner identifies fictional fixture outputs. No model was loaded, trained, or measured by demo mode. Demo reports and CSV exports are created alongside the fixture comparison.

## Full local experiment

```bash
finetuning-regression-tester data generate --output data/product
finetuning-regression-tester data validate --dataset data/product
finetuning-regression-tester model download --config configs/product-extraction.yaml

# Evaluate exactly the same target cases before and after adaptation.
finetuning-regression-tester baseline --cases data/product/evaluation_cases.jsonl \
  --output runs/manual/baseline_results.jsonl --config configs/product-extraction.yaml
finetuning-regression-tester train --dataset data/product --adapter runs/manual/adapter \
  --config configs/product-extraction.yaml --smoke
finetuning-regression-tester evaluate --cases data/product/evaluation_cases.jsonl \
  --adapter runs/manual/adapter --output runs/manual/adapted_results.jsonl \
  --config configs/product-extraction.yaml
finetuning-regression-tester compare --baseline runs/manual/baseline_results.jsonl \
  --adapted runs/manual/adapted_results.jsonl --output runs/manual/comparison.json
finetuning-regression-tester report --comparison runs/manual/comparison.json --output runs/manual/report
finetuning-regression-tester dashboard --comparison runs/manual/comparison.json
```

The one-command orchestration generates data and runs baseline → training → adapted evaluation → comparison → report sequentially:

```bash
finetuning-regression-tester run --config configs/product-extraction.yaml --output runs
# Short bounded end-to-end check:
finetuning-regression-tester run --config configs/product-extraction.yaml --output runs --smoke
```

Use `--dry-run` on supported commands to inspect intended work. For ordinary tests, FineTuning Regression Tester never downloads model weights. The default full training preset is 160 iterations; `--smoke` uses the configured five-iteration smoke preset. These are starting points, not performance guarantees. Interrupted evaluations keep completed case results and resume by case ID. Completed adapter directories are not silently reused.

## Product schema and normalization

The product record has nullable string fields `product_name`, `brand`, `price`, `currency`, `color`, and `size`, plus a nullable positive integer `quantity`. Price values are decimal strings (for example, `"79.99"`), never binary floats. The synthetic English examples map an explicit `$` to `USD`. FineTuning Regression Tester does not infer currency from locale or convert currencies. Unknown fields are JSON `null`; additional keys and commentary fail strict extraction grading.

## Interpreting the comparison

“Forgetting” is an operational label for an observed baseline-pass/adapted-fail transition. It is undefined when baseline successes are zero. A regression does not establish permanent loss of an underlying ability. Paired improvements and regressions can cancel in aggregate accuracy, so reports show both counts and case-level outputs. Small suite counts are exploratory and do not support broad claims about a model. See [methodology](docs/methodology.md), [reproducibility](docs/reproducibility.md), and [limitations](docs/limitations.md).

## Artifacts

An experiment directory contains `manifest.json`, frozen evaluation cases, raw baseline/adapted JSONL results, `comparison.json`, `capabilities.json`, `comparisons.csv`, `report.md`, and a self-contained `report.html`. Model revision, decoding settings, data hashes, durations, and completed stages are recorded when available. No absent result is replaced by a made-up score.

## Architecture

`core.py` handles portable schemas, deterministic data, grading, and analysis. `mlx.py` isolates optional model inference and invokes the documented MLX-LM LoRA interface. `report.py` exports offline artifacts. `dashboard.py` renders persisted comparisons. The CLI composes these layers. See [architecture](docs/architecture.md).

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest
```

CI runs on Linux and does not attempt MLX training. Real training checks are hardware-dependent and intentionally excluded from the standard test suite. Optional MLX behavior is based on the [MLX-LM LoRA documentation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md).

## Model attribution and license

FineTuning Regression Tester is MIT licensed. The default Qwen model is distributed under the model's own [Hugging Face model card and license](https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-4bit); check its terms before use or redistribution. FineTuning Regression Tester does not redistribute model weights.

## Roadmap

Future work may add repeated-sampling reliability, confidence intervals, multi-checkpoint comparisons, imported evaluation sets, and regression checks for CI. These are not claims of current support. Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md). Report vulnerabilities using the [Security Policy](.github/SECURITY.md).
