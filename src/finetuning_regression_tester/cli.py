"""Command-line interface for the complete local experiment workflow."""

from __future__ import annotations

import json
import logging
import platform
import shutil
import sys
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from difflib import SequenceMatcher
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .config import ExperimentConfig, load_config
from .core import (
    Case,
    Product,
    Result,
    compare,
    digest,
    eval_cases,
    generate_dataset,
    read_jsonl,
    write_json,
    write_jsonl,
)
from .dashboard import launch
from .mlx import MLXRunner, mlx_available, resolve_model
from .mlx import train as mlx_train
from .report import export_reports

NEAR_DUPLICATE_THRESHOLD = 0.92

app = typer.Typer(
    help="FineTuning Regression Tester — paired behavioral regression analysis for fine-tuned models.",
    no_args_is_help=True,
)


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", help="Show diagnostic log output."),
) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING)


data_app = typer.Typer(help="Generate and validate deterministic datasets.", no_args_is_help=True)
model_app = typer.Typer(
    help="Inspect and explicitly download local model snapshots.", no_args_is_help=True
)
app.add_typer(data_app, name="data")
app.add_typer(model_app, name="model")


def _config(path: Optional[Path]) -> ExperimentConfig:
    if not path:
        return ExperimentConfig()
    try:
        return load_config(path)
    except Exception as exc:
        raise typer.BadParameter(f"Invalid configuration {path}: {exc}") from exc


@app.command()
def doctor() -> None:
    """Inspect Python, operating system, and optional MLX support."""
    ok, message = mlx_available()
    typer.echo(
        f"FineTuning Regression Tester {__version__}\nPython: {sys.version.split()[0]} ({'supported' if sys.version_info >= (3, 11) else 'unsupported; requires 3.11+'})\nOS: {platform.platform()}\nApple Silicon MLX: {message}"
    )
    for dependency in (
        "pydantic",
        "typer",
        "yaml",
        "mlx_lm",
        "huggingface_hub",
        "streamlit",
        "plotly",
    ):
        typer.echo(
            f"Dependency {dependency}: {'installed' if find_spec(dependency) else 'not installed (optional where applicable)'}"
        )
    try:
        import os

        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        typer.echo(f"Physical memory: {pages * page_size / 1024**3:.1f} GiB")
    except (ValueError, OSError, AttributeError):
        typer.echo("Physical memory: unavailable")
    typer.echo("Offline analysis: available")
    if not ok:
        typer.echo("Training/inference: unavailable in this environment")


@data_app.command("generate")
def data_generate(
    output: Path = typer.Option(Path("data/product"), "--output"),
    train: int = 200,
    valid: int = 40,
    test: int = 60,
    seed: int = 42,
) -> None:
    """Generate product fine-tuning data and independent evaluation cases."""
    try:
        manifest = generate_dataset(output, train, valid, test, seed)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        f"Generated deterministic dataset in {output}\nHashes: {json.dumps(manifest['hashes'], sort_keys=True)}"
    )


@data_app.command("validate")
def data_validate(dataset: Path = typer.Option(..., "--dataset")) -> None:
    """Validate JSONL structure, duplicates, contamination, and hashes."""
    names = ["train.jsonl", "valid.jsonl", "test.jsonl"]
    try:
        splits = {name: read_jsonl(dataset / name) for name in names}
        norm = {
            name: {
                " ".join(
                    r.get("messages", [{}])[0]
                    .get("content", "")
                    .split("\n\n", 1)[-1]
                    .casefold()
                    .split()
                )
                for r in rows
            }
            for name, rows in splits.items()
        }
        errors = []
        for name, rows in splits.items():
            if not rows:
                errors.append(f"{name} is empty")
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                overlap = norm[a] & norm[b]
                if overlap:
                    errors.append(
                        f"normalized input contamination between {a} and {b}: {len(overlap)} records"
                    )
        cases = read_jsonl(dataset / "target_cases.jsonl")
        evaluation_inputs = {" ".join(str(r.get("input", "")).casefold().split()) for r in cases}
        near_duplicates = []
        for name in ("train.jsonl", "valid.jsonl"):
            contamination = evaluation_inputs & norm[name]
            if contamination:
                errors.append(
                    f"target evaluation contamination with {name}: {len(contamination)} records"
                )
            duplicates = [value for value, count in Counter(norm[name]).items() if count > 1]
            if duplicates:
                errors.append(f"{name} contains {len(duplicates)} normalized duplicate prompts")
        all_prompts = [(name, prompt) for name in names for prompt in norm[name]]
        for index, (left_split, left) in enumerate(all_prompts):
            for right_split, right in all_prompts[index + 1 :]:
                if (
                    left_split != right_split
                    and left != right
                    and SequenceMatcher(None, left, right).ratio() >= NEAR_DUPLICATE_THRESHOLD
                ):
                    near_duplicates.append(
                        {
                            "left_split": left_split,
                            "right_split": right_split,
                            "similarity": round(SequenceMatcher(None, left, right).ratio(), 3),
                        }
                    )
        ids = [r.get("case_id") for r in cases]
        if len(ids) != len(set(ids)):
            errors.append("duplicate target evaluation case IDs")
        # Verify each training assistant completion and schema.
        for split, rows in splits.items():
            for i, row in enumerate(rows):
                try:
                    messages = row["messages"]
                    expected = json.loads(messages[-1]["content"])
                    Product.model_validate(expected)
                except Exception as exc:
                    errors.append(f"{split} row {i}: invalid MLX chat/schema record ({exc})")
                    break
        for case in cases:
            try:
                Case.model_validate(case)
                if not case.get("metadata") or not case.get("grading"):
                    raise ValueError("metadata and grading must both be present and non-empty")
                if not case.get("case_id") or not case.get("suite") or not case.get("capability"):
                    raise ValueError("case_id, suite, and capability are required")
            except Exception as exc:
                errors.append(f"invalid target evaluation case: {exc}")
                break
        # Product identities in training and validation must never appear in
        # the held-out target cases. The test.jsonl rows intentionally mirror
        # held-out cases for inspection, but they are not passed to training.
        split_entities: dict[str, set[str]] = {}
        for split_name, rows in splits.items():
            entities = set()
            for row in rows:
                try:
                    entities.add(json.loads(row["messages"][-1]["content"]).get("product_name"))
                except (KeyError, IndexError, json.JSONDecodeError, TypeError):
                    continue
            split_entities[split_name] = entities - {None}
        target_entities = {
            str(case.get("expected", {}).get("product_name"))
            for case in cases
            if isinstance(case.get("expected"), dict) and case["expected"].get("product_name")
        }
        for split_name in ("train.jsonl", "valid.jsonl"):
            overlap = split_entities.get(split_name, set()) & target_entities
            if overlap:
                errors.append(
                    f"product identity contamination between {split_name} and target evaluation: {len(overlap)} identities"
                )
        manifest_path = dataset / "dataset_manifest.json"
        if not manifest_path.exists():
            errors.append("dataset_manifest.json is missing")
        else:
            manifest = json.loads(manifest_path.read_text())
            for name, rows in splits.items():
                if manifest.get("hashes", {}).get(name.removesuffix(".jsonl")) != digest(rows):
                    errors.append(f"{name} hash differs from dataset_manifest.json")
            retention = read_jsonl(dataset / "retention_cases.jsonl")
            combined = read_jsonl(dataset / "evaluation_cases.jsonl")
            combined_ids: list[str] = []
            for row in combined:
                try:
                    parsed_case = Case.model_validate(row)
                    combined_ids.append(parsed_case.case_id)
                    if not parsed_case.metadata or not parsed_case.grading:
                        errors.append(
                            f"evaluation case {parsed_case.case_id} is missing metadata or grading configuration"
                        )
                except Exception as exc:
                    errors.append(f"invalid evaluation case: {exc}")
            if len(combined_ids) != len(set(combined_ids)):
                errors.append("duplicate case IDs in combined evaluation suite")
            expected_hashes = manifest.get("hashes", {})
            for key, rows in (
                ("target_cases", cases),
                ("retention_cases", retention),
                ("evaluation_cases", combined),
            ):
                if expected_hashes.get(key) != digest(rows):
                    errors.append(f"{key} hash differs from dataset_manifest.json")
        if errors:
            raise ValueError("\n".join(errors))
        hashes = {name: digest(rows) for name, rows in splits.items()}
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        f"Valid: {sum(map(len, splits.values()))} training rows, {len(cases)} target cases; hashes {json.dumps(hashes, sort_keys=True)}"
    )
    if near_duplicates:
        typer.echo(
            f"Warning: {len(near_duplicates)} suspicious cross-split near-duplicate pairs (similarity >= {NEAR_DUPLICATE_THRESHOLD:.2f})"
        )
        for pair in near_duplicates[:10]:
            typer.echo(
                f"  {pair['left_split']} ↔ {pair['right_split']}: similarity {pair['similarity']:.3f}"
            )


@model_app.command("download")
def model_download(
    config: Optional[Path] = typer.Option(None, "--config"), dry_run: bool = False
) -> None:
    """Download/cache the configured model separately from evaluation or training."""
    cfg = _config(config)
    local_path = Path(cfg.model.id).expanduser()
    if local_path.exists():
        typer.echo(f"Model is already local: {local_path.resolve()}")
        return
    if dry_run:
        typer.echo(f"Would download {cfg.model.id} at revision {cfg.model.revision or 'latest'}")
        return
    try:
        snapshot = resolve_model(cfg.model.id, cfg.model.revision, download=True)
    except ImportError as exc:
        raise typer.BadParameter(
            "Model download support is missing. Install with: uv sync --extra models"
        ) from exc
    except Exception as exc:
        raise typer.BadParameter(
            f"Could not download {cfg.model.id}: {type(exc).__name__}: {exc}"
        ) from exc
    typer.echo(f"Cached model snapshot: {snapshot}")


def _load_cases(path: Path) -> list[Case]:
    try:
        cases = [Case.model_validate(row) for row in read_jsonl(path)]
    except (OSError, ValueError, TypeError) as exc:
        raise typer.BadParameter(f"Cannot load evaluation cases from {path}: {exc}") from exc
    if not cases:
        raise typer.BadParameter(f"Evaluation suite is empty: {path}")
    return cases


def _cases_provenance(path: Path) -> dict[str, object]:
    manifest_path = path.parent / "dataset_manifest.json"
    if not manifest_path.is_file():
        return {"dataset_hashes": None, "dataset_seed": None, "dataset_version": None}
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"Invalid dataset manifest {manifest_path}: {exc}") from exc
    return {
        "dataset_hashes": manifest.get("hashes"),
        "dataset_seed": manifest.get("seed"),
        "dataset_version": manifest.get("generator_version"),
    }


def _environment_provenance() -> dict[str, str | None]:
    try:
        mlx_version: str | None = version("mlx-lm")
    except PackageNotFoundError:
        mlx_version = None
    return {
        "finetuning_regression_tester_version": __version__,
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "mlx_lm_version": mlx_version,
    }


def _runner(config: ExperimentConfig, adapter: Path | None = None) -> MLXRunner:
    return MLXRunner(
        config.model.id,
        adapter,
        config.decoding.max_tokens,
        config.model.revision,
        config.decoding.temperature,
        config.decoding.seed,
    )


def _prepare_runner(config: ExperimentConfig, adapter: Path | None = None) -> MLXRunner:
    available, message = mlx_available()
    if not available:
        raise typer.BadParameter(message)
    runner = _runner(config, adapter)
    try:
        runner.resolve()
    except Exception as exc:
        raise typer.BadParameter(
            f"Model snapshot is not available locally ({type(exc).__name__}: {exc}). "
            f"Download it explicitly with: finetuning-regression-tester model download --config CONFIG.yaml"
        ) from exc
    return runner


@app.command()
def baseline(
    cases: Path = typer.Option(..., "--cases"),
    output: Path = typer.Option(..., "--output"),
    config: Optional[Path] = typer.Option(None, "--config"),
    dry_run: bool = False,
) -> None:
    """Evaluate the unchanged base model against a JSONL case suite."""
    cfg = _config(config)
    suite = _load_cases(cases)
    if dry_run:
        typer.echo(
            f"Would evaluate {len(suite)} cases with base model {cfg.model.id}; no model loaded."
        )
        return
    runner = _prepare_runner(cfg)
    config_path = output.with_suffix(output.suffix + ".config.json")
    eval_config = {
        "model": cfg.model.model_dump(),
        "decoding": cfg.decoding.model_dump(),
        "cases_hash": digest([c.model_dump() for c in suite]),
        "adapter": None,
        "resolved_model_revision": runner.resolved_revision,
        **_cases_provenance(cases),
        "environment": _environment_provenance(),
    }
    if config_path.exists() and json.loads(config_path.read_text()) != eval_config:
        raise typer.BadParameter("Cannot resume: saved evaluation configuration differs")
    write_json(config_path, eval_config)
    rows = eval_cases(suite, runner, output)
    if not any(row.passed is not None for row in rows):
        raise typer.BadParameter(
            f"No baseline response could be graded; inconclusive outputs were saved to {output}"
        )
    typer.echo(f"Saved {len(rows)} baseline results to {output}")


@app.command()
def train(
    dataset: Path = typer.Option(..., "--dataset"),
    adapter: Path = typer.Option(..., "--adapter"),
    config: Optional[Path] = typer.Option(None, "--config"),
    dry_run: bool = False,
    smoke: bool = False,
    resume: Optional[Path] = typer.Option(None, "--resume-adapter-file"),
) -> None:
    """Train a bounded MLX-LM LoRA adapter on Apple Silicon."""
    try:
        cfg = _config(config)
        data_validate(dataset)
        resume_path = resume or (
            Path(cfg.training.resume_adapter_file) if cfg.training.resume_adapter_file else None
        )
        status = mlx_train(cfg, dataset, adapter, dry_run, smoke, resume_path)
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(status, indent=2))


@app.command()
def evaluate(
    cases: Path = typer.Option(..., "--cases"),
    adapter: Path = typer.Option(..., "--adapter"),
    output: Path = typer.Option(..., "--output"),
    config: Optional[Path] = typer.Option(None, "--config"),
    dry_run: bool = False,
) -> None:
    """Evaluate an adapter against the exact same suite used for baseline."""
    cfg = _config(config)
    suite = _load_cases(cases)
    if not adapter.exists():
        raise typer.BadParameter(f"Adapter does not exist: {adapter}")
    config_path = output.with_suffix(output.suffix + ".config.json")
    if dry_run:
        typer.echo(f"Would evaluate {len(suite)} cases using {cfg.model.id} plus {adapter}")
        return
    runner = _prepare_runner(cfg, adapter)
    eval_config = {
        "model": cfg.model.model_dump(),
        "decoding": cfg.decoding.model_dump(),
        "cases_hash": digest([c.model_dump() for c in suite]),
        "adapter": str(adapter.resolve()),
        "training_config": cfg.training.model_dump(),
        "resolved_model_revision": runner.resolved_revision,
        **_cases_provenance(cases),
        "environment": _environment_provenance(),
    }
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        compatible = {key: saved.get(key) == eval_config.get(key) for key in eval_config}
        if not all(compatible.values()):
            raise typer.BadParameter("Cannot resume: saved evaluation configuration differs")
    write_json(config_path, eval_config)
    rows = eval_cases(suite, runner, output)
    if not any(row.passed is not None for row in rows):
        raise typer.BadParameter(
            f"No adapted response could be graded; inconclusive outputs were saved to {output}"
        )
    typer.echo(f"Saved {len(rows)} adapted results to {output}")


@app.command("compare")
def compare_cmd(
    baseline_file: Path = typer.Option(..., "--baseline"),
    adapted_file: Path = typer.Option(..., "--adapted"),
    output: Path = typer.Option(..., "--output"),
) -> None:
    """Compare paired results and classify behavioral transitions."""
    try:
        base = [Result.model_validate(r) for r in read_jsonl(baseline_file)]
        adapted = [Result.model_validate(r) for r in read_jsonl(adapted_file)]
    except (OSError, ValueError, TypeError) as exc:
        raise typer.BadParameter(f"Cannot load evaluation results: {exc}") from exc
    if not base or not adapted:
        raise typer.BadParameter("Cannot compare empty evaluation results")
    bcfg = baseline_file.with_suffix(baseline_file.suffix + ".config.json")
    acfg = adapted_file.with_suffix(adapted_file.suffix + ".config.json")
    if not bcfg.exists() or not acfg.exists():
        raise typer.BadParameter(
            "Evaluation configuration sidecars are required to verify paired comparability"
        )
    try:
        bdata, adata = json.loads(bcfg.read_text()), json.loads(acfg.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"Cannot read evaluation configuration sidecars: {exc}") from exc
    comparison_keys = (
        "model",
        "decoding",
        "cases_hash",
        "resolved_model_revision",
        "dataset_hashes",
        "dataset_seed",
        "environment",
    )
    if any(bdata.get(key) != adata.get(key) for key in comparison_keys):
        raise typer.BadParameter(
            "Evaluation configurations differ: model snapshot, decoding, dataset, environment, or case hashes do not match"
        )
    try:
        report = compare(base, adapted)
    except ValueError as exc:
        raise typer.BadParameter(f"Results cannot be paired: {exc}") from exc
    report["model"] = bdata.get("model")
    report["adapter"] = adata.get("adapter")
    report["resolved_model_revision"] = bdata.get("resolved_model_revision")
    report["dataset_hashes"] = bdata.get("dataset_hashes")
    report["dataset_seed"] = bdata.get("dataset_seed")
    report["dataset_version"] = bdata.get("dataset_version")
    report["environment"] = bdata.get("environment")
    report["decoding"] = bdata.get("decoding")
    report["training_config"] = adata.get("training_config")
    report["evaluation_cases_hash"] = bdata.get("cases_hash")
    write_json(output, report)
    typer.echo(json.dumps(report["summary"], indent=2))


@app.command()
def report(
    comparison: Path = typer.Option(..., "--comparison"),
    output: Path = typer.Option(..., "--output"),
) -> None:
    """Export offline Markdown, HTML, CSV, and capability JSON reports."""
    try:
        paths = export_reports(comparison, output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise typer.BadParameter(f"Cannot create report from {comparison}: {exc}") from exc
    typer.echo("\n".join(map(str, paths)))


@app.command()
def dashboard(comparison: Path = typer.Option(..., "--comparison"), demo: bool = False) -> None:
    """Open the local interactive dashboard for a comparison artifact."""
    try:
        launch(comparison, demo)
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def demo(output: Path = typer.Option(Path("demo-results"), "--output")) -> None:
    """Create explicitly fictional fixture results, exports, and a demo dashboard input."""
    if output.exists() and any(output.iterdir()):
        raise typer.BadParameter(f"Output directory already contains files: {output}")
    output.mkdir(parents=True, exist_ok=True)
    fixture_dir = Path(__file__).parent / "fixtures" / "demo"
    fixture_files = (
        "baseline_results.jsonl",
        "adapted_results.jsonl",
        "comparison.json",
        "manifest.json",
    )
    for filename in fixture_files:
        source = fixture_dir / filename
        if not source.is_file():
            raise typer.BadParameter(f"Packaged demo fixture is missing: {filename}")
        shutil.copyfile(source, output / filename)
    export_reports(output / "comparison.json", output)
    typer.echo(
        f"Created explicitly fictional DEMO DATA under {output}; nothing was trained or evaluated."
    )
    typer.echo(
        f"Launch with: finetuning-regression-tester dashboard --comparison {output / 'comparison.json'} --demo"
    )


@app.command()
def run(
    dataset: Path = typer.Option(Path("data/product"), "--dataset"),
    output: Path = typer.Option(Path("runs"), "--output"),
    config: Optional[Path] = typer.Option(None, "--config"),
    experiment_id: Optional[str] = typer.Option(None, "--experiment-id"),
    dry_run: bool = False,
    smoke: bool = False,
) -> None:
    """Run the complete local experiment and persist every stage."""
    cfg = _config(config)
    eid = experiment_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    run_dir = output / eid
    if run_dir.exists():
        raise typer.BadParameter(f"Experiment directory already exists: {run_dir}")
    if dry_run:
        typer.echo(
            f"Would create {run_dir}; generate dataset, evaluate baseline, train LoRA, evaluate adapter, compare, report."
        )
        return
    if dataset.exists() and not (dataset / "dataset_manifest.json").exists():
        raise typer.BadParameter(
            f"Dataset directory exists without an FineTuning Regression Tester manifest: {dataset}; choose another path"
        )
    if not (dataset / "dataset_manifest.json").exists():
        generate_dataset(dataset)
    data_validate(dataset)
    dataset_metadata = json.loads((dataset / "dataset_manifest.json").read_text())
    run_dir.mkdir(parents=True)
    manifest = {
        "experiment_id": eid,
        "created_at": datetime.now(UTC).isoformat(),
        "finetuning_regression_tester_version": __version__,
        "grading_version": "1",
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "model": cfg.model.model_dump(),
        "training": cfg.training.model_dump(),
        "dataset_manifest": str(dataset / "dataset_manifest.json"),
        "dataset_hashes": dataset_metadata.get("hashes", {}),
        "dataset_version": dataset_metadata.get("generator_version"),
        "decoding": cfg.decoding.model_dump(),
        "stages": {
            "data": "complete",
            "baseline": "pending",
            "training": "pending",
            "adapted_evaluation": "pending",
            "comparison": "pending",
            "report": "pending",
        },
        "mlx_lm_version": None,
        "resolved_model_revision": None,
    }
    try:
        manifest["mlx_lm_version"] = version("mlx-lm")
    except PackageNotFoundError:
        manifest["mlx_lm_version"] = None
    write_json(run_dir / "manifest.json", manifest)
    # Stages run sequentially to keep only one model resident at a time.
    try:
        target = _load_cases(dataset / "target_cases.jsonl")
        retention = _load_cases(dataset / "retention_cases.jsonl")
        all_cases = target + retention
        write_jsonl(run_dir / "evaluation_cases.jsonl", [c.model_dump() for c in all_cases])
        manifest["evaluation_cases_hash"] = digest([c.model_dump() for c in all_cases])
        baseline_runner = _prepare_runner(cfg)
        manifest["resolved_model_revision"] = baseline_runner.resolved_revision
        write_json(run_dir / "manifest.json", manifest)
        stage_start = time.monotonic()
        base = eval_cases(all_cases, baseline_runner, run_dir / "baseline_results.jsonl")
        manifest["baseline_elapsed_seconds"] = time.monotonic() - stage_start
        if not any(result.passed is not None for result in base):
            raise RuntimeError(
                "Baseline inference produced no gradeable responses; results are inconclusive"
            )
        manifest["stages"]["baseline"] = "complete"
        write_json(run_dir / "manifest.json", manifest)
        adapter = run_dir / "adapter"
        status = mlx_train(cfg, dataset, adapter, smoke=smoke)
        manifest["training_elapsed_seconds"] = status["elapsed_seconds"]
        manifest["stages"]["training"] = "complete"
        write_json(run_dir / "manifest.json", manifest)
        stage_start = time.monotonic()
        adapted_runner = _prepare_runner(cfg, adapter)
        if adapted_runner.resolved_revision != baseline_runner.resolved_revision:
            raise RuntimeError("Base and adapted evaluations resolved different model revisions")
        adapted = eval_cases(all_cases, adapted_runner, run_dir / "adapted_results.jsonl")
        manifest["adapted_evaluation_elapsed_seconds"] = time.monotonic() - stage_start
        if not any(result.passed is not None for result in adapted):
            raise RuntimeError(
                "Adapted inference produced no gradeable responses; comparison is incomplete"
            )
        manifest["stages"]["adapted_evaluation"] = "complete"
        write_json(run_dir / "manifest.json", manifest)
        comp = compare(base, adapted)
        comp["model"] = cfg.model.model_dump()
        comp["resolved_model_revision"] = baseline_runner.resolved_revision
        comp["environment"] = {
            "finetuning_regression_tester_version": manifest[
                "finetuning_regression_tester_version"
            ],
            "python": manifest["python"],
            "os": manifest["os"],
            "mlx_lm_version": manifest["mlx_lm_version"],
        }
        comp["adapter"] = str(adapter)
        comp["training_config"] = cfg.training.model_dump()
        comp["decoding"] = cfg.decoding.model_dump()
        comp["dataset_hashes"] = manifest["dataset_hashes"]
        comp["timings"] = {
            "baseline_evaluation_seconds": manifest["baseline_elapsed_seconds"],
            "training_seconds": manifest["training_elapsed_seconds"],
            "adapted_evaluation_seconds": manifest["adapted_evaluation_elapsed_seconds"],
        }
        write_json(run_dir / "comparison.json", comp)
        manifest["stages"]["comparison"] = "complete"
        write_json(run_dir / "manifest.json", manifest)
        export_reports(run_dir / "comparison.json", run_dir)
        manifest["stages"]["report"] = "complete"
        write_json(run_dir / "manifest.json", manifest)
        typer.echo(f"Experiment complete: {run_dir}")
    except Exception as exc:
        manifest["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(run_dir / "manifest.json", manifest)
        raise typer.BadParameter(
            f"Experiment {eid} stopped; manifest records completed stages: {exc}"
        ) from exc


if __name__ == "__main__":
    app()
