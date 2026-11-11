"""Optional MLX-LM inference and bounded LoRA training integration."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .core import write_json


def mlx_available() -> tuple[bool, str]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return (
            False,
            "MLX requires macOS on Apple Silicon; this is not an Apple Silicon environment.",
        )
    try:
        import mlx_lm  # noqa: F401
    except ImportError as exc:
        try:
            installed_version = version("mlx-lm")
        except PackageNotFoundError:
            return False, "mlx-lm is not installed. Install with: uv sync --extra mlx"
        if "No Metal device" in str(exc):
            return (
                False,
                f"mlx-lm {installed_version} is installed, but this process cannot access a Metal GPU: {exc}",
            )
        return False, f"mlx-lm {installed_version} could not initialize: {exc}"
    return True, "Apple Silicon and mlx-lm are available."


class MLXRunner:
    def __init__(
        self,
        model_id: str,
        adapter: Path | None = None,
        max_tokens: int = 256,
        revision: str | None = None,
        temperature: float = 0,
        seed: int = 42,
    ):
        self.model_id, self.adapter, self.max_tokens = model_id, adapter, max_tokens
        self.revision = revision
        self.temperature = temperature
        self.seed = seed
        self.model: Any = None
        self.tokenizer: Any = None
        self.resolved_model: str | None = None
        self.resolved_revision: str | None = None

    def resolve(self) -> str:
        if self.resolved_model is None:
            self.resolved_model = resolve_model(self.model_id, self.revision)
            resolved = Path(self.resolved_model)
            if resolved.parent.name == "snapshots":
                self.resolved_revision = resolved.name
            else:
                self.resolved_revision = self.revision
        return self.resolved_model

    def load(self) -> None:
        ok, message = mlx_available()
        if not ok:
            raise RuntimeError(message)
        from mlx_lm import load

        kwargs = {"adapter_path": str(self.adapter)} if self.adapter else {}
        self.model, self.tokenizer = load(self.resolve(), **kwargs)

    def generate(self, prompt: str) -> str:
        if self.model is None:
            self.load()
        import mlx.core as mx
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        messages = [{"role": "user", "content": prompt}]
        if getattr(self.tokenizer, "chat_template", None):
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_text = prompt
        mx.random.seed(self.seed)
        return generate(
            self.model,
            self.tokenizer,
            prompt=prompt_text,
            max_tokens=self.max_tokens,
            sampler=make_sampler(temp=self.temperature),
        )

    def unload(self) -> None:
        self.model = self.tokenizer = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except (ImportError, AttributeError):
            pass

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "mlx-lm",
            "model_id": self.model_id,
            "revision": self.resolved_revision or self.revision,
            "resolved_model": self.resolved_model,
            "adapter": str(self.adapter) if self.adapter else None,
            "temperature": self.temperature,
            "seed": self.seed,
        }


def resolve_model(model_id: str, revision: str | None = None, *, download: bool = False) -> str:
    local_path = Path(model_id).expanduser()
    if local_path.exists():
        return str(local_path.resolve())
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_files_only=not download,
    )


def train(
    config: ExperimentConfig,
    dataset: Path,
    adapter: Path,
    dry_run: bool = False,
    smoke: bool = False,
    resume_adapter_file: Path | None = None,
) -> dict[str, Any]:
    if not (dataset / "train.jsonl").exists() or not (dataset / "valid.jsonl").exists():
        raise ValueError(f"Expected train.jsonl and valid.jsonl under {dataset}")
    if adapter.exists() and any(adapter.iterdir()) and resume_adapter_file is None:
        raise FileExistsError(
            f"Adapter directory already contains files: {adapter}; use a new experiment directory"
        )
    if resume_adapter_file is not None and not resume_adapter_file.is_file():
        raise FileNotFoundError(f"Resume adapter file does not exist: {resume_adapter_file}")
    executable = shutil.which("mlx_lm.lora")
    prefix = [executable] if executable else [sys.executable, "-m", "mlx_lm", "lora"]
    args = [
        *prefix,
        "--model",
        config.model.id,
        "--train",
        "--data",
        str(dataset),
        "--adapter-path",
        str(adapter),
        "--fine-tune-type",
        "lora",
        "--iters",
        str(config.training.smoke_iterations if smoke else config.training.iterations),
        "--batch-size",
        str(config.training.batch_size),
        "--num-layers",
        str(config.training.num_layers),
        "--learning-rate",
        str(config.training.learning_rate),
        "--grad-accumulation-steps",
        str(config.training.gradient_accumulation_steps),
        "--max-seq-length",
        str(config.training.max_sequence_length),
        "--seed",
        str(config.training.seed),
    ]
    args.append("--grad-checkpoint")
    if config.training.mask_prompt:
        args.append("--mask-prompt")
    if resume_adapter_file:
        args.extend(["--resume-adapter-file", str(resume_adapter_file)])
    iterations = config.training.smoke_iterations if smoke else config.training.iterations
    report_every = min(10, iterations)
    args.extend(
        [
            "--steps-per-report",
            str(report_every),
            "--steps-per-eval",
            str(report_every),
            "--save-every",
            str(min(50, iterations)),
        ]
    )
    if dry_run:
        return {
            "command": args,
            "dry_run": True,
            "iterations": config.training.smoke_iterations if smoke else config.training.iterations,
            "model_revision": config.model.revision,
        }
    ok, message = mlx_available()
    if not ok:
        raise RuntimeError(message)
    try:
        resolved = resolve_model(config.model.id, config.model.revision)
    except Exception as exc:
        raise RuntimeError(
            f"Could not resolve local model snapshot {config.model.id}: {type(exc).__name__}: {exc}. "
            "Download it explicitly with: finetuning-regression-tester model download --config CONFIG.yaml"
        ) from exc
    args[args.index(config.model.id)] = resolved
    adapter.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    log_path = adapter.parent / "training.log"
    adapter.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("a" if resume_adapter_file else "w") as log:
            subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
    except KeyboardInterrupt:
        write_json(
            adapter.parent / "training_status.json",
            {
                "status": "interrupted",
                "elapsed_seconds": time.monotonic() - start,
                "log": str(log_path),
            },
        )
        raise
    except subprocess.CalledProcessError as exc:
        log_text = log_path.read_text(errors="replace").casefold()
        memory_failure = any(
            phrase in log_text
            for phrase in (
                "out of memory",
                "failed to allocate",
                "memory allocation",
                "metal gpu memory",
            )
        )
        write_json(
            adapter.parent / "training_status.json",
            {
                "status": "failed",
                "failure_kind": "memory" if memory_failure else "training",
                "returncode": exc.returncode,
                "elapsed_seconds": time.monotonic() - start,
                "log": str(log_path),
            },
        )
        raise RuntimeError(
            f"MLX-LM training failed; inspect {log_path} for memory or training errors"
        ) from exc
    status = {
        "status": "completed",
        "elapsed_seconds": time.monotonic() - start,
        "command": args,
        "log": str(log_path),
    }
    write_json(adapter.parent / "training_status.json", status)
    return status
