"""Validated YAML experiment settings."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .core import DEFAULT_MODEL


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default=DEFAULT_MODEL, min_length=1)
    revision: str | None = None
    quantization: str | None = "4-bit MLX checkpoint"


class TrainingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["lora"] = "lora"
    iterations: int = Field(default=160, gt=0, le=5000)
    batch_size: int = Field(default=1, gt=0, le=64)
    gradient_accumulation_steps: int = Field(default=4, gt=0, le=128)
    learning_rate: float = Field(default=1e-5, gt=0, le=1)
    num_layers: int = Field(default=8, gt=0, le=256)
    max_sequence_length: int = Field(default=1024, gt=0, le=32768)
    mask_prompt: bool = True
    seed: int = Field(default=42, ge=0)
    smoke_iterations: int = Field(default=5, gt=0, le=100)
    resume_adapter_file: str | None = None


class DecodingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(default=0, ge=0, le=2)
    max_tokens: int = Field(default=256, gt=0, le=4096)
    seed: int = Field(default=42, ge=0)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelSettings = Field(default_factory=ModelSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    decoding: DecodingSettings = Field(default_factory=DecodingSettings)


def load_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return ExperimentConfig.model_validate(raw)
