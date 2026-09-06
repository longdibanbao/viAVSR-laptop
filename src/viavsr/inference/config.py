from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from .errors import ConfigurationError

DeviceName = Literal["auto", "cpu", "cuda"]
DTypeName = Literal["float32", "float16", "bfloat16"]


@dataclass(frozen=True)
class ModelAssetsConfig:
    """Configuration required to load Vietnamese AVSR model assets."""

    repository_id: str
    revision: str
    cache_dir: Path
    tokenizer_model_path: Path
    tokenizer_units_path: Path
    device: DeviceName = "auto"
    dtype: DTypeName = "float32"


def _repository_root(config_path: Path) -> Path:
    for directory in (config_path.parent, *config_path.parents):
        if (directory / "pyproject.toml").is_file():
            return directory
    raise ConfigurationError(
        f"Could not find repository root above config: {config_path}",
        stage="config",
    )


def _required(mapping: dict, key: str, section: str) -> object:
    if key not in mapping:
        raise ConfigurationError(
            f"Missing required configuration key: {section}.{key}",
            stage="config",
        )
    return mapping[key]


def load_model_assets_config(
    path: Path | str, *, project_root: Path | None = None
) -> ModelAssetsConfig:
    """Load YAML configuration and resolve paths from the repository root."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(
            f"Configuration file does not exist: {config_path}", stage="config"
        )

    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(
            f"Could not read configuration: {exc}", stage="config"
        ) from exc

    if not isinstance(payload, dict):
        raise ConfigurationError("Configuration root must be a mapping.", stage="config")
    model = payload.get("model")
    tokenizer = payload.get("tokenizer")
    if not isinstance(model, dict) or not isinstance(tokenizer, dict):
        raise ConfigurationError(
            "Configuration must contain model and tokenizer mappings.",
            stage="config",
        )

    root = project_root.resolve() if project_root else _repository_root(config_path)
    device = str(model.get("device", "auto"))
    dtype = str(model.get("dtype", "float32"))
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError(
            "model.device must be 'auto', 'cpu', or 'cuda'.", stage="config"
        )
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ConfigurationError(
            "model.dtype must be float32, float16, or bfloat16.", stage="config"
        )

    def resolve(value: object) -> Path:
        candidate = Path(str(value)).expanduser()
        return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

    return ModelAssetsConfig(
        repository_id=str(_required(model, "repository_id", "model")),
        revision=str(_required(model, "revision", "model")),
        cache_dir=resolve(_required(model, "cache_dir", "model")),
        tokenizer_model_path=resolve(
            _required(tokenizer, "model_path", "tokenizer")
        ),
        tokenizer_units_path=resolve(
            _required(tokenizer, "units_path", "tokenizer")
        ),
        device=device,  # type: ignore[arg-type]
        dtype=dtype,  # type: ignore[arg-type]
    )
