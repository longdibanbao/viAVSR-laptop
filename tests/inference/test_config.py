from pathlib import Path

import pytest

from viavsr.inference.config import load_model_assets_config
from viavsr.inference.errors import ConfigurationError


def test_load_config_resolves_paths_from_repository_root(tmp_path: Path):
    root = tmp_path / "project"
    config_dir = root / "configs"
    config_dir.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    path = config_dir / "assets.yaml"
    path.write_text(
        """
model:
  repository_id: owner/model
  revision: abc123
  cache_dir: .cache/models
  device: cpu
  dtype: float32
tokenizer:
  model_path: assets/unigram2048.model
  units_path: assets/unigram2048_units.txt
""",
        encoding="utf-8",
    )

    config = load_model_assets_config(path)

    assert config.repository_id == "owner/model"
    assert config.revision == "abc123"
    assert config.cache_dir == root / ".cache/models"
    assert config.tokenizer_model_path == root / "assets/unigram2048.model"
    assert config.device == "cpu"


@pytest.mark.parametrize("key,value", [("device", "tpu"), ("dtype", "float64")])
def test_load_config_rejects_unsupported_runtime_values(
    tmp_path: Path, key: str, value: str
):
    path = tmp_path / "assets.yaml"
    runtime = {"device": "cpu", "dtype": "float32", key: value}
    path.write_text(
        f"""
model:
  repository_id: owner/model
  revision: abc123
  cache_dir: cache
  device: {runtime['device']}
  dtype: {runtime['dtype']}
tokenizer:
  model_path: tokenizer.model
  units_path: units.txt
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError):
        load_model_assets_config(path, project_root=tmp_path)


def test_load_config_accepts_auto_device(tmp_path: Path):
    path = tmp_path / "assets.yaml"
    path.write_text(
        """
model:
  repository_id: owner/model
  revision: abc123
  cache_dir: cache
  device: auto
  dtype: float32
tokenizer:
  model_path: tokenizer.model
  units_path: units.txt
""",
        encoding="utf-8",
    )

    config = load_model_assets_config(path, project_root=tmp_path)

    assert config.device == "auto"
