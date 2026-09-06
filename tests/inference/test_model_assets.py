from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from viavsr.inference.config import ModelAssetsConfig
from viavsr.inference.errors import DeviceUnavailableError, VocabularyMismatchError
from viavsr.inference.model_assets import (
    _validate_device,
    _validate_model_placement,
    collect_vocabulary_dimensions,
    load_vietnamese_avsr_assets,
    run_tokenizer_sanity_checks,
)


class FakeTokenizer:
    sentencepiece_vocabulary_size = 2048
    units_vocabulary_size = 2055
    asr_vocabulary_size = 2057
    unknown_token_id = 1
    model_sha256 = "model-hash"
    units_sha256 = "units-hash"

    def encode(self, text: str) -> list[int]:
        return [1] if any(character.isdigit() for character in text) else [2]

    def decode(self, token_ids: list[int]) -> str:
        if 1 in token_ids:
            return "năm <unk>"
        return self.current_text


def _model(odim: int = 2057) -> torch.nn.Module:
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(odim=odim)
            self.avsr = torch.nn.Module()
            self.avsr.odim = odim
            self.avsr.ctc = torch.nn.Module()
            self.avsr.ctc.ctc_lo = torch.nn.Linear(1, odim)
            self.avsr.decoder = torch.nn.Module()
            self.avsr.decoder.embed = torch.nn.Sequential(torch.nn.Embedding(odim, 1))
            self.avsr.decoder.output_layer = torch.nn.Linear(1, odim)

    return FakeModel()


def _config(tmp_path: Path, device: str = "cpu") -> ModelAssetsConfig:
    return ModelAssetsConfig(
        repository_id="owner/model",
        revision="immutable-revision",
        cache_dir=tmp_path / "cache",
        tokenizer_model_path=tmp_path / "unigram2048.model",
        tokenizer_units_path=tmp_path / "unigram2048_units.txt",
        device=device,  # type: ignore[arg-type]
        dtype="float32",
    )


def test_cuda_request_fails_without_fallback(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(DeviceUnavailableError, match="CUDA was requested"):
        load_vietnamese_avsr_assets(_config(tmp_path, "cuda"))


@pytest.mark.parametrize(
    ("cuda_available", "expected_device"),
    [(False, "cpu"), (True, "cuda")],
)
def test_auto_device_selects_best_available_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    expected_device: str,
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)

    device = _validate_device(_config(tmp_path, "auto"))

    assert device.type == expected_device


def test_load_passes_pinned_arguments_and_sets_eval(tmp_path: Path, monkeypatch):
    tokenizer = FakeTokenizer()
    loaded_model = _model()

    class FakeModelClass:
        arguments = None

        @classmethod
        def from_pretrained(cls, repository_id, **kwargs):
            cls.arguments = (repository_id, kwargs)
            return loaded_model

    def fake_tokenizer(*args, **kwargs):
        return tokenizer

    def fake_sanity(instance):
        return []

    monkeypatch.setattr(
        "viavsr.inference.model_assets.VietnameseSentencePieceTokenizer",
        fake_tokenizer,
    )
    monkeypatch.setattr(
        "viavsr.inference.model_assets.run_tokenizer_sanity_checks", fake_sanity
    )
    monkeypatch.setattr(
        "viavsr.inference.model_assets._load_model_class", lambda: FakeModelClass
    )

    assets = load_vietnamese_avsr_assets(_config(tmp_path))

    repository_id, arguments = FakeModelClass.arguments
    assert repository_id == "owner/model"
    assert arguments["revision"] == "immutable-revision"
    assert arguments["cache_dir"] == str(tmp_path / "cache")
    assert arguments["torch_dtype"] == torch.float32
    assert assets.model.training is False
    assert next(assets.model.parameters()).device.type == "cpu"
    assert assets.report.vocabulary_compatible is True


def test_cuda_device_without_index_accepts_cuda_zero_parameter():
    parameter = SimpleNamespace(
        device=torch.device("cuda:0"),
        dtype=torch.float32,
        is_floating_point=lambda: True,
    )
    model = SimpleNamespace(parameters=lambda: [parameter], training=False)

    _validate_model_placement(model, torch.device("cuda"), torch.float32)


@pytest.mark.parametrize(
    "field",
    ["config", "model", "ctc", "decoder_embedding", "decoder_output"],
)
def test_each_model_vocabulary_mismatch_fails(field: str):
    tokenizer = FakeTokenizer()
    model = _model()
    if field == "config":
        model.config.odim = 2056
    elif field == "model":
        model.avsr.odim = 2056
    elif field == "ctc":
        model.avsr.ctc.ctc_lo = torch.nn.Linear(1, 2056)
    elif field == "decoder_embedding":
        model.avsr.decoder.embed = torch.nn.Sequential(torch.nn.Embedding(2056, 1))
    else:
        model.avsr.decoder.output_layer = torch.nn.Linear(1, 2056)

    with pytest.raises(VocabularyMismatchError):
        collect_vocabulary_dimensions(model, tokenizer)  # type: ignore[arg-type]


def test_number_case_is_unsupported_without_failing_smoke_test():
    tokenizer = FakeTokenizer()
    sentences = ["xin chào", "năm 2026"]
    results = []
    for sentence in sentences:
        tokenizer.current_text = sentence
        results.extend(run_tokenizer_sanity_checks(tokenizer, [sentence]))  # type: ignore[arg-type]
    assert [result.status for result in results] == ["passed", "unsupported"]
