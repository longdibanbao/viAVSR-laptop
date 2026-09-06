from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from .config import ModelAssetsConfig
from .errors import (
    DeviceUnavailableError,
    ModelAssetsError,
    VocabularyMismatchError,
)
from .schemas import (
    LoadedAVSRAssets,
    ModelAssetsReport,
    RoundTripCase,
    VocabularyDimensions,
)
from .tokenizer import (
    TOKENIZER_REVISION,
    VietnameseSentencePieceTokenizer,
    normalize_tokenizer_text,
)

MODEL_IMPLEMENTATION_REVISION = "51107b66864c42687638a00df8dd398ec9210872"
TOKENIZER_REPOSITORY = "nguyenvulebinh/viCocktail"
SANITY_SENTENCES = (
    "hôm nay thời tiết rất đẹp",
    "tôi đang học máy học",
    "xin chào các bạn",
    "xin xin chào",
    "TÔI Đang Học",
    "năm 2026",
)


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _validate_device(config: ModelAssetsConfig) -> torch.device:
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise DeviceUnavailableError(
            "CUDA was requested but torch.cuda.is_available() is false. "
            "Use device: auto for a deliberate CPU fallback or device: cpu "
            "for an intentional CPU-only run.",
            stage="device",
        )
    return torch.device(config.device)


def _load_model_class() -> type:
    from .vendor.avsrcocktail import AVHubertAVSR

    return AVHubertAVSR


def _dimension(value: Any, label: str) -> int:
    if not isinstance(value, int):
        raise VocabularyMismatchError(
            f"Could not determine integer vocabulary dimension for {label}: {value!r}",
            stage="vocabulary",
        )
    return value


def collect_vocabulary_dimensions(
    model: Any, tokenizer: VietnameseSentencePieceTokenizer
) -> VocabularyDimensions:
    """Read every model output dimension used by CTC/attention decoding."""
    try:
        avsr = model.avsr
        values = VocabularyDimensions(
            sentencepiece_pieces=tokenizer.sentencepiece_vocabulary_size,
            units_entries=tokenizer.units_vocabulary_size,
            asr_tokenizer=tokenizer.asr_vocabulary_size,
            config_odim=_dimension(model.config.odim, "config.odim"),
            model_odim=_dimension(avsr.odim, "model.avsr.odim"),
            ctc_output=_dimension(avsr.ctc.ctc_lo.out_features, "CTC output"),
            decoder_embedding=_dimension(
                avsr.decoder.embed[0].num_embeddings, "decoder embedding"
            ),
            decoder_output=_dimension(
                avsr.decoder.output_layer.out_features, "decoder output"
            ),
        )
    except ModelAssetsError:
        raise
    except (AttributeError, IndexError, TypeError) as exc:
        raise VocabularyMismatchError(
            f"Released model does not expose the expected CTC/attention heads: {exc}",
            stage="vocabulary",
        ) from exc

    model_dimensions = (
        values.config_odim,
        values.model_odim,
        values.ctc_output,
        values.decoder_embedding,
        values.decoder_output,
    )
    if any(value != values.asr_tokenizer for value in model_dimensions):
        raise VocabularyMismatchError(
            "Vocabulary mismatch: ASR tokenizer has "
            f"{values.asr_tokenizer} entries, model dimensions are {model_dimensions}.",
            stage="vocabulary",
        )
    return values


def run_tokenizer_sanity_checks(
    tokenizer: VietnameseSentencePieceTokenizer,
    sentences: Iterable[str] = SANITY_SENTENCES,
) -> list[RoundTripCase]:
    """Run supported Vietnamese round trips and identify unsupported numbers."""
    results: list[RoundTripCase] = []
    for sentence in sentences:
        normalized = normalize_tokenizer_text(sentence)
        token_ids = tokenizer.encode(sentence)
        decoded = tokenizer.decode(token_ids)
        contains_unknown = tokenizer.unknown_token_id in token_ids
        if contains_unknown:
            status = "unsupported"
        elif decoded == normalized:
            status = "passed"
        else:
            status = "failed"
        results.append(
            RoundTripCase(
                input_text=sentence,
                normalized_text=normalized,
                token_ids=token_ids,
                decoded_text=decoded,
                status=status,
                contains_unknown=contains_unknown,
            )
        )
    ordinary_failures = [
        result
        for result in results
        if not any(character.isdigit() for character in result.normalized_text)
        and result.status != "passed"
    ]
    if ordinary_failures:
        failed_inputs = ", ".join(repr(result.input_text) for result in ordinary_failures)
        raise ModelAssetsError(
            f"Vietnamese tokenizer round-trip failed for: {failed_inputs}",
            stage="round_trip",
        )
    return results


def _validate_model_placement(model: Any, device: torch.device, dtype: torch.dtype) -> None:
    parameters = list(model.parameters())
    misplaced = [
        parameter.device
        for parameter in parameters
        if parameter.device.type != device.type
    ]
    if misplaced:
        raise ModelAssetsError(
            f"Model parameters were not all placed on {device}.", stage="model"
        )
    wrong_dtype = [
        parameter.dtype
        for parameter in parameters
        if parameter.is_floating_point() and parameter.dtype != dtype
    ]
    if wrong_dtype:
        raise ModelAssetsError(
            f"Model floating-point parameters do not all use {dtype}.", stage="model"
        )
    if model.training:
        raise ModelAssetsError("Model did not enter eval mode.", stage="model")


def load_vietnamese_avsr_assets(
    config: ModelAssetsConfig,
) -> LoadedAVSRAssets:
    """Load and validate the released Vietnamese AVSR model and tokenizer."""
    device = _validate_device(config)
    dtype = _torch_dtype(config.dtype)
    tokenizer = VietnameseSentencePieceTokenizer(
        config.tokenizer_model_path, config.tokenizer_units_path
    )
    round_trip_cases = run_tokenizer_sanity_checks(tokenizer)
    config.cache_dir.mkdir(parents=True, exist_ok=True)

    model_class = _load_model_class()
    try:
        model = model_class.from_pretrained(
            config.repository_id,
            revision=config.revision,
            cache_dir=str(config.cache_dir),
            torch_dtype=dtype,
        )
        model = model.to(device)
        model.eval()
    except ModelAssetsError:
        raise
    except Exception as exc:
        raise ModelAssetsError(
            f"Could not load model {config.repository_id}@{config.revision}: {exc}",
            stage="model",
        ) from exc

    _validate_model_placement(model, device, dtype)
    vocabulary = collect_vocabulary_dimensions(model, tokenizer)
    report = ModelAssetsReport(
        status="passed",
        repository_id=config.repository_id,
        model_revision=config.revision,
        model_implementation_revision=MODEL_IMPLEMENTATION_REVISION,
        tokenizer_repository=TOKENIZER_REPOSITORY,
        tokenizer_revision=TOKENIZER_REVISION,
        tokenizer_model_sha256=tokenizer.model_sha256,
        tokenizer_units_sha256=tokenizer.units_sha256,
        model_class=f"{model.__class__.__module__}.{model.__class__.__name__}",
        device=str(device),
        dtype=config.dtype,
        eval_mode=not model.training,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        vocabulary=vocabulary,
        vocabulary_compatible=True,
        round_trip_cases=round_trip_cases,
    )
    return LoadedAVSRAssets(model=model, tokenizer=tokenizer, report=report)
