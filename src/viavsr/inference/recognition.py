from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Literal

import torch

from viavsr.preprocessing import PreparedAVInput
from viavsr.preprocessing.media import AUDIO_FEATURE_DIM, MODEL_VIDEO_SIZE

from .decoding import (
    DEFAULT_BEAM_SIZE,
    DEFAULT_CTC_WEIGHT,
    decode_joint_ctc_attention,
)
from .errors import InferenceError
from .schemas import InferenceResult, LoadedAVSRAssets

DecoderName = Literal["ctc_greedy", "joint_beam_search"]
InferenceMode = Literal[
    "audio_visual",
    "audio_visual_corrupted",
    "audio_visual_interval_gated",
    "audio_only_experimental",
    "audio_only_fallback",
]


def collapse_ctc_predictions(
    frame_token_ids: Sequence[int], *, blank_id: int = 0
) -> list[int]:
    """Collapse repeated CTC frame predictions and remove blank tokens."""
    collapsed: list[int] = []
    previous: int | None = None
    for token_id in frame_token_ids:
        if token_id != previous and token_id != blank_id:
            collapsed.append(token_id)
        previous = token_id
    return collapsed


def _validate_prepared_input(prepared: PreparedAVInput) -> None:
    videos = prepared.videos
    audios = prepared.audios
    if (
        videos.ndim != 5
        or videos.shape[1] != 1
        or tuple(videos.shape[3:]) != (MODEL_VIDEO_SIZE, MODEL_VIDEO_SIZE)
    ):
        raise InferenceError(
            f"videos must have shape [B, 1, T, 88, 88], got {list(videos.shape)}.",
            stage="inference_input",
        )
    if audios.ndim != 3 or audios.shape[1] != AUDIO_FEATURE_DIM:
        raise InferenceError(
            f"audios must have shape [B, 104, T], got {list(audios.shape)}.",
            stage="inference_input",
        )
    if videos.shape[0] != 1 or audios.shape[0] != 1:
        raise InferenceError(
            "Inference currently supports batch size one.",
            stage="inference_input",
        )
    if videos.shape[2] != audios.shape[2]:
        raise InferenceError(
            "Audio/video feature lengths differ: "
            f"{audios.shape[2]} audio vs {videos.shape[2]} video frames.",
            stage="inference_input",
        )
    expected_frames = videos.shape[2]
    if prepared.video_lengths.tolist() != [expected_frames]:
        raise InferenceError(
            "video_lengths does not match the prepared video tensor.",
            stage="inference_input",
        )
    if prepared.audio_lengths.tolist() != [expected_frames]:
        raise InferenceError(
            "audio_lengths does not match the prepared audio tensor.",
            stage="inference_input",
        )
    availability = prepared.visual_availability
    if availability is not None and (
        availability.ndim != 2 or tuple(availability.shape) != (1, expected_frames)
    ):
        raise InferenceError(
            "visual_availability must have shape [1, T] matching the video.",
            stage="inference_input",
        )


def _model_device_and_dtype(model: Any) -> tuple[torch.device, torch.dtype]:
    try:
        parameter = next(model.parameters())
    except (AttributeError, StopIteration) as exc:
        raise InferenceError(
            "Could not determine model device and dtype.", stage="inference"
        ) from exc
    return parameter.device, parameter.dtype


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def recognize_prepared_av(
    assets: LoadedAVSRAssets,
    prepared: PreparedAVInput,
    *,
    decoder: DecoderName = "ctc_greedy",
    beam_size: int = DEFAULT_BEAM_SIZE,
    ctc_weight: float = DEFAULT_CTC_WEIGHT,
    inference_mode: InferenceMode = "audio_visual",
) -> InferenceResult:
    """Run batch-one AV-HuBERT encoding and configurable decoding.

    Both audio-only modes exercise the released encoder's native video=None
    branch. audio_only_experimental is an explicit caller request, while
    audio_only_fallback records an automatic orchestration decision after
    visual preprocessing could not provide a trustworthy input.

    audio_visual_interval_gated requires a frame-aligned visual availability
    mask and removes unavailable visual features before fusion.

    audio_visual_corrupted applies the same frame-aligned mask only to the
    normalized video tensor, then runs the released checkpoint's ordinary
    audio-visual path without feature gating.
    """
    _validate_prepared_input(prepared)
    if decoder not in {"ctc_greedy", "joint_beam_search"}:
        raise InferenceError(
            f"Unsupported decoder: {decoder!r}.",
            stage="decoding",
        )
    if inference_mode not in {
        "audio_visual",
        "audio_visual_corrupted",
        "audio_visual_interval_gated",
        "audio_only_experimental",
        "audio_only_fallback",
    }:
        raise InferenceError(
            f"Unsupported inference mode: {inference_mode!r}.",
            stage="inference_input",
        )

    model = assets.model
    device, dtype = _model_device_and_dtype(model)
    videos = prepared.videos.to(device=device, dtype=dtype)
    audios = prepared.audios.to(device=device, dtype=dtype)
    availability = prepared.visual_availability
    if inference_mode in {
        "audio_visual_corrupted",
        "audio_visual_interval_gated",
    } and availability is None:
        raise InferenceError(
            f"{inference_mode} inference requires visual_availability.",
            stage="inference_input",
        )
    availability_device = (
        availability.to(device=device) if availability is not None else None
    )
    encoder_video = (
        videos
        if inference_mode
        in {
            "audio_visual",
            "audio_visual_corrupted",
            "audio_visual_interval_gated",
        }
        else None
    )
    if inference_mode in {
        "audio_visual_corrupted",
        "audio_visual_interval_gated",
    }:
        assert encoder_video is not None
        assert availability_device is not None
        encoder_video = encoder_video * availability_device[:, None, :, None, None]
    visual_coverage: float | None = None
    visual_masked_frames: int | None = None
    if availability_device is not None:
        visual_coverage = float(availability_device.float().mean().item())
        visual_masked_frames = int((~availability_device.bool()).sum().item())
    model.eval()
    hypothesis_score: float | None = None

    try:
        _synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            if inference_mode == "audio_visual_interval_gated":
                features, _ = model.avsr.encoder.extract_finetune(
                    {
                        "audio": audios,
                        "video": encoder_video,
                    },
                    padding_mask=None,
                    visual_availability=availability_device,
                )
            elif encoder_video is None:
                features, _ = model.avsr.encoder.extract_finetune(
                    {
                        "audio": audios,
                        "video": None,
                    },
                    padding_mask=None,
                )
            else:
                output = model.avsr.encoder(
                    input_features=audios,
                    video=encoder_video,
                )
                features = output.last_hidden_state
            if decoder == "ctc_greedy":
                log_probabilities = model.avsr.ctc.log_softmax(features)
                frame_ids = log_probabilities.argmax(dim=-1)[0].tolist()
                token_ids = collapse_ctc_predictions(frame_ids)
            else:
                hypothesis = decode_joint_ctc_attention(
                    model.avsr,
                    features[0],
                    assets.tokenizer.token_list,
                    beam_size=beam_size,
                    ctc_weight=ctc_weight,
                )
                token_ids = hypothesis.token_ids
                hypothesis_score = hypothesis.score
        _synchronize(device)
        elapsed = time.perf_counter() - started
    except InferenceError:
        raise
    except (AttributeError, RuntimeError, ValueError) as exc:
        raise InferenceError(
            f"AV-HuBERT inference failed: {exc}",
            stage="inference",
        ) from exc

    try:
        transcript = assets.tokenizer.decode(token_ids)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise InferenceError(
            f"Vietnamese tokenizer decoding failed: {exc}", stage="decoding"
        ) from exc
    return InferenceResult(
        transcript=transcript,
        token_ids=token_ids,
        decoder=decoder,
        input_video_frames=int(prepared.videos.shape[2]),
        encoder_frames=int(features.shape[1]),
        inference_seconds=elapsed,
        device=str(device),
        dtype=str(dtype).removeprefix("torch."),
        inference_mode=inference_mode,
        visual_input_used=encoder_video is not None,
        visual_coverage=visual_coverage,
        visual_masked_frames=visual_masked_frames,
        beam_size=beam_size if decoder == "joint_beam_search" else None,
        ctc_weight=ctc_weight if decoder == "joint_beam_search" else None,
        hypothesis_score=hypothesis_score,
    )
