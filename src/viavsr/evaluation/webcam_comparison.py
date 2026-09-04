"""Paired inference comparison for one raw webcam recording."""

from __future__ import annotations

import hashlib
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from viavsr.inference import (
    DEFAULT_BEAM_SIZE,
    DEFAULT_CTC_WEIGHT,
    load_model_assets_config,
    load_vietnamese_avsr_assets,
    recognize_prepared_av,
)
from viavsr.inference.errors import InferenceError, ModelAssetsError
from viavsr.inference.recognition import DecoderName, InferenceMode
from viavsr.inference.reporting import redact_secrets, write_json_report
from viavsr.preprocessing import (
    FANFaceLandmarker,
    MediaInputError,
    PreparedAVInput,
    export_aligned_mouth_roi_video,
    export_mouth_roi_display_video,
    load_face_tracking_quality_policy,
    prepare_mouth_roi_media,
    probe_av_media,
    save_face_tracking_artifacts,
    track_face_landmarks,
)
from viavsr.preprocessing.face_tracking import DEFAULT_DETECTION_MAX_SIZE, DeviceRequest
from viavsr.preprocessing.media import TARGET_FRAME_RATE

from .error_rates import evaluate_transcript

PAIRED_WEBCAM_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PairedWebcamArtifactPaths:
    """Compact public and transient paths for one paired comparison."""

    run_directory: Path
    work_directory: Path
    face_track: Path
    face_tracking_report: Path
    inference_mouth_roi: Path
    mouth_roi_display: Path
    report: Path

    @classmethod
    def for_media(
        cls, media_path: Path, output_root: Path
    ) -> PairedWebcamArtifactPaths:
        run_directory = output_root.expanduser().resolve() / media_path.stem
        work_directory = run_directory / ".work"
        return cls(
            run_directory=run_directory,
            work_directory=work_directory,
            face_track=work_directory / "face_track.npz",
            face_tracking_report=work_directory / "face_tracking.json",
            inference_mouth_roi=work_directory / "inference_mouth96.mp4",
            mouth_roi_display=run_directory / "mouth_roi.mp4",
            report=run_directory / "comparison_report.json",
        )

    def prepare(self) -> None:
        """Invalidate only artifacts owned by this exact comparison run."""
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.cleanup_work()
        self.work_directory.mkdir(parents=True)
        self.mouth_roi_display.unlink(missing_ok=True)
        self.report.unlink(missing_ok=True)

    def cleanup_work(self) -> None:
        """Delete the known transient directory after verifying its parent."""
        work = self.work_directory.resolve()
        if work.parent != self.run_directory.resolve():
            raise ValueError("Comparison work directory escaped its run directory.")
        shutil.rmtree(work, ignore_errors=True)

    def public_artifacts(self) -> dict[str, str]:
        artifacts = {"report": str(self.report)}
        if self.mouth_roi_display.is_file():
            artifacts["mouth_roi"] = str(self.mouth_roi_display)
        return artifacts


def _validate_options(
    *,
    decoder: DecoderName,
    beam_size: int,
    ctc_weight: float,
    max_duration_seconds: float,
    frame_rate: int,
    max_detection_size: int,
    reference_text: str,
) -> None:
    if decoder not in {"ctc_greedy", "joint_beam_search"}:
        raise ValueError(f"Unsupported decoder: {decoder!r}.")
    if beam_size <= 0:
        raise ValueError("beam_size must be greater than zero.")
    if not 0.0 <= ctc_weight <= 1.0:
        raise ValueError("ctc_weight must be between zero and one.")
    if max_duration_seconds <= 0 or frame_rate <= 0 or max_detection_size <= 0:
        raise ValueError("Duration, frame rate, and detection size must be positive.")
    if not reference_text.strip():
        raise ValueError("reference_text must not be empty.")


def _tensor_sha256(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _prepared_fingerprints(prepared: PreparedAVInput) -> dict[str, str]:
    availability = prepared.visual_availability
    if availability is None:
        raise ValueError("Paired webcam comparison requires visual_availability.")
    return {
        "video_tensor_sha256": _tensor_sha256(prepared.videos),
        "audio_tensor_sha256": _tensor_sha256(prepared.audios),
        "visual_availability_sha256": _tensor_sha256(availability),
    }


def _run_condition(
    *,
    condition: str,
    inference_mode: InferenceMode,
    assets: Any,
    prepared: PreparedAVInput,
    reference_text: str,
    decoder: DecoderName,
    beam_size: int,
    ctc_weight: float,
) -> dict[str, Any]:
    result = recognize_prepared_av(
        assets,
        prepared,
        decoder=decoder,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        inference_mode=inference_mode,
    )
    result_payload = result.to_dict()
    result_payload.pop("token_ids", None)
    if not result_payload.get("visual_input_used", True):
        result_payload.pop("visual_coverage", None)
        result_payload.pop("visual_masked_frames", None)
    evaluation = evaluate_transcript(reference_text, result.transcript)
    return {
        "condition": condition,
        "result": result_payload,
        "evaluation": evaluation.to_dict(),
    }


def _comparison_summary(conditions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    corrupted = conditions["corrupted_av"]["evaluation"]
    gated = conditions["interval_gated"]["evaluation"]
    audio_only = conditions["audio_only"]["evaluation"]
    wers = {
        "corrupted_av": corrupted["wer"],
        "interval_gated": gated["wer"],
        "audio_only": audio_only["wer"],
    }
    best_wer = min(wers.values())
    return {
        "best_wer": best_wer,
        "best_conditions": [name for name, value in wers.items() if value == best_wer],
        "corrupted_minus_interval_gated": {
            "wer": corrupted["wer"] - gated["wer"],
            "cer": corrupted["cer"] - gated["cer"],
        },
        "audio_only_minus_interval_gated": {
            "wer": audio_only["wer"] - gated["wer"],
            "cer": audio_only["cer"] - gated["cer"],
        },
    }


def run_paired_webcam_comparison(
    *,
    config_path: Path,
    media_path: Path,
    output_root: Path,
    reference_text: str,
    tracking_device: DeviceRequest = "auto",
    decoder: DecoderName = "joint_beam_search",
    beam_size: int = DEFAULT_BEAM_SIZE,
    ctc_weight: float = DEFAULT_CTC_WEIGHT,
    max_duration_seconds: float = 15.0,
    frame_rate: int = TARGET_FRAME_RATE,
    max_detection_size: int = DEFAULT_DETECTION_MAX_SIZE,
    confidence_threshold: float | None = None,
) -> dict[str, Any]:
    """Compare three modes while sharing tracking, tensors, mask, and model."""

    started = time.perf_counter()
    resolved_media = media_path.expanduser().resolve()
    resolved_config = config_path.expanduser().resolve()
    paths = PairedWebcamArtifactPaths.for_media(resolved_media, output_root)
    payload: dict[str, Any] = {
        "schema_version": PAIRED_WEBCAM_REPORT_SCHEMA_VERSION,
        "status": "running",
        "stage": "initialization",
        "request": {
            "config_path": str(resolved_config),
            "media_path": str(resolved_media),
            "reference_text": reference_text,
            "tracking_device": tracking_device,
            "decoder": decoder,
            "beam_size": beam_size,
            "ctc_weight": ctc_weight,
            "max_duration_seconds": max_duration_seconds,
            "frame_rate": frame_rate,
            "max_detection_size": max_detection_size,
            "confidence_threshold": confidence_threshold,
        },
        "timings_seconds": {},
    }
    stage = "output_initialization"

    try:
        paths.prepare()
        stage = "configuration"
        _validate_options(
            decoder=decoder,
            beam_size=beam_size,
            ctc_weight=ctc_weight,
            max_duration_seconds=max_duration_seconds,
            frame_rate=frame_rate,
            max_detection_size=max_detection_size,
            reference_text=reference_text,
        )
        quality_policy = load_face_tracking_quality_policy(resolved_config)
        if confidence_threshold is not None:
            quality_policy = replace(
                quality_policy,
                min_detection_confidence=confidence_threshold,
            )

        stage = "media_preflight"
        stage_started = time.perf_counter()
        raw_media = probe_av_media(resolved_media)
        payload["raw_media"] = raw_media.to_dict()
        payload["timings_seconds"][stage] = time.perf_counter() - stage_started
        if raw_media.duration_seconds > max_duration_seconds:
            raise MediaInputError(
                f"Media duration {raw_media.duration_seconds:.3f}s exceeds the "
                f"configured maximum of {max_duration_seconds:.3f}s."
            )

        stage = "face_tracking_backend"
        stage_started = time.perf_counter()
        landmarker = FANFaceLandmarker(
            device=tracking_device,
            confidence_threshold=quality_policy.min_detection_confidence,
        )
        payload["timings_seconds"][stage] = time.perf_counter() - stage_started

        stage = "face_tracking"
        stage_started = time.perf_counter()
        sequence = track_face_landmarks(
            resolved_media,
            landmarker=landmarker,
            frame_rate=frame_rate,
            policy=quality_policy,
            max_detection_size=max_detection_size,
        )
        if not bool(sequence.mouth_visible.any()):
            raise MediaInputError(
                "Paired comparison requires at least one usable mouth frame."
            )
        face_report = save_face_tracking_artifacts(
            sequence,
            artifact_path=paths.face_track,
            report_path=paths.face_tracking_report,
        )
        face_report.pop("artifact_path", None)
        face_report.pop("report_path", None)
        face_report["tracking_seconds"] = time.perf_counter() - stage_started
        payload["face_tracking"] = face_report
        payload["timings_seconds"][stage] = face_report["tracking_seconds"]

        stage = "mouth_roi_display"
        stage_started = time.perf_counter()
        try:
            display = export_mouth_roi_display_video(
                resolved_media,
                paths.mouth_roi_display,
                track_path=paths.face_track,
            )
        except MediaInputError as exc:
            payload.setdefault("warnings", []).append("mouth_roi_display_unavailable")
            payload["mouth_roi_display"] = {
                "status": "unavailable",
                "error": {
                    "type": type(exc).__name__,
                    "message": redact_secrets(str(exc)),
                },
            }
        else:
            display_payload = display.to_dict()
            display_payload.pop("output_path", None)
            display_payload.pop("track_path", None)
            payload["mouth_roi_display"] = {
                **display_payload,
                "status": "passed",
                "artifact_role": "ui_visualization_only",
                "used_for_inference": False,
            }
        payload["timings_seconds"][stage] = time.perf_counter() - stage_started

        stage = "mouth_roi"
        stage_started = time.perf_counter()
        export_aligned_mouth_roi_video(
            resolved_media,
            paths.face_track,
            paths.inference_mouth_roi,
            require_quality_passed=False,
        )
        payload["timings_seconds"][stage] = time.perf_counter() - stage_started

        stage = "av_preprocessing"
        stage_started = time.perf_counter()
        prepared = prepare_mouth_roi_media(
            paths.inference_mouth_roi,
            max_duration_seconds=max_duration_seconds,
            visual_availability=sequence.mouth_visible,
        )
        payload["timings_seconds"][stage] = time.perf_counter() - stage_started
        prepared_media = prepared.metadata.to_dict()
        prepared_media.pop("path", None)
        payload["shared_input"] = {
            "single_preprocessing_pass": True,
            "media": prepared_media,
            "shapes": prepared.shape_report(),
            "visual_coverage": float(sequence.mouth_visible.mean()),
            "visual_masked_frames": int((~sequence.mouth_visible).sum()),
            "fingerprints": _prepared_fingerprints(prepared),
        }

        stage = "model_loading"
        stage_started = time.perf_counter()
        model_config = load_model_assets_config(resolved_config)
        assets = load_vietnamese_avsr_assets(model_config)
        payload["model"] = assets.report.to_dict()
        payload["timings_seconds"][stage] = time.perf_counter() - stage_started

        stage = "paired_inference"
        stage_started = time.perf_counter()
        conditions: dict[str, dict[str, Any]] = {}
        pairs: tuple[tuple[str, InferenceMode], ...] = (
            ("corrupted_av", "audio_visual_corrupted"),
            ("interval_gated", "audio_visual_interval_gated"),
            ("audio_only", "audio_only_experimental"),
        )
        for condition, inference_mode in pairs:
            condition_started = time.perf_counter()
            conditions[condition] = _run_condition(
                condition=condition,
                inference_mode=inference_mode,
                assets=assets,
                prepared=prepared,
                reference_text=reference_text,
                decoder=decoder,
                beam_size=beam_size,
                ctc_weight=ctc_weight,
            )
            conditions[condition]["wall_seconds"] = (
                time.perf_counter() - condition_started
            )
        payload["timings_seconds"][stage] = time.perf_counter() - stage_started
        payload["conditions"] = conditions
        payload["comparison"] = _comparison_summary(conditions)
        payload["status"] = "passed"
        payload["stage"] = "complete"
    except (
        InferenceError,
        MediaInputError,
        ModelAssetsError,
        OSError,
        ValueError,
    ) as exc:
        payload.update(
            {
                "status": "failed",
                "stage": stage,
                "error": {
                    "type": type(exc).__name__,
                    "message": redact_secrets(str(exc)),
                },
            }
        )
    finally:
        payload["timings_seconds"]["total"] = time.perf_counter() - started
        paths.cleanup_work()
        payload["artifacts"] = paths.public_artifacts()
        write_json_report(paths.report, payload)

    return payload
