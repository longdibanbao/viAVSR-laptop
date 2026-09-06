from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch
import yaml

from .errors import MediaInputError
from .media import TARGET_FRAME_RATE, MediaMetadata, probe_av_media

DeviceRequest = Literal["auto", "cpu", "cuda"]
LANDMARK_COUNT = 68
DEFAULT_DETECTION_MAX_SIZE = 640
MOUTH_START_INDEX = 48
MOUTH_STOP_INDEX = 68
FACE_TRACK_ARTIFACT_VERSION = 2


@dataclass(frozen=True)
class FaceTrackingQualityPolicy:
    """Thresholds that decide whether a single-speaker face track is reliable."""

    min_detection_confidence: float = 0.8
    min_landmark_confidence: float = 0.6
    min_face_area_ratio: float = 0.005
    min_mouth_landmark_confidence: float = 0.6
    min_mouth_landmarks_in_frame_ratio: float = 0.9
    min_detection_rate: float = 0.9
    max_missing_run: int = 5
    max_center_distance_ratio: float = 0.75
    min_association_iou: float = 0.1
    ambiguity_score_margin: float = 0.15
    initial_face_area_margin_ratio: float = 0.15
    max_face_area_change_ratio: float = 2.0
    max_out_of_frame_ratio: float = 0.1
    max_ambiguous_frames: int = 0
    max_edge_missing_run: int = 0
    min_detected_frames: int = 2
    max_short_visual_gap: int = 3
    min_visual_run: int = 2

    def __post_init__(self) -> None:
        probabilities = {
            "min_mouth_landmark_confidence": self.min_mouth_landmark_confidence,
            "min_mouth_landmarks_in_frame_ratio": self.min_mouth_landmarks_in_frame_ratio,
            "min_detection_confidence": self.min_detection_confidence,
            "min_landmark_confidence": self.min_landmark_confidence,
            "min_face_area_ratio": self.min_face_area_ratio,
            "min_detection_rate": self.min_detection_rate,
            "min_association_iou": self.min_association_iou,
            "initial_face_area_margin_ratio": self.initial_face_area_margin_ratio,
            "max_out_of_frame_ratio": self.max_out_of_frame_ratio,
        }
        for name, value in probabilities.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be between 0 and 1, got {value}.")
        if self.min_association_iou == 0:
            raise ValueError("min_association_iou must be greater than 0.")
        positive_reals = {
            "max_center_distance_ratio": self.max_center_distance_ratio,
            "max_face_area_change_ratio": self.max_face_area_change_ratio,
        }
        for name, value in positive_reals.items():
            minimum = 1.0 if name == "max_face_area_change_ratio" else 0.0
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or value <= minimum
            ):
                requirement = "greater than 1" if minimum else "positive"
                raise ValueError(f"{name} must be {requirement}, got {value}.")
        if (
            isinstance(self.ambiguity_score_margin, bool)
            or not isinstance(self.ambiguity_score_margin, Real)
            or not math.isfinite(float(self.ambiguity_score_margin))
            or self.ambiguity_score_margin < 0
        ):
            raise ValueError("ambiguity_score_margin must be finite and non-negative.")
        non_negative_integers = {
            "max_missing_run": self.max_missing_run,
            "max_short_visual_gap": self.max_short_visual_gap,
            "max_ambiguous_frames": self.max_ambiguous_frames,
            "max_edge_missing_run": self.max_edge_missing_run,
        }
        for name, value in non_negative_integers.items():
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if (
            isinstance(self.min_detected_frames, bool)
            or not isinstance(self.min_detected_frames, Integral)
            or self.min_detected_frames < 2
        ):
            raise ValueError("min_detected_frames must be an integer of at least 2.")
        if (
            isinstance(self.min_visual_run, bool)
            or not isinstance(self.min_visual_run, Integral)
            or self.min_visual_run < 1
        ):
            raise ValueError("min_visual_run must be a positive integer.")


@dataclass(frozen=True)
class TrackingDecision:
    """Identity-association outcome for one frame."""

    candidate: FaceCandidate | None
    status: Literal["accepted", "missing", "ambiguous"]
    reason: str
    score: float | None = None
    rejected_reasons: tuple[str, ...] = ()


def load_face_tracking_quality_policy(path: Path | str) -> FaceTrackingQualityPolicy:
    """Load and validate the face_tracking quality section from YAML."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise MediaInputError(f"Configuration file does not exist: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        section = payload["face_tracking"]
        if not isinstance(section, dict):
            raise TypeError("face_tracking must be a mapping")
        return FaceTrackingQualityPolicy(**section)
    except (
        OSError,
        UnicodeError,
        yaml.YAMLError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise MediaInputError(
            f"Could not load face-tracking quality policy from {config_path}: {exc}"
        ) from exc


@dataclass(frozen=True)
class FaceCandidate:
    """One detected face and its FAN 68-point landmark prediction."""

    bounding_box: np.ndarray
    detection_confidence: float
    landmarks: np.ndarray
    landmark_scores: np.ndarray

    def __post_init__(self) -> None:
        if self.bounding_box.shape != (4,):
            raise ValueError("bounding_box must have shape [4].")
        if self.landmarks.shape != (LANDMARK_COUNT, 2):
            raise ValueError("landmarks must have shape [68, 2].")
        if self.landmark_scores.shape != (LANDMARK_COUNT,):
            raise ValueError("landmark_scores must have shape [68].")


def mouth_landmarks_visible(
    candidate: FaceCandidate,
    *,
    frame_size: tuple[float, float],
    policy: FaceTrackingQualityPolicy,
) -> bool:
    """Return whether the detected mouth landmarks are usable in this frame."""
    frame_width, frame_height = frame_size
    mouth = candidate.landmarks[MOUTH_START_INDEX:MOUTH_STOP_INDEX]
    mouth_scores = candidate.landmark_scores[MOUTH_START_INDEX:MOUTH_STOP_INDEX]
    if (
        not np.isfinite(mouth).all()
        or not np.isfinite(mouth_scores).all()
        or float(mouth_scores.mean()) < policy.min_mouth_landmark_confidence
    ):
        return False
    in_frame = (
        (mouth[:, 0] >= 0.0)
        & (mouth[:, 0] < frame_width)
        & (mouth[:, 1] >= 0.0)
        & (mouth[:, 1] < frame_height)
    )
    return float(in_frame.mean()) >= policy.min_mouth_landmarks_in_frame_ratio


class FaceLandmarker(Protocol):
    """Backend interface used by the deterministic temporal tracker."""

    name: str
    device: str

    def detect(self, frame_rgb: np.ndarray) -> list[FaceCandidate]: ...


@dataclass(frozen=True)
class TrackedFaceSequence:
    """Frame-aligned face boxes and 68-point landmarks for one video."""

    media: MediaMetadata
    processing_width: int
    processing_height: int
    frame_rate: int
    backend: str
    detection_stride: int
    device: str
    policy: FaceTrackingQualityPolicy
    landmarks: np.ndarray
    bounding_boxes: np.ndarray
    landmark_scores: np.ndarray
    detection_confidences: np.ndarray
    detected: np.ndarray
    mouth_visible_raw: np.ndarray
    mouth_visible: np.ndarray
    ambiguous: np.ndarray
    rejected_candidates: int
    candidate_rejection_reasons: dict[str, int]
    association_rejected_frames: int

    @property
    def frame_count(self) -> int:
        return int(self.landmarks.shape[0])

    @property
    def detected_frames(self) -> int:
        return int(self.detected.sum())

    @property
    def observed_frames(self) -> int:
        sampled = len(range(0, self.frame_count, self.detection_stride))
        final_was_skipped = (self.frame_count - 1) % self.detection_stride != 0
        return sampled + int(final_was_skipped)

    @property
    def detector_skipped_frames(self) -> int:
        return self.frame_count - self.observed_frames

    @property
    def interpolated_frames(self) -> int:
        return self.frame_count - self.detected_frames

    @property
    def detection_rate(self) -> float:
        return self.detected_frames / self.observed_frames

    @property
    def mouth_visible_frames(self) -> int:
        return int(self.mouth_visible.sum())

    @property
    def visual_coverage(self) -> float:
        return self.mouth_visible_frames / self.frame_count

    @property
    def raw_mouth_visible_frames(self) -> int:
        return int(self.mouth_visible_raw.sum())

    @property
    def ambiguous_frames(self) -> int:
        return int(self.ambiguous.sum())

    @property
    def quality_issues(self) -> list[str]:
        issues: list[str] = []
        if self.mouth_visible_frames < self.policy.min_detected_frames:
            issues.append(
                "mouth_visible_frames_below_minimum:"
                f"{self.mouth_visible_frames}<{self.policy.min_detected_frames}"
            )
        if self.visual_coverage < self.policy.min_detection_rate:
            issues.append(
                "visual_coverage_below_minimum:"
                f"{self.visual_coverage:.6f}<{self.policy.min_detection_rate:.6f}"
            )
        longest_gap = maximum_false_run(self.mouth_visible)
        if longest_gap > self.policy.max_missing_run:
            issues.append(
                f"maximum_missing_run_exceeded:{longest_gap}>"
                f"{self.policy.max_missing_run}"
            )
        leading_gap, trailing_gap = edge_false_runs(self.mouth_visible)
        if leading_gap > self.policy.max_edge_missing_run:
            issues.append(
                "leading_visual_gap_exceeded:"
                f"{leading_gap}>{self.policy.max_edge_missing_run}"
            )
        if trailing_gap > self.policy.max_edge_missing_run:
            issues.append(
                "trailing_visual_gap_exceeded:"
                f"{trailing_gap}>{self.policy.max_edge_missing_run}"
            )
        if self.ambiguous_frames > self.policy.max_ambiguous_frames:
            issues.append(
                "ambiguous_frames_exceeded:"
                f"{self.ambiguous_frames}>{self.policy.max_ambiguous_frames}"
            )
        return issues

    @property
    def quality_passed(self) -> bool:
        return not self.quality_issues

    def report(self) -> dict[str, Any]:
        valid_confidences = self.detection_confidences[self.detected]
        valid_landmark_scores = self.landmark_scores[self.detected]
        detected_boxes = self.bounding_boxes[self.detected]
        widths = detected_boxes[:, 2] - detected_boxes[:, 0]
        heights = detected_boxes[:, 3] - detected_boxes[:, 1]
        frame_area = self.media.video_width * self.media.video_height
        face_area_ratios = widths * heights / frame_area
        quality_issues = self.quality_issues
        leading_gap, trailing_gap = edge_false_runs(self.mouth_visible)
        filled_frames = int((~self.mouth_visible_raw & self.mouth_visible).sum())
        suppressed_frames = int((self.mouth_visible_raw & ~self.mouth_visible).sum())
        warnings: list[str] = []
        if self.interpolated_frames:
            warnings.append("missing_detections_interpolated")
        if filled_frames:
            warnings.append("short_visual_gaps_interpolated_for_display")
        return {
            "status": "passed" if not quality_issues else "degraded",
            "quality_status": "passed" if not quality_issues else "failed",
            "quality_issues": quality_issues,
            "warnings": warnings,
            "quality_thresholds": asdict(self.policy),
            "artifact_version": FACE_TRACK_ARTIFACT_VERSION,
            "media": self.media.to_dict(),
            "backend": self.backend,
            "device": self.device,
            "frame_rate": self.frame_rate,
            "frame_count": self.frame_count,
            "detection_stride": self.detection_stride,
            "processing_resolution": [
                self.processing_width,
                self.processing_height,
            ],
            "landmark_topology": "ibug_68",
            "detected_frames": self.detected_frames,
            "observed_frames": self.observed_frames,
            "detector_skipped_frames": self.detector_skipped_frames,
            "interpolated_frames": self.interpolated_frames,
            "detection_rate": self.detection_rate,
            "ambiguous_frames": self.ambiguous_frames,
            "mouth_visible_raw_frames": self.raw_mouth_visible_frames,
            "mouth_visible_frames": self.mouth_visible_frames,
            "visual_coverage": self.visual_coverage,
            "short_gap_frames_filled": filled_frames,
            "short_valid_frames_suppressed": suppressed_frames,
            "rejected_candidates": self.rejected_candidates,
            "candidate_rejection_reasons": dict(self.candidate_rejection_reasons),
            "association_rejected_frames": self.association_rejected_frames,
            "maximum_missing_run": maximum_false_run(self.mouth_visible),
            "leading_visual_gap": leading_gap,
            "trailing_visual_gap": trailing_gap,
            "mean_detection_confidence": float(valid_confidences.mean()),
            "mean_landmark_confidence": float(valid_landmark_scores.mean()),
            "minimum_face_area_ratio": float(face_area_ratios.min()),
            "median_face_area_ratio": float(np.median(face_area_ratios)),
            "visual_availability": build_visual_availability(
                self.mouth_visible,
                self.frame_rate,
            ),
        }


def resolve_tracking_device(requested: DeviceRequest) -> str:
    """Resolve an explicit or automatic tracking device."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise MediaInputError(
            "CUDA was requested for face tracking, but torch.cuda.is_available() "
            "is false. Use --device cpu or --device auto."
        )
    return requested


class FANFaceLandmarker:
    """RetinaFace detection plus compatible 68-point FAN landmarks."""

    name = "face_alignment_1.5.0_retinaface_fan4"

    def __init__(
        self,
        *,
        device: DeviceRequest = "auto",
        confidence_threshold: float = 0.8,
    ) -> None:
        self.device = resolve_tracking_device(device)
        try:
            import face_alignment
        except ImportError as exc:
            raise MediaInputError(
                "Face tracking requires face-alignment==1.5.0. "
                "Install the project dependencies with pip install -e '.[dev]'."
            ) from exc

        try:
            self._predictor = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                device=self.device,
                flip_input=False,
                face_detector="retinaface",
                face_detector_kwargs={
                    "confidence_threshold": confidence_threshold,
                },
                compile=False,
            )
        except Exception as exc:
            raise MediaInputError(
                f"Could not initialize RetinaFace/FAN on {self.device}: {exc}"
            ) from exc

    def detect(self, frame_rgb: np.ndarray) -> list[FaceCandidate]:
        """Detect every face and return its box, confidence, and landmarks."""
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise MediaInputError(
                f"Expected an RGB frame [H, W, 3], got {list(frame_rgb.shape)}."
            )
        try:
            landmarks, scores, boxes = self._predictor.get_landmarks_from_image(
                frame_rgb,
                return_bboxes=True,
                return_landmark_score=True,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise MediaInputError(f"RetinaFace/FAN prediction failed: {exc}") from exc
        if landmarks is None or scores is None or boxes is None:
            return []

        candidates: list[FaceCandidate] = []
        for points, point_scores, raw_box in zip(landmarks, scores, boxes, strict=True):
            box = np.asarray(raw_box, dtype=np.float32)
            candidates.append(
                FaceCandidate(
                    bounding_box=box[:4].copy(),
                    detection_confidence=float(box[4]),
                    landmarks=np.asarray(points, dtype=np.float32),
                    landmark_scores=np.asarray(point_scores, dtype=np.float32),
                )
            )
        return candidates


def scaled_detection_size(
    width: int, height: int, max_dimension: int
) -> tuple[int, int]:
    """Return an even, aspect-preserving size without upscaling."""
    if width <= 0 or height <= 0 or max_dimension <= 0:
        raise ValueError("Dimensions must be positive.")
    scale = min(1.0, max_dimension / max(width, height))
    scaled_width = max(2, round(width * scale / 2) * 2)
    scaled_height = max(2, round(height * scale / 2) * 2)
    return scaled_width, scaled_height


def _require_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise MediaInputError(
            "Required executable 'ffmpeg' was not found in the active environment."
        )
    return executable


def _read_exact(stream: Any, byte_count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = byte_count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def iter_resampled_rgb_frames(
    path: Path,
    *,
    width: int,
    height: int,
    frame_rate: int = TARGET_FRAME_RATE,
) -> Iterator[np.ndarray]:
    """Stream CFR RGB frames from FFmpeg without loading a video into memory."""
    ffmpeg = _require_ffmpeg()
    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-vf",
        f"fps={frame_rate},scale={width}:{height}:flags=bicubic,format=rgb24",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise MediaInputError(f"Could not start FFmpeg: {exc}") from exc
    assert process.stdout is not None
    assert process.stderr is not None
    frame_bytes = width * height * 3
    completed = False
    try:
        while True:
            payload = _read_exact(process.stdout, frame_bytes)
            if not payload:
                break
            if len(payload) != frame_bytes:
                raise MediaInputError(
                    "FFmpeg returned an incomplete RGB frame while tracking faces."
                )
            yield (
                np.frombuffer(payload, dtype=np.uint8).copy().reshape(height, width, 3)
            )
        stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = process.wait()
        completed = True
        if return_code != 0:
            raise MediaInputError(f"FFmpeg video decoding failed: {stderr}")
    finally:
        process.stdout.close()
        process.stderr.close()
        if not completed and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def bounding_box_iou(left: np.ndarray, right: np.ndarray) -> float:
    """Calculate intersection over union for two xyxy boxes."""
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    right_area = max(0.0, float(right[2] - right[0])) * max(
        0.0, float(right[3] - right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _candidate_area(candidate: FaceCandidate) -> float:
    box = candidate.bounding_box
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _candidate_rejection_reason(
    candidate: FaceCandidate,
    *,
    frame_area: float | None,
    frame_size: tuple[float, float] | None,
    policy: FaceTrackingQualityPolicy,
) -> str | None:
    """Return why a candidate is unusable, or None when it passes."""
    if not np.isfinite(candidate.bounding_box).all():
        return "non_finite_bounding_box"
    if not math.isfinite(candidate.detection_confidence):
        return "non_finite_detection_confidence"
    if not 0.0 <= candidate.detection_confidence <= 1.0:
        return "detection_confidence_out_of_range"
    if not np.isfinite(candidate.landmarks).all():
        return "non_finite_landmarks"
    if not np.isfinite(candidate.landmark_scores).all():
        return "non_finite_landmark_scores"
    if (candidate.landmark_scores < 0.0).any() or (
        candidate.landmark_scores > 1.0
    ).any():
        return "landmark_confidence_out_of_range"
    if candidate.detection_confidence < policy.min_detection_confidence:
        return "detection_confidence_below_minimum"
    if float(candidate.landmark_scores.mean()) < policy.min_landmark_confidence:
        return "landmark_confidence_below_minimum"

    area = _candidate_area(candidate)
    if area <= 0:
        return "non_positive_bounding_box_area"

    effective_area = area
    effective_frame_area = frame_area
    if frame_size is not None:
        frame_width, frame_height = frame_size
        x1, y1, x2, y2 = (float(value) for value in candidate.bounding_box)
        visible_width = max(0.0, min(x2, frame_width) - max(x1, 0.0))
        visible_height = max(0.0, min(y2, frame_height) - max(y1, 0.0))
        visible_area = visible_width * visible_height
        out_of_frame_ratio = 1.0 - visible_area / area
        mouth_is_visible = mouth_landmarks_visible(
            candidate, frame_size=frame_size, policy=policy
        )
        if out_of_frame_ratio > policy.max_out_of_frame_ratio and not mouth_is_visible:
            return "bounding_box_too_far_outside_frame"
        effective_area = visible_area
        effective_frame_area = frame_width * frame_height

    if (
        effective_frame_area is not None
        and effective_area / effective_frame_area < policy.min_face_area_ratio
    ):
        return "face_area_below_minimum"
    return None


def _tracking_metrics(
    candidate: FaceCandidate, previous_box: np.ndarray
) -> tuple[float, float, float]:
    previous_center = (previous_box[:2] + previous_box[2:]) / 2.0
    previous_diagonal = max(
        math.hypot(
            float(previous_box[2] - previous_box[0]),
            float(previous_box[3] - previous_box[1]),
        ),
        1.0,
    )
    center = (candidate.bounding_box[:2] + candidate.bounding_box[2:]) / 2.0
    normalized_distance = (
        float(np.linalg.norm(center - previous_center)) / previous_diagonal
    )
    iou = bounding_box_iou(previous_box, candidate.bounding_box)
    score = 3.0 * iou - normalized_distance + 0.25 * candidate.detection_confidence
    return iou, normalized_distance, score


def decide_tracked_face(
    candidates: Sequence[FaceCandidate],
    previous_box: np.ndarray | None,
    *,
    frame_area: float | None,
    policy: FaceTrackingQualityPolicy,
    frame_size: tuple[float, float] | None = None,
) -> TrackingDecision:
    """Validate candidates and associate one face without silently switching tracks."""
    valid: list[FaceCandidate] = []
    rejected_reasons: list[str] = []
    for candidate in candidates:
        rejection = _candidate_rejection_reason(
            candidate,
            frame_area=frame_area,
            frame_size=frame_size,
            policy=policy,
        )
        if rejection is None:
            valid.append(candidate)
        else:
            rejected_reasons.append(rejection)
    if not valid:
        return TrackingDecision(
            None,
            "missing",
            "no_valid_candidates",
            rejected_reasons=tuple(rejected_reasons),
        )

    if previous_box is None:
        ranked = sorted(
            valid,
            key=lambda item: (_candidate_area(item), item.detection_confidence),
            reverse=True,
        )
        if len(ranked) > 1:
            largest_area = _candidate_area(ranked[0])
            second_area = _candidate_area(ranked[1])
            relative_margin = (largest_area - second_area) / largest_area
            if relative_margin <= policy.initial_face_area_margin_ratio:
                return TrackingDecision(
                    None,
                    "ambiguous",
                    "initial_faces_have_similar_size",
                    rejected_reasons=tuple(rejected_reasons),
                )
        return TrackingDecision(
            ranked[0],
            "accepted",
            "largest_initial_face",
            rejected_reasons=tuple(rejected_reasons),
        )

    previous_area = max(
        (float(previous_box[2]) - float(previous_box[0]))
        * (float(previous_box[3]) - float(previous_box[1])),
        1.0,
    )
    associated: list[tuple[float, FaceCandidate]] = []
    for candidate in valid:
        candidate_area = _candidate_area(candidate)
        area_change = max(
            candidate_area / previous_area, previous_area / candidate_area
        )
        iou, normalized_distance, score = _tracking_metrics(candidate, previous_box)
        scale_reacquisition = (
            len(valid) == 1 and normalized_distance <= policy.max_center_distance_ratio
        )
        if area_change > policy.max_face_area_change_ratio and not scale_reacquisition:
            rejected_reasons.append("face_area_change_exceeded")
            continue
        if area_change > policy.max_face_area_change_ratio:
            score -= 0.25 * math.log(area_change)
        if (
            iou >= policy.min_association_iou
            or normalized_distance <= policy.max_center_distance_ratio
        ):
            associated.append((score, candidate))
        else:
            rejected_reasons.append("association_gate_failed")
    if not associated:
        return TrackingDecision(
            None,
            "missing",
            "association_gate_failed",
            rejected_reasons=tuple(rejected_reasons),
        )

    associated.sort(key=lambda item: item[0], reverse=True)
    if (
        len(associated) > 1
        and associated[0][0] - associated[1][0] <= policy.ambiguity_score_margin
    ):
        return TrackingDecision(
            None,
            "ambiguous",
            "candidates_have_similar_tracking_scores",
            associated[0][0],
            tuple(rejected_reasons),
        )
    score, candidate = associated[0]
    return TrackingDecision(
        candidate,
        "accepted",
        "association_gate_passed",
        score,
        tuple(rejected_reasons),
    )


def select_tracked_face(
    candidates: Sequence[FaceCandidate], previous_box: np.ndarray | None
) -> FaceCandidate | None:
    """Compatibility wrapper around the quality-aware association decision."""
    decision = decide_tracked_face(
        candidates,
        previous_box,
        frame_area=None,
        policy=FaceTrackingQualityPolicy(),
    )
    return decision.candidate


def _scale_candidate(
    candidate: FaceCandidate, *, scale_x: float, scale_y: float
) -> FaceCandidate:
    box = candidate.bounding_box.copy()
    box[[0, 2]] *= scale_x
    box[[1, 3]] *= scale_y
    landmarks = candidate.landmarks.copy()
    landmarks[:, 0] *= scale_x
    landmarks[:, 1] *= scale_y
    return FaceCandidate(
        bounding_box=box,
        detection_confidence=candidate.detection_confidence,
        landmarks=landmarks,
        landmark_scores=candidate.landmark_scores.copy(),
    )


def maximum_false_run(mask: np.ndarray) -> int:
    """Return the longest consecutive run of false values."""
    longest = current = 0
    for value in mask:
        if bool(value):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def edge_false_runs(mask: np.ndarray) -> tuple[int, int]:
    """Return the leading and trailing runs of false values."""
    leading = 0
    for value in mask:
        if bool(value):
            break
        leading += 1
    trailing = 0
    for value in mask[::-1]:
        if bool(value):
            break
        trailing += 1
    return leading, trailing


def stabilize_visual_availability(
    mask: np.ndarray,
    *,
    max_short_gap: int,
    min_valid_run: int,
) -> np.ndarray:
    """Suppress isolated valid blips and bridge only short internal dropouts."""
    values = np.asarray(mask, dtype=np.bool_)
    if values.ndim != 1:
        raise ValueError("mask must be one-dimensional.")
    if max_short_gap < 0:
        raise ValueError("max_short_gap must be non-negative.")
    if min_valid_run < 1:
        raise ValueError("min_valid_run must be positive.")
    result = values.copy()

    def runs(target: bool) -> Iterator[tuple[int, int]]:
        index = 0
        while index < len(result):
            if bool(result[index]) != target:
                index += 1
                continue
            start = index
            while index < len(result) and bool(result[index]) == target:
                index += 1
            yield start, index

    for start, end in list(runs(True)):
        if end - start < min_valid_run:
            result[start:end] = False

    for start, end in list(runs(False)):
        if start > 0 and end < len(result) and end - start <= max_short_gap:
            result[start:end] = True
    return result


def build_visual_availability(
    detected: np.ndarray,
    frame_rate: int,
) -> dict[str, Any]:
    """Describe every missing visual interval from a frame-aligned mask."""
    mask = np.asarray(detected, dtype=np.bool_)
    if mask.ndim != 1:
        raise ValueError("detected must be a one-dimensional frame mask.")
    if frame_rate <= 0:
        raise ValueError("frame_rate must be positive.")

    intervals: list[dict[str, int | float]] = []
    missing_start: int | None = None
    for frame_index, is_detected in enumerate(mask):
        if not is_detected and missing_start is None:
            missing_start = frame_index
        if is_detected and missing_start is not None:
            intervals.append(_missing_interval(missing_start, frame_index, frame_rate))
            missing_start = None
    if missing_start is not None:
        intervals.append(_missing_interval(missing_start, len(mask), frame_rate))

    valid_frames = int(mask.sum())
    frame_count = len(mask)
    return {
        "frame_rate": frame_rate,
        "frame_count": frame_count,
        "valid_frames": valid_frames,
        "missing_frames": frame_count - valid_frames,
        "coverage": valid_frames / frame_count if frame_count else 0.0,
        "missing_intervals": intervals,
    }


def _missing_interval(
    start_frame: int,
    end_frame_exclusive: int,
    frame_rate: int,
) -> dict[str, int | float]:
    frame_count = end_frame_exclusive - start_frame
    return {
        "start_frame": start_frame,
        "end_frame_exclusive": end_frame_exclusive,
        "frame_count": frame_count,
        "start_seconds": start_frame / frame_rate,
        "end_seconds": end_frame_exclusive / frame_rate,
        "duration_seconds": frame_count / frame_rate,
    }


def interpolate_missing_rows(values: np.ndarray, detected: np.ndarray) -> np.ndarray:
    """Linearly fill internal gaps and extend nearest values at sequence ends."""
    if values.shape[0] != detected.shape[0]:
        raise ValueError("values and detected must have the same frame count.")
    valid_indices = np.flatnonzero(detected)
    if len(valid_indices) == 0:
        raise MediaInputError("No face was detected in any video frame.")
    all_indices = np.arange(len(values))
    flattened = values.reshape(len(values), -1)
    result = np.empty_like(flattened, dtype=np.float32)
    for column in range(flattened.shape[1]):
        result[:, column] = np.interp(
            all_indices,
            valid_indices,
            flattened[valid_indices, column],
        )
    return result.reshape(values.shape)


def build_tracked_sequence(
    candidates_by_frame: Sequence[Sequence[FaceCandidate]],
    *,
    media: MediaMetadata,
    processing_width: int,
    processing_height: int,
    frame_rate: int,
    backend: str,
    device: str,
    detection_stride: int = 1,
    policy: FaceTrackingQualityPolicy | None = None,
) -> TrackedFaceSequence:
    """Select one track, record quality failures, and interpolate for handoff."""
    if (
        isinstance(detection_stride, bool)
        or not isinstance(detection_stride, Integral)
        or detection_stride <= 0
    ):
        raise ValueError("detection_stride must be a positive integer.")
    active_policy = policy or FaceTrackingQualityPolicy()
    frame_count = len(candidates_by_frame)
    if frame_count == 0:
        raise MediaInputError("Video contains no decodable frames.")
    landmarks = np.full((frame_count, LANDMARK_COUNT, 2), np.nan, dtype=np.float32)
    boxes = np.full((frame_count, 4), np.nan, dtype=np.float32)
    scores = np.full((frame_count, LANDMARK_COUNT), np.nan, dtype=np.float32)
    confidences = np.full(frame_count, np.nan, dtype=np.float32)
    detected = np.zeros(frame_count, dtype=np.bool_)
    mouth_visible_raw = np.zeros(frame_count, dtype=np.bool_)
    ambiguous = np.zeros(frame_count, dtype=np.bool_)
    rejection_counts: Counter[str] = Counter()
    association_rejected_frames = 0
    previous_box: np.ndarray | None = None
    frame_area = float(media.video_width * media.video_height)

    for frame_index, candidates in enumerate(candidates_by_frame):
        decision = decide_tracked_face(
            candidates,
            previous_box,
            frame_area=frame_area,
            policy=active_policy,
            frame_size=(media.video_width, media.video_height),
        )
        rejection_counts.update(decision.rejected_reasons)
        if decision.status == "ambiguous":
            ambiguous[frame_index] = True
        if decision.candidate is None:
            if decision.reason == "association_gate_failed":
                association_rejected_frames += 1
            continue
        selected = decision.candidate
        landmarks[frame_index] = selected.landmarks
        boxes[frame_index] = selected.bounding_box
        scores[frame_index] = selected.landmark_scores
        confidences[frame_index] = selected.detection_confidence
        detected[frame_index] = True
        mouth_visible_raw[frame_index] = mouth_landmarks_visible(
            selected,
            frame_size=(media.video_width, media.video_height),
            policy=active_policy,
        )
        previous_box = selected.bounding_box

    if not detected.any() and any(candidates_by_frame):
        reasons = ", ".join(
            f"{name}={count}" for name, count in sorted(rejection_counts.items())
        )
        raise MediaInputError(
            "Face candidates were detected, but none passed the tracking quality "
            f"gates ({reasons or 'no accepted candidates'})."
        )
    mouth_visible = stabilize_visual_availability(
        mouth_visible_raw,
        max_short_gap=active_policy.max_short_visual_gap,
        min_valid_run=1 if detection_stride > 1 else active_policy.min_visual_run,
    )
    interpolated_landmarks = interpolate_missing_rows(landmarks, detected)
    interpolated_boxes = interpolate_missing_rows(boxes, detected)
    return TrackedFaceSequence(
        media=media,
        processing_width=processing_width,
        processing_height=processing_height,
        frame_rate=frame_rate,
        detection_stride=detection_stride,
        backend=backend,
        device=device,
        policy=active_policy,
        landmarks=interpolated_landmarks,
        bounding_boxes=interpolated_boxes,
        landmark_scores=scores,
        detection_confidences=confidences,
        detected=detected,
        ambiguous=ambiguous,
        rejected_candidates=sum(rejection_counts.values()),
        mouth_visible_raw=mouth_visible_raw,
        mouth_visible=mouth_visible,
        candidate_rejection_reasons=dict(sorted(rejection_counts.items())),
        association_rejected_frames=association_rejected_frames,
    )


def track_face_landmarks(
    path: Path | str,
    *,
    landmarker: FaceLandmarker | None = None,
    device: DeviceRequest = "auto",
    frame_rate: int = TARGET_FRAME_RATE,
    max_detection_size: int = DEFAULT_DETECTION_MAX_SIZE,
    detection_stride: int = 1,
    policy: FaceTrackingQualityPolicy | None = None,
) -> TrackedFaceSequence:
    """Detect and temporally track one 68-point face across a media file."""
    if frame_rate <= 0:
        raise MediaInputError("Face-tracking frame rate must be positive.")
    if max_detection_size <= 0:
        raise MediaInputError("Face-tracking maximum detection size must be positive.")
    if (
        isinstance(detection_stride, bool)
        or not isinstance(detection_stride, Integral)
        or detection_stride <= 0
    ):
        raise MediaInputError("Face-tracking detection stride must be positive.")
    active_policy = policy or FaceTrackingQualityPolicy()
    media_path = Path(path).expanduser().resolve()
    metadata = probe_av_media(media_path)
    processing_width, processing_height = scaled_detection_size(
        metadata.video_width,
        metadata.video_height,
        max_detection_size,
    )
    active_landmarker = landmarker or FANFaceLandmarker(
        device=device,
        confidence_threshold=active_policy.min_detection_confidence,
    )
    scale_x = metadata.video_width / processing_width
    scale_y = metadata.video_height / processing_height
    candidates_by_frame: list[list[FaceCandidate]] = []
    final_skipped: tuple[int, np.ndarray] | None = None

    def detect_candidates(frame: np.ndarray) -> list[FaceCandidate]:
        return [
            _scale_candidate(item, scale_x=scale_x, scale_y=scale_y)
            for item in active_landmarker.detect(frame)
        ]

    for frame_index, frame in enumerate(
        iter_resampled_rgb_frames(
            media_path,
            width=processing_width,
            height=processing_height,
            frame_rate=frame_rate,
        )
    ):
        if frame_index % detection_stride == 0:
            candidates_by_frame.append(detect_candidates(frame))
            final_skipped = None
        else:
            candidates_by_frame.append([])
            final_skipped = (frame_index, frame)

    # Always observe the final frame so stride-based sampling cannot manufacture
    # a trailing visual gap. Intermediate skipped frames are reconstructed by the
    # same temporal interpolation/stabilization already used for short dropouts.
    if final_skipped is not None:
        frame_index, frame = final_skipped
        candidates_by_frame[frame_index] = detect_candidates(frame)

    return build_tracked_sequence(
        candidates_by_frame,
        media=metadata,
        processing_width=processing_width,
        processing_height=processing_height,
        frame_rate=frame_rate,
        detection_stride=detection_stride,
        backend=active_landmarker.name,
        device=active_landmarker.device,
        policy=active_policy,
    )


def save_face_tracking_artifacts(
    sequence: TrackedFaceSequence,
    *,
    artifact_path: Path | str,
    report_path: Path | str,
) -> dict[str, Any]:
    """Write a safe numeric NPZ handoff and a human-readable JSON report."""
    artifact = Path(artifact_path).expanduser().resolve()
    report = Path(report_path).expanduser().resolve()
    artifact.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact,
        landmarks=sequence.landmarks,
        bounding_boxes=sequence.bounding_boxes,
        landmark_scores=sequence.landmark_scores,
        detection_confidences=sequence.detection_confidences,
        detected=sequence.detected,
        detection_stride=np.asarray(
            [sequence.detection_stride],
            dtype=np.int32,
        ),
        mouth_visible_raw=sequence.mouth_visible_raw,
        mouth_visible=sequence.mouth_visible,
        original_resolution=np.asarray(
            [sequence.media.video_width, sequence.media.video_height],
            dtype=np.int32,
        ),
        processing_resolution=np.asarray(
            [sequence.processing_width, sequence.processing_height],
            dtype=np.int32,
        ),
        frame_rate=np.asarray([sequence.frame_rate], dtype=np.int32),
        ambiguous=sequence.ambiguous,
        quality_passed=np.asarray([sequence.quality_passed], dtype=np.bool_),
        artifact_version=np.asarray(
            [FACE_TRACK_ARTIFACT_VERSION],
            dtype=np.int32,
        ),
    )
    payload = sequence.report()
    payload["artifact_path"] = str(artifact)
    payload["report_path"] = str(report)
    report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload
