from __future__ import annotations

import io
import inspect

import numpy as np
import pytest

from viavsr.preprocessing import mouth_roi
from viavsr.preprocessing.errors import MediaInputError
from viavsr.preprocessing.mouth_roi import (
    MOUTH_START_INDEX,
    MOUTH_STOP_INDEX,
    align_and_crop_mouth_frame,
    create_no_signal_frame,
    estimate_alignment_transform,
    export_aligned_mouth_roi_video,
    load_face_track_artifact,
    load_mean_face,
    smooth_landmarks,
)


def test_aligned_export_exposes_quality_override_as_keyword_only() -> None:
    parameter = inspect.signature(export_aligned_mouth_roi_video).parameters[
        "require_quality_passed"
    ]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is True


def test_create_no_signal_frame_is_visible_grayscale_placeholder() -> None:
    frame = create_no_signal_frame()

    assert frame.shape == (96, 96)
    assert frame.dtype == np.uint8
    assert frame.min() < frame.max()
    assert np.count_nonzero(frame > 100) > 0


def test_load_mean_face_returns_official_68_point_geometry() -> None:
    reference = load_mean_face()

    assert reference.shape == (68, 2)
    assert reference.dtype == np.float32
    np.testing.assert_allclose(reference[0], [70.92384, 97.13758], rtol=1e-6)
    np.testing.assert_allclose(reference[-1], [122.487755, 160.50879], rtol=1e-6)


def test_load_mean_face_rejects_invalid_asset(tmp_path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text('{"points": [[1, 2]]}', encoding="utf-8")

    with pytest.raises(MediaInputError, match=r"\[68, 2\]"):
        load_mean_face(path)


def test_load_mean_face_wraps_invalid_numeric_data(tmp_path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text('{"points": [["not-a-number", 2]]}', encoding="utf-8")

    with pytest.raises(MediaInputError, match="Could not load mean-face asset"):
        load_mean_face(path)


def test_smooth_landmarks_preserves_per_frame_translation() -> None:
    reference = load_mean_face()
    translations = np.asarray(
        [[0.0, 0.0], [2.0, 3.0], [7.0, -1.0], [8.0, 4.0]],
        dtype=np.float32,
    )
    landmarks = reference[None, :, :] + translations[:, None, :]

    result = smooth_landmarks(landmarks, window_size=4)

    np.testing.assert_allclose(result, landmarks, atol=5e-5)


def test_estimate_alignment_transform_is_identity_for_reference() -> None:
    reference = load_mean_face()

    transform = estimate_alignment_transform(reference, reference)

    np.testing.assert_allclose(
        transform,
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        atol=1e-5,
    )


def test_align_and_crop_mouth_frame_returns_expected_96_square() -> None:
    reference = load_mean_face()
    y_coordinates = np.arange(256, dtype=np.uint8)[:, None]
    grayscale = np.broadcast_to(y_coordinates, (256, 256))
    frame_rgb = np.repeat(grayscale[:, :, None], 3, axis=2)

    result = align_and_crop_mouth_frame(frame_rgb, reference, reference)

    mouth_center = reference[MOUTH_START_INDEX:MOUTH_STOP_INDEX].mean(axis=0)
    expected_y_min = round(mouth_center[1]) - 48
    assert result.shape == (96, 96)
    assert result.dtype == np.uint8
    assert result[0, 0] == expected_y_min
    assert result[-1, -1] == expected_y_min + 95


def test_load_face_track_artifact_reads_numeric_arrays(tmp_path) -> None:
    path = tmp_path / "track.npz"
    np.savez_compressed(
        path,
        landmarks=np.ones((3, 68, 2), dtype=np.float32),
        detected=np.asarray([True, False, True]),
        original_resolution=np.asarray([1920, 1080], dtype=np.int32),
        frame_rate=np.asarray([25], dtype=np.int32),
        artifact_version=np.asarray([1], dtype=np.int32),
        quality_passed=np.asarray([True], dtype=np.bool_),
    )

    artifact = load_face_track_artifact(path)

    assert artifact.frame_count == 3
    assert artifact.original_width == 1920
    assert artifact.original_height == 1080
    assert artifact.frame_rate == 25
    assert artifact.artifact_version == 1
    assert artifact.detected.tolist() == [True, False, True]
    assert artifact.mouth_visible_raw.tolist() == [True, False, True]
    assert artifact.mouth_visible.tolist() == [True, False, True]


def test_load_face_track_artifact_reads_version_two_visibility_masks(tmp_path) -> None:
    path = tmp_path / "track_v2.npz"
    np.savez_compressed(
        path,
        landmarks=np.ones((3, 68, 2), dtype=np.float32),
        detected=np.asarray([True, False, True]),
        mouth_visible_raw=np.asarray([True, False, True]),
        mouth_visible=np.asarray([True, True, True]),
        original_resolution=np.asarray([1920, 1080], dtype=np.int32),
        frame_rate=np.asarray([25], dtype=np.int32),
        artifact_version=np.asarray([2], dtype=np.int32),
        quality_passed=np.asarray([True], dtype=np.bool_),
    )

    artifact = load_face_track_artifact(path)

    assert artifact.artifact_version == 2
    assert artifact.detected.tolist() == [True, False, True]
    assert artifact.mouth_visible_raw.tolist() == [True, False, True]
    assert artifact.mouth_visible.tolist() == [True, True, True]


@pytest.mark.parametrize(
    ("landmarks", "detected", "message"),
    [
        (
            np.ones((3, 67, 2), dtype=np.float32),
            np.ones(3, dtype=np.bool_),
            "shape",
        ),
        (
            np.ones((3, 68, 2), dtype=np.float32),
            np.ones(2, dtype=np.bool_),
            "detection mask",
        ),
    ],
)
def test_load_face_track_artifact_rejects_invalid_shapes(
    tmp_path,
    landmarks,
    detected,
    message,
) -> None:
    path = tmp_path / "track.npz"
    np.savez_compressed(
        path,
        landmarks=landmarks,
        detected=detected,
        original_resolution=np.asarray([1920, 1080], dtype=np.int32),
        frame_rate=np.asarray([25], dtype=np.int32),
        artifact_version=np.asarray([1], dtype=np.int32),
        quality_passed=np.asarray([True], dtype=np.bool_),
    )

    with pytest.raises(MediaInputError, match=message):
        load_face_track_artifact(path)


def test_load_face_track_artifact_rejects_failed_quality_gate(tmp_path) -> None:
    path = tmp_path / "failed_track.npz"
    np.savez_compressed(
        path,
        landmarks=np.ones((3, 68, 2), dtype=np.float32),
        detected=np.asarray([True, False, True]),
        original_resolution=np.asarray([1920, 1080], dtype=np.int32),
        frame_rate=np.asarray([25], dtype=np.int32),
        artifact_version=np.asarray([1], dtype=np.int32),
        quality_passed=np.asarray([False], dtype=np.bool_),
    )

    with pytest.raises(MediaInputError, match="quality gates failed"):
        load_face_track_artifact(path)

    diagnostic_artifact = load_face_track_artifact(path, require_quality_passed=False)
    assert diagnostic_artifact.quality_passed is False
    assert diagnostic_artifact.detected.tolist() == [True, False, True]


def test_load_face_track_artifact_rejects_legacy_artifact_without_quality(
    tmp_path,
) -> None:
    path = tmp_path / "legacy_track.npz"
    np.savez_compressed(
        path,
        landmarks=np.ones((3, 68, 2), dtype=np.float32),
        detected=np.asarray([True, False, True]),
        original_resolution=np.asarray([1920, 1080], dtype=np.int32),
        frame_rate=np.asarray([25], dtype=np.int32),
    )

    with pytest.raises(
        MediaInputError,
        match="VIAVSR-7 face-track artifact with quality metadata",
    ):
        load_face_track_artifact(path)


def test_load_face_track_artifact_rejects_unsupported_version(tmp_path) -> None:
    path = tmp_path / "future_track.npz"
    np.savez_compressed(
        path,
        landmarks=np.ones((3, 68, 2), dtype=np.float32),
        detected=np.asarray([True, True, True]),
        original_resolution=np.asarray([1920, 1080], dtype=np.int32),
        frame_rate=np.asarray([25], dtype=np.int32),
        artifact_version=np.asarray([999], dtype=np.int32),
        quality_passed=np.asarray([True], dtype=np.bool_),
    )

    with pytest.raises(
        MediaInputError, match="Unsupported face-track artifact version"
    ):
        load_face_track_artifact(path)


def test_mouth_encoder_pads_audio_to_preserve_video_timeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = io.BytesIO()
            self.stderr = io.BytesIO()

        def poll(self) -> int:
            return 0

        def wait(self, timeout: int | None = None) -> int:
            return 0

    captured: dict[str, list[str]] = {}

    def fake_popen(command: list[str], **_kwargs: object) -> FakeProcess:
        captured["command"] = command
        return FakeProcess()

    monkeypatch.setattr(mouth_roi, "_require_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(mouth_roi.subprocess, "Popen", fake_popen)

    patches = [np.zeros((96, 96), dtype=np.uint8) for _ in range(2)]
    encoded_frames = mouth_roi._encode_mouth_video(
        patches,
        source_path=tmp_path / "source.mp4",
        output_path=tmp_path / "mouth.mp4",
        frame_rate=25,
    )

    assert encoded_frames == 2
