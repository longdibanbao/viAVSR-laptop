import json

import numpy as np
import pytest

from viavsr.preprocessing import face_tracking
from viavsr.preprocessing.errors import MediaInputError
from viavsr.preprocessing.face_tracking import (
    FaceCandidate,
    FaceTrackingQualityPolicy,
    bounding_box_iou,
    build_tracked_sequence,
    build_visual_availability,
    decide_tracked_face,
    interpolate_missing_rows,
    load_face_tracking_quality_policy,
    maximum_false_run,
    save_face_tracking_artifacts,
    scaled_detection_size,
    select_tracked_face,
    stabilize_visual_availability,
    track_face_landmarks,
)
from viavsr.preprocessing.media import MediaMetadata


def _candidate(
    box: tuple[float, float, float, float],
    *,
    confidence: float = 0.99,
    landmark_value: float = 1.0,
    landmark_confidence: float = 0.9,
) -> FaceCandidate:
    return FaceCandidate(
        bounding_box=np.asarray(box, dtype=np.float32),
        detection_confidence=confidence,
        landmarks=np.full((68, 2), landmark_value, dtype=np.float32),
        landmark_scores=np.full(68, landmark_confidence, dtype=np.float32),
    )


def _metadata() -> MediaMetadata:
    return MediaMetadata(
        path="/tmp/webcam.mp4",
        duration_seconds=4.0,
        video_width=1920,
        video_height=1080,
        frame_rate=15.0,
        audio_sample_rate=48_000,
        audio_channels=2,
    )


def test_scaled_detection_size_preserves_aspect_ratio_without_upscaling() -> None:
    assert scaled_detection_size(1920, 1080, 640) == (640, 360)
    assert scaled_detection_size(320, 240, 640) == (320, 240)


def test_track_face_landmarks_rejects_invalid_processing_parameters() -> None:
    with pytest.raises(MediaInputError, match="frame rate must be positive"):
        track_face_landmarks("missing.mp4", frame_rate=0)

    with pytest.raises(
        MediaInputError,
        match="maximum detection size must be positive",
    ):
        track_face_landmarks("missing.mp4", max_detection_size=0)

    with pytest.raises(MediaInputError, match="detection stride must be positive"):
        track_face_landmarks("missing.mp4", detection_stride=0)


def test_tracking_stride_interpolates_skipped_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _metadata()
    frames = [np.zeros((360, 640, 3), dtype=np.uint8) for _ in range(5)]
    calls: list[int] = []

    class FakeLandmarker:
        name = "fake"
        device = "cpu"

        def detect(self, frame: np.ndarray) -> list[FaceCandidate]:
            calls.append(id(frame))
            return [_candidate((10, 10, 50, 50), landmark_value=20.0)]

    monkeypatch.setattr(face_tracking, "probe_av_media", lambda _path: metadata)
    monkeypatch.setattr(
        face_tracking,
        "iter_resampled_rgb_frames",
        lambda *_args, **_kwargs: iter(frames),
    )

    sequence = track_face_landmarks(
        "/tmp/webcam.mp4",
        landmarker=FakeLandmarker(),
        detection_stride=2,
    )

    assert len(calls) == 3
    assert sequence.detected.tolist() == [True, False, True, False, True]
    assert sequence.mouth_visible.tolist() == [True, True, True, True, True]
    assert sequence.observed_frames == 3
    assert sequence.detector_skipped_frames == 2
    assert sequence.detection_rate == 1.0
    assert sequence.quality_passed


def test_bounding_box_iou() -> None:
    left = np.asarray([0, 0, 10, 10], dtype=np.float32)
    right = np.asarray([5, 5, 15, 15], dtype=np.float32)

    assert bounding_box_iou(left, right) == pytest.approx(25 / 175)


def test_tracker_selects_largest_initial_face_then_preserves_identity() -> None:
    target = _candidate((10, 10, 50, 50))
    other = _candidate((100, 100, 130, 130))
    initial = select_tracked_face([other, target], previous_box=None)
    assert initial is target

    moved_target = _candidate((12, 11, 52, 51))
    larger_other = _candidate((80, 80, 160, 160))
    selected = select_tracked_face(
        [larger_other, moved_target],
        previous_box=target.bounding_box,
    )

    assert selected is moved_target


def test_interpolate_missing_rows_fills_internal_and_edge_gaps() -> None:
    values = np.asarray(
        [
            [np.nan, np.nan],
            [2.0, 4.0],
            [np.nan, np.nan],
            [6.0, 8.0],
            [np.nan, np.nan],
        ],
        dtype=np.float32,
    )
    detected = np.asarray([False, True, False, True, False])

    result = interpolate_missing_rows(values, detected)

    np.testing.assert_allclose(
        result,
        np.asarray(
            [
                [2.0, 4.0],
                [2.0, 4.0],
                [4.0, 6.0],
                [6.0, 8.0],
                [6.0, 8.0],
            ],
            dtype=np.float32,
        ),
    )


def test_interpolate_missing_rows_rejects_sequence_without_a_face() -> None:
    values = np.full((3, 2), np.nan, dtype=np.float32)

    with pytest.raises(MediaInputError, match="No face was detected"):
        interpolate_missing_rows(values, np.zeros(3, dtype=np.bool_))


def test_maximum_false_run() -> None:
    mask = np.asarray([True, False, False, True, False])
    assert maximum_false_run(mask) == 2


def test_build_visual_availability_reports_every_missing_interval() -> None:
    mask = np.asarray(
        [False, False, True, False, True, False, False],
        dtype=np.bool_,
    )

    result = build_visual_availability(mask, frame_rate=2)

    assert result["frame_count"] == 7
    assert result["valid_frames"] == 2
    assert result["missing_frames"] == 5
    assert result["coverage"] == pytest.approx(2 / 7)
    assert result["missing_intervals"] == [
        {
            "start_frame": 0,
            "end_frame_exclusive": 2,
            "frame_count": 2,
            "start_seconds": 0.0,
            "end_seconds": 1.0,
            "duration_seconds": 1.0,
        },
        {
            "start_frame": 3,
            "end_frame_exclusive": 4,
            "frame_count": 1,
            "start_seconds": 1.5,
            "end_seconds": 2.0,
            "duration_seconds": 0.5,
        },
        {
            "start_frame": 5,
            "end_frame_exclusive": 7,
            "frame_count": 2,
            "start_seconds": 2.5,
            "end_seconds": 3.5,
            "duration_seconds": 1.0,
        },
    ]


def test_stabilize_visual_availability_removes_flicker_only() -> None:
    raw = np.asarray(
        [
            False,
            True,
            False,
            True,
            True,
            False,
            True,
            True,
            False,
            False,
            False,
            True,
            False,
        ]
    )

    result = stabilize_visual_availability(raw, max_short_gap=1, min_valid_run=2)

    assert result.tolist() == [False] * 3 + [True] * 5 + [False] * 5


def test_build_tracked_sequence_interpolates_missing_frame() -> None:
    frame_zero = _candidate((10, 10, 50, 50), landmark_value=0.0)
    frame_one = _candidate((12, 12, 52, 52), landmark_value=2.0)
    frame_three = _candidate((16, 16, 56, 56), landmark_value=6.0)

    result = build_tracked_sequence(
        [[frame_zero], [frame_one], [], [frame_three]],
        media=_metadata(),
        processing_width=640,
        processing_height=360,
        frame_rate=25,
        backend="fake",
        device="cpu",
        policy=FaceTrackingQualityPolicy(
            min_face_area_ratio=0.0,
            min_detection_rate=0.75,
            max_missing_run=1,
        ),
    )

    assert result.frame_count == 4
    assert result.detected_frames == 3
    assert result.interpolated_frames == 1
    assert result.detection_rate == pytest.approx(0.75)
    np.testing.assert_allclose(result.landmarks[2], 4.0)
    np.testing.assert_allclose(result.bounding_boxes[2], [14, 14, 54, 54])
    assert np.isnan(result.landmark_scores[2]).all()


def test_save_face_tracking_artifacts_writes_numeric_npz_and_json(tmp_path) -> None:
    sequence = build_tracked_sequence(
        [[_candidate((10, 10, 50, 50))], [_candidate((11, 10, 51, 50))]],
        media=_metadata(),
        processing_width=640,
        processing_height=360,
        frame_rate=25,
        backend="fake",
        device="cpu",
        policy=FaceTrackingQualityPolicy(min_face_area_ratio=0.0),
    )
    artifact_path = tmp_path / "track.npz"
    report_path = tmp_path / "track.json"

    payload = save_face_tracking_artifacts(
        sequence,
        artifact_path=artifact_path,
        report_path=report_path,
    )

    with np.load(artifact_path, allow_pickle=False) as artifact:
        assert artifact["landmarks"].shape == (2, 68, 2)
        assert artifact["bounding_boxes"].shape == (2, 4)
        assert artifact["detected"].tolist() == [True, True]
        assert artifact["original_resolution"].tolist() == [1920, 1080]
        assert artifact["processing_resolution"].tolist() == [640, 360]
        assert artifact["frame_rate"].tolist() == [25]
        assert artifact["quality_passed"].tolist() == [True]
        assert artifact["mouth_visible_raw"].tolist() == [True, True]
        assert artifact["mouth_visible"].tolist() == [True, True]
        assert artifact["artifact_version"].tolist() == [2]

    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["quality_status"] == "passed"
    assert payload["artifact_version"] == 2
    assert payload["landmark_topology"] == "ibug_68"
    assert "identity_switches_prevented" not in payload
    assert saved_report["detection_rate"] == 1.0
    assert saved_report["artifact_path"] == str(artifact_path.resolve())
    assert saved_report["visual_availability"]["coverage"] == 1.0
    assert saved_report["visual_availability"]["missing_frames"] == 0
    assert saved_report["visual_availability"]["missing_intervals"] == []


def test_identity_gate_rejects_distant_face() -> None:
    policy = FaceTrackingQualityPolicy(min_face_area_ratio=0.0)
    previous = np.asarray([10, 10, 110, 110], dtype=np.float32)
    distant = _candidate((1000, 700, 1200, 900))

    decision = decide_tracked_face(
        [distant],
        previous,
        frame_area=1920 * 1080,
        policy=policy,
    )

    assert decision.status == "missing"
    assert decision.reason == "association_gate_failed"
    assert decision.candidate is None


def test_ambiguity_gate_rejects_similarly_sized_initial_faces() -> None:
    policy = FaceTrackingQualityPolicy(min_face_area_ratio=0.0)
    left = _candidate((10, 10, 110, 110))
    right = _candidate((200, 10, 300, 110))

    decision = decide_tracked_face(
        [left, right],
        None,
        frame_area=1920 * 1080,
        policy=policy,
    )

    assert decision.status == "ambiguous"
    assert decision.reason == "initial_faces_have_similar_size"


def test_long_missing_run_fails_sequence_quality() -> None:
    policy = FaceTrackingQualityPolicy(
        min_face_area_ratio=0.0,
        min_detection_rate=0.2,
        max_missing_run=5,
        min_visual_run=1,
    )
    start = _candidate((100, 100, 300, 300), landmark_value=0.0)
    end = _candidate((102, 100, 302, 300), landmark_value=7.0)

    sequence = build_tracked_sequence(
        [[start], [], [], [], [], [], [], [end]],
        media=_metadata(),
        processing_width=640,
        processing_height=360,
        frame_rate=25,
        backend="fake",
        device="cpu",
        policy=policy,
    )

    assert not sequence.quality_passed
    assert sequence.report()["status"] == "degraded"
    assert sequence.report()["quality_status"] == "failed"
    assert "maximum_missing_run_exceeded:6>5" in sequence.quality_issues


def test_load_face_tracking_quality_policy_reads_yaml(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
face_tracking:
  min_detection_rate: 0.95
  max_missing_run: 3
""".strip(),
        encoding="utf-8",
    )

    policy = load_face_tracking_quality_policy(path)

    assert policy.min_detection_rate == 0.95
    assert policy.max_missing_run == 3


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("min_detection_confidence", float("nan")),
        ("max_center_distance_ratio", float("inf")),
        ("min_association_iou", 0.0),
        ("max_missing_run", 1.5),
        ("min_detected_frames", 1),
    ],
)
def test_quality_policy_rejects_invalid_threshold_types_and_values(
    keyword,
    value,
) -> None:
    with pytest.raises(ValueError):
        FaceTrackingQualityPolicy(**{keyword: value})


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        (
            _candidate((10, 10, 110, 110), confidence=float("nan")),
            "non_finite_detection_confidence",
        ),
        (
            _candidate((-200, 10, -100, 110), landmark_value=-150),
            "bounding_box_too_far_outside_frame",
        ),
        (_candidate((10, 10, 20, 20)), "face_area_below_minimum"),
        (
            _candidate((10, 10, 110, 110), landmark_confidence=0.1),
            "landmark_confidence_below_minimum",
        ),
    ],
)
def test_candidate_quality_gate_reports_rejection_reason(candidate, reason) -> None:
    decision = decide_tracked_face(
        [candidate],
        None,
        frame_area=1920 * 1080,
        frame_size=(1920, 1080),
        policy=FaceTrackingQualityPolicy(),
    )

    assert decision.status == "missing"
    assert decision.rejected_reasons == (reason,)


def test_candidate_with_cropped_face_box_is_accepted_when_mouth_is_visible() -> None:
    candidate = _candidate((-50, -50, 150, 150), landmark_value=50)

    decision = decide_tracked_face(
        [candidate],
        None,
        frame_area=1920 * 1080,
        frame_size=(1920, 1080),
        policy=FaceTrackingQualityPolicy(),
    )

    assert decision.status == "accepted"
    assert decision.candidate is candidate


def test_single_speaker_scale_change_is_reacquired() -> None:
    previous = np.asarray([10, 10, 110, 110], dtype=np.float32)
    much_larger = _candidate((-25, -25, 145, 145))

    decision = decide_tracked_face(
        [much_larger],
        previous,
        frame_area=1920 * 1080,
        frame_size=(1920, 1080),
        policy=FaceTrackingQualityPolicy(
            min_face_area_ratio=0.0,
            max_out_of_frame_ratio=1.0,
        ),
    )

    assert decision.status == "accepted"


def test_scale_reacquisition_does_not_bypass_multi_face_identity_gate() -> None:
    previous = np.asarray([10, 10, 110, 110], dtype=np.float32)
    much_larger = _candidate((-25, -25, 145, 145))
    distant_decoy = _candidate((1000, 700, 1100, 800))

    decision = decide_tracked_face(
        [much_larger, distant_decoy],
        previous,
        frame_area=1920 * 1080,
        frame_size=(1920, 1080),
        policy=FaceTrackingQualityPolicy(
            min_face_area_ratio=0.0,
            max_out_of_frame_ratio=1.0,
        ),
    )

    assert decision.status == "missing"
    assert decision.candidate is None
    assert "face_area_change_exceeded" in decision.rejected_reasons
    assert "association_gate_failed" in decision.rejected_reasons


def test_edge_extrapolation_fails_sequence_quality_by_default() -> None:
    face = _candidate((100, 100, 300, 300))
    sequence = build_tracked_sequence(
        [[], [face], [face]],
        media=_metadata(),
        processing_width=640,
        processing_height=360,
        frame_rate=25,
        backend="fake",
        device="cpu",
        policy=FaceTrackingQualityPolicy(
            min_face_area_ratio=0.0,
            min_detection_rate=0.5,
            max_missing_run=1,
        ),
    )

    assert not sequence.quality_passed
    assert "leading_visual_gap_exceeded:1>0" in sequence.quality_issues
    assert sequence.report()["leading_visual_gap"] == 1


def test_tracking_cli_invalidates_stale_artifact_on_tracking_failure(
    tmp_path,
    monkeypatch,
) -> None:
    import sys

    import scripts.track_webcam_faces as tracking_cli

    media = tmp_path / "sample.mp4"
    output_dir = tmp_path / "tracks"
    output_dir.mkdir()
    stale_artifact = output_dir / "sample_face_track.npz"
    stale_artifact.write_bytes(b"stale")

    monkeypatch.setattr(tracking_cli, "FANFaceLandmarker", lambda **_: object())

    def fail_tracking(*args, **kwargs):
        raise MediaInputError("No face was detected.")

    monkeypatch.setattr(tracking_cli, "track_face_landmarks", fail_tracking)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "track_webcam_faces.py",
            "--media",
            str(media),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert tracking_cli.main() == 1
    assert not stale_artifact.exists()
    report = json.loads(
        (output_dir / "sample_face_track.json").read_text(encoding="utf-8")
    )
    assert report["quality_status"] == "failed"
    assert report["quality_issues"] == ["tracking_failed"]
