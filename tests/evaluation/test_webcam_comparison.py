from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from viavsr.evaluation import webcam_comparison
from viavsr.preprocessing.face_tracking import FaceTrackingQualityPolicy
from viavsr.preprocessing.media import MediaMetadata, PreparedAVInput


def _metadata(path: Path, *, mouth_roi: bool = False) -> MediaMetadata:
    size = 96 if mouth_roi else 1280
    height = 96 if mouth_roi else 720
    return MediaMetadata(
        path=str(path),
        duration_seconds=0.24,
        video_width=size,
        video_height=height,
        frame_rate=25.0,
        audio_sample_rate=16_000,
        audio_channels=1,
    )


def test_paired_comparison_reuses_tracking_tensors_mask_and_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    media = tmp_path / "webcam.mp4"
    media.write_bytes(b"raw")
    mask = np.array([True, True, False, False, True, True])
    calls: list[str] = []

    monkeypatch.setattr(
        webcam_comparison,
        "load_face_tracking_quality_policy",
        lambda _path: FaceTrackingQualityPolicy(),
    )
    monkeypatch.setattr(
        webcam_comparison,
        "probe_av_media",
        lambda _path: _metadata(media),
    )
    monkeypatch.setattr(
        webcam_comparison,
        "FANFaceLandmarker",
        lambda **_kwargs: SimpleNamespace(),
    )
    sequence = SimpleNamespace(mouth_visible=mask)

    def track(*_args, **_kwargs):
        calls.append("track")
        return sequence

    monkeypatch.setattr(webcam_comparison, "track_face_landmarks", track)

    def save_track(_sequence, *, artifact_path: Path, report_path: Path):
        calls.append("save_track")
        artifact_path.write_bytes(b"track")
        report_path.write_text("{}", encoding="utf-8")
        return {
            "artifact_path": str(artifact_path),
            "report_path": str(report_path),
            "visual_coverage": float(mask.mean()),
        }

    monkeypatch.setattr(
        webcam_comparison,
        "save_face_tracking_artifacts",
        save_track,
    )

    def export_display(_media, output: Path, *, track_path: Path):
        calls.append("display")
        assert track_path.is_file()
        output.write_bytes(b"display")
        return SimpleNamespace(
            to_dict=lambda: {
                "output_path": str(output),
                "visual_availability": {
                    "frame_count": 6,
                    "valid_frames": 4,
                    "missing_frames": 2,
                    "coverage": 4 / 6,
                },
            }
        )

    monkeypatch.setattr(
        webcam_comparison,
        "export_mouth_roi_display_video",
        export_display,
    )

    def export_inference_roi(
        _media,
        track_path: Path,
        output: Path,
        *,
        require_quality_passed: bool,
    ):
        calls.append("export_inference_roi")
        assert track_path.is_file()
        assert require_quality_passed is False
        output.write_bytes(b"roi")

    monkeypatch.setattr(
        webcam_comparison,
        "export_aligned_mouth_roi_video",
        export_inference_roi,
    )

    prepared = PreparedAVInput(
        videos=torch.ones((1, 1, 6, 88, 88)),
        audios=torch.zeros((1, 104, 6)),
        video_lengths=torch.tensor([6]),
        audio_lengths=torch.tensor([6]),
        metadata=_metadata(tmp_path / "mouth96.mp4", mouth_roi=True),
        visual_availability=torch.from_numpy(mask).reshape(1, -1),
    )

    def prepare(_path, *, max_duration_seconds, visual_availability):
        calls.append("prepare")
        assert max_duration_seconds == 15.0
        assert np.array_equal(visual_availability, mask)
        return prepared

    monkeypatch.setattr(webcam_comparison, "prepare_mouth_roi_media", prepare)
    monkeypatch.setattr(
        webcam_comparison,
        "load_model_assets_config",
        lambda _path: calls.append("load_config") or object(),
    )
    assets = SimpleNamespace(
        report=SimpleNamespace(
            to_dict=lambda: {
                "repository_id": "example/vi-model",
                "model_revision": "abc123",
            }
        )
    )
    monkeypatch.setattr(
        webcam_comparison,
        "load_vietnamese_avsr_assets",
        lambda _config: calls.append("load_model") or assets,
    )
    transcripts = {
        "audio_visual_corrupted": "xin sao",
        "audio_visual_interval_gated": "xin chào",
        "audio_only_experimental": "sai",
    }
    seen_prepared_ids: list[int] = []
    seen_modes: list[str] = []

    def recognize(actual_assets, actual_prepared, **kwargs):
        assert actual_assets is assets
        seen_prepared_ids.append(id(actual_prepared))
        mode = kwargs["inference_mode"]
        seen_modes.append(mode)
        transcript = transcripts[mode]
        return SimpleNamespace(
            transcript=transcript,
            to_dict=lambda: {
                "transcript": transcript,
                "token_ids": [1, 2],
                "inference_mode": mode,
            },
        )

    monkeypatch.setattr(webcam_comparison, "recognize_prepared_av", recognize)

    payload = webcam_comparison.run_paired_webcam_comparison(
        config_path=tmp_path / "config.yaml",
        media_path=media,
        output_root=tmp_path / "outputs",
        reference_text="xin chào",
    )

    paths = webcam_comparison.PairedWebcamArtifactPaths.for_media(
        media, tmp_path / "outputs"
    )
    assert payload["status"] == "passed"
    assert payload["shared_input"]["single_preprocessing_pass"] is True
    assert payload["shared_input"]["visual_masked_frames"] == 2
    assert len(payload["shared_input"]["fingerprints"]["video_tensor_sha256"]) == 64
    assert seen_prepared_ids == [id(prepared)] * 3
    assert seen_modes == [
        "audio_visual_corrupted",
        "audio_visual_interval_gated",
        "audio_only_experimental",
    ]
    assert payload["comparison"]["best_conditions"] == ["interval_gated"]
    assert "token_ids" not in payload["conditions"]["corrupted_av"]["result"]
    assert "visual_coverage" not in payload["conditions"]["audio_only"]["result"]
    assert "/.work/" not in json.dumps(payload)
    assert calls.count("track") == 1
    assert calls.count("prepare") == 1
    assert calls.count("load_model") == 1
    assert not paths.work_directory.exists()
    assert paths.mouth_roi_display.is_file()
    assert json.loads(paths.report.read_text(encoding="utf-8")) == payload
