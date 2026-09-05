"""Exercise UI state with model loading and media probing replaced by test doubles."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "src/viavsr/ui/app.py"


@pytest.fixture
def ui(monkeypatch, tmp_path):
    import streamlit as st
    import viavsr.inference as inference
    import viavsr.inference.tokenizer_assets as tokenizer_assets
    import viavsr.preprocessing as preprocessing

    st.cache_resource.clear()
    st.cache_data.clear()
    media = tmp_path / "example.mp4"
    media.write_bytes(b"UI test media")
    original_glob = Path.glob

    def sample_glob(path, pattern):
        if path == ROOT / "samples/webcam" and pattern == "*.mp4":
            return iter([media])
        return original_glob(path, pattern)

    monkeypatch.setattr(Path, "glob", sample_glob)
    monkeypatch.setenv("HF_TOKEN", "test-ui-placeholder")
    monkeypatch.setattr(inference, "load_vietnamese_avsr_assets", lambda *_: object())
    monkeypatch.setattr(preprocessing, "FANFaceLandmarker", lambda **_: object())
    monkeypatch.setattr(tokenizer_assets, "fetch_tokenizer_assets", lambda *_: None)
    monkeypatch.setattr(
        preprocessing, "probe_av_media",
        lambda *_: SimpleNamespace(to_dict=lambda: {
            "duration_seconds": 4.0, "audio_sample_rate": 16000,
            "frame_rate": 25.0, "video_width": 640, "video_height": 480,
        }),
    )
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    yield app, media
    st.cache_resource.clear()
    st.cache_data.clear()


def seed_result(app, media, *, passed=True, audio_only=False):
    file_stat = media.stat()
    app.session_state.result_request_key = (
        str(media.resolve()), file_stat.st_mtime_ns, file_stat.st_size,
        audio_only, "whole_utterance", True, 5.0, "joint_beam_search", "",
    )
    app.session_state.result = {
        "status": "passed" if passed else "failed",
        "result": {
            "transcript": "xin chào", "decoder": "joint_beam_search",
            "device": "cpu", "hypothesis_score": -12.0, "beam_size": 3,
        } if passed else {},
        "modality_decision": {
            "selected_mode": "audio_only_fallback" if audio_only else "audio_visual_corrupted",
        },
        "mouth_roi_display": {
            "mouth_roi_size": 96, "output_media": {"frame_rate": 25},
            "visual_availability": {
                "frame_count": 100, "valid_frames": 75, "missing_frames": 25,
                "coverage": 0.75, "missing_intervals": [{
                    "start_seconds": 1.0, "end_seconds": 2.0, "frame_count": 25,
                }],
            },
        } if not audio_only else {},
        "timings_seconds": {"total": 6.0, "inference": 4.0, "face_tracking": 1.0},
        "evaluation": {"wer": 0.1, "cer": 0.05} if passed else None,
        "error": {"message": "Example preprocessing failure"} if not passed else None,
    }


def text(app):
    return "\n".join(element.value for element in app.markdown)


def test_three_panels_and_empty_state(ui):
    app, _ = ui
    output = text(app)
    assert all(title in output for title in (
        "1. Original Input", "2. Processed Mouth ROI", "3. Recognition Result",
    ))
    assert "Run recognition to see" in output
    assert app.button[0].disabled


def test_metrics_and_stale_result_invalidated_when_options_change(ui):
    app, media = ui
    app.radio[0].set_value("Sample").run()
    seed_result(app, media)
    app.run()
    assert not app.exception
    assert "xin chào" in text(app)
    assert "75 / 100 (75.0%)" in text(app)
    assert [metric.value for metric in app.metric] == ["10.00%", "5.00%"]
    assert "not a calibrated confidence" in "\n".join(x.value for x in app.caption)
    app.toggle[0].set_value(True).run()
    assert not app.exception
    assert "xin chào" not in text(app)
    assert len(app.metric) == 0


def test_failed_run_shows_error_without_transcript_or_scores(ui):
    app, media = ui
    app.radio[0].set_value("Sample").run()
    seed_result(app, media, passed=False)
    app.run()
    assert not app.exception
    assert app.error[0].value == "Example preprocessing failure"
    assert "xin chào" not in text(app)
    assert len(app.metric) == 0


def test_audio_only_without_roi_is_rendered_without_invented_coverage(ui):
    app, media = ui
    app.radio[0].set_value("Sample").run()
    app.toggle[0].set_value(True).run()
    seed_result(app, media, audio_only=True)
    app.run()
    assert not app.exception
    assert "Audio-only fallback" in text(app)
    assert "75 / 100" not in text(app)
    assert "Mouth ROI is unavailable" in text(app)
