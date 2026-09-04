import base64
import gc
import html
import importlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

SRC = Path(__file__).resolve().parents[2]
ROOT = SRC.parent
_src = str(SRC)
if _src in sys.path:
    sys.path.remove(_src)
sys.path.insert(0, _src)


def _ensure_repo_viavsr() -> None:
    """Prefer this repository over a stale pip-installed viavsr package."""
    src = SRC.resolve()
    src_str = str(src)
    if sys.path[0] != src_str:
        if src_str in sys.path:
            sys.path.remove(src_str)
        sys.path.insert(0, src_str)

    pkg_init = (src / "viavsr" / "__init__.py").resolve()
    expected_demo = (src / "viavsr" / "demo.py").resolve()
    if not pkg_init.is_file():
        return

    def _origin(module_name: str) -> Path | None:
        module = sys.modules.get(module_name)
        if module is None:
            return None
        module_file = getattr(module, "__file__", None)
        if not module_file:
            return None
        return Path(module_file).resolve()

    stale = False
    pkg_origin = _origin("viavsr")
    if pkg_origin is not None and pkg_origin != pkg_init:
        stale = True
    demo_origin = _origin("viavsr.demo")
    if demo_origin is not None and demo_origin != expected_demo:
        stale = True

    if stale:
        for name in list(sys.modules):
            if name == "viavsr" or name.startswith("viavsr."):
                del sys.modules[name]
        importlib.invalidate_caches()


_ensure_repo_viavsr()

CONFIG = ROOT / "configs" / "config.yaml"
UPLOAD_DIR = ROOT / "uploads"
SAMPLE_DIR = ROOT / "samples" / "webcam"
OUTPUT_ROOT = ROOT / "outputs" / "demo"
DEFAULT_CLIP_SECONDS = 8
FAST_CLIP_SECONDS = 5
DURATION_SLACK = 0.5
ON_CLOUD = Path("/mount/src").is_dir()
CLOUD_MAX_WIDTH = 360
LOCAL_MAX_WIDTH = 480
TORCH_THREADS = 2 if ON_CLOUD else min(4, os.cpu_count() or 4)
_TORCH_SPEED_CONFIGURED = False

if ON_CLOUD:
    os.environ.setdefault("OMP_NUM_THREADS", str(TORCH_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(TORCH_THREADS))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

st.set_page_config(
    page_title="viAVSR",
    layout="wide",
    initial_sidebar_state="expanded",
)

RECORDER = components.declare_component(
    "webcam_recorder",
    path=str(Path(__file__).parent / "webcam_recorder"),
)


def _bootstrap_hf_credentials() -> None:
    """Expose HF token to huggingface_hub from env, Streamlit secrets, or CLI login."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return
    try:
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "hf_token"):
            value = st.secrets.get(name)
            if value:
                os.environ.setdefault("HF_TOKEN", str(value))
                return
    except (FileNotFoundError, KeyError, AttributeError, TypeError):
        pass


def _hf_credentials_ready() -> bool:
    _bootstrap_hf_credentials()
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    try:
        return (
            token_file.is_file()
            and token_file.read_text(encoding="utf-8").strip() != ""
        )
    except OSError:
        return token_file.is_file()


_bootstrap_hf_credentials()

st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;1,9..144,400&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    /* Streamlit chrome che kicker — ẩn header/footer mặc định */
    header[data-testid="stHeader"],
    footer,
    [data-testid="stDecoration"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        min-height: 0 !important;
    }
    .stApp {
        background-color: #ffffff;
        background-image: radial-gradient(#e8e6e1 0.65px, transparent 0.65px);
        background-size: 18px 18px;
    }
    .block-container {
        padding-top: 2.25rem !important;
        padding-bottom: 4rem !important;
        max-width: 1040px;
    }
    html, body, [class*="css"] {
        font-family: "DM Sans", sans-serif;
        color: #1a1a1a;
    }

    /* Sidebar */
    div[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e2dc;
    }
    div[data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }
    div[data-testid="stSidebar"] .stMarkdown h3 {
        font-family: "Fraunces", serif;
        font-weight: 600;
        font-size: 1.35rem;
        letter-spacing: -0.02em;
    }

    /* Hero */
    .hero {
        margin-bottom: 2.75rem;
        padding-top: 0.25rem;
    }
    .hero-eyebrow {
        margin: 0 0 1.1rem;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #c45c26;
    }
    .hero-title {
        font-family: "Fraunces", serif;
        font-size: clamp(3.25rem, 9vw, 5.5rem);
        font-weight: 600;
        line-height: 0.92;
        margin: 0 0 1rem;
        letter-spacing: -0.045em;
        color: #1a1a1a;
    }
    .hero-title em {
        font-style: italic;
        font-weight: 300;
        color: #c45c26;
    }
    .hero-tagline {
        margin: 0 0 1.75rem;
        font-size: 1.08rem;
        line-height: 1.6;
        color: #5c5c58;
        max-width: 28rem;
    }
    .hero-rule {
        height: 3px;
        background: #1a1a1a;
        position: relative;
    }
    .hero-rule::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        width: 88px;
        height: 3px;
        background: #c45c26;
    }

    /* Sections */
    .section-label {
        display: flex;
        align-items: baseline;
        gap: 0.65rem;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #8a8a86;
        margin: 0 0 1rem;
    }
    .section-num {
        font-family: "Fraunces", serif;
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 0;
        color: #c45c26;
        text-transform: none;
    }

    /* Panels — Streamlit bordered containers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #e5e2dc !important;
        border-radius: 2px !important;
        background: #ffffff !important;
        box-shadow: 0 8px 32px rgba(26, 26, 26, 0.04) !important;
        padding: 0.25rem 0.5rem !important;
    }

    .viavsr-empty {
        border: 1px dashed #d8d4cc;
        background: #faf9f7;
        padding: 3rem 1.5rem;
        text-align: center;
        color: #9a9690;
        font-size: 0.9rem;
        min-height: 180px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* Transcript */
    .viavsr-transcript-wrap {
        margin: 1.25rem 0 1.75rem;
    }
    .viavsr-transcript {
        font-family: "Fraunces", serif;
        font-size: clamp(1.55rem, 2.8vw, 2.35rem);
        font-weight: 400;
        line-height: 1.48;
        color: #1a1a1a;
        padding: 2rem 2rem 2rem 2.25rem;
        margin: 0;
        background: #ffffff;
        border: 2px solid #1a1a1a;
        box-shadow: 6px 6px 0 #c45c26;
    }
    .viavsr-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem 1.5rem;
        font-size: 0.78rem;
        color: #6b6b68;
        padding-bottom: 0.25rem;
    }
    .viavsr-meta span { white-space: nowrap; }
    .viavsr-meta strong {
        color: #1a1a1a;
        font-weight: 600;
        text-transform: capitalize;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: #faf9f7;
        border: 1px solid #e5e2dc;
        padding: 0.85rem 1rem;
        border-radius: 2px;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.65rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: #8a8a86 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-family: "Fraunces", serif !important;
        font-size: 1.35rem !important;
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: #1a1a1a !important;
        color: #fff !important;
        border: 2px solid #1a1a1a !important;
        border-radius: 2px !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        font-size: 0.82rem !important;
        padding: 0.85rem 2.5rem !important;
        transition: all 0.15s ease !important;
        box-shadow: 4px 4px 0 #c45c26 !important;
    }
    .stButton > button[kind="primary"]:hover:not(:disabled) {
        background: #c45c26 !important;
        border-color: #c45c26 !important;
        transform: translate(2px, 2px) !important;
        box-shadow: 2px 2px 0 #1a1a1a !important;
    }
    .stButton > button[kind="primary"]:disabled {
        opacity: 0.35 !important;
        box-shadow: none !important;
    }

    /* Radio / tabs feel */
    div[data-testid="stRadio"] > div {
        gap: 0.5rem;
    }
    hr { border: none; border-top: 1px solid #e5e2dc; margin: 2.5rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1 class="hero-title"><em>vi</em>AVSR</h1>
  <div class="hero-rule"></div>
</div>
""",
    unsafe_allow_html=True,
)


def _tool(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    prefix = Path(os.environ.get("CONDA_PREFIX", ""))
    for candidate in (
        prefix / "Library" / "bin" / f"{name}.exe",
        Path.home()
        / "miniconda3"
        / "envs"
        / "viavsr"
        / "Library"
        / "bin"
        / f"{name}.exe",
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def _save_upload(name: str, data: bytes) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / name
    path.write_bytes(data)
    return path


def _data_url_to_webm(data_url: str) -> Path:
    marker = "base64,"
    encoded = (
        data_url[data_url.find(marker) + len(marker) :]
        if marker in data_url
        else data_url.rsplit(",", 1)[-1]
    )
    encoded = encoded.strip().replace("\n", "").replace("\r", "").replace(" ", "+")
    encoded += "=" * ((4 - len(encoded) % 4) % 4)
    return _save_upload("webcam_record.webm", base64.b64decode(encoded))


def _duration_seconds(src: Path) -> float | None:
    ffprobe = _tool("ffprobe")
    if ffprobe is None:
        return None
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(src),
        ],
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        return None
    try:
        duration = float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return duration


@st.cache_resource(show_spinner=False)
def _ensure_tokenizer_assets() -> None:
    """Download pinned tokenizer files on first run (not committed to git)."""
    _ensure_repo_viavsr()
    from viavsr.inference import load_model_assets_config
    from viavsr.inference.tokenizer_assets import fetch_tokenizer_assets

    config = load_model_assets_config(CONFIG)
    fetch_tokenizer_assets(config.tokenizer_model_path, config.tokenizer_units_path)


def _configure_torch_speed() -> None:
    global _TORCH_SPEED_CONFIGURED
    if _TORCH_SPEED_CONFIGURED:
        return
    import torch

    torch.set_num_threads(TORCH_THREADS)
    try:
        torch.set_num_interop_threads(max(1, TORCH_THREADS // 2))
    except RuntimeError:
        pass
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("medium")
    _TORCH_SPEED_CONFIGURED = True


_configure_torch_speed()


@st.cache_resource(show_spinner="Đang tải model…")
def _load_cached_model_assets():
    """Keep one model instance in memory across reruns (saves ~1.7 GB per inference)."""
    _ensure_repo_viavsr()
    _configure_torch_speed()
    from viavsr.inference import load_model_assets_config, load_vietnamese_avsr_assets

    _ensure_tokenizer_assets()
    config = load_model_assets_config(CONFIG)
    return load_vietnamese_avsr_assets(config)


@st.cache_resource(show_spinner="Đang tải bộ theo dõi khuôn mặt…")
def _load_cached_face_landmarker():
    """Keep one RetinaFace/FAN instance in memory across visual demo runs."""
    _ensure_repo_viavsr()
    from viavsr.preprocessing import (
        FANFaceLandmarker,
        load_face_tracking_quality_policy,
    )

    policy = load_face_tracking_quality_policy(CONFIG)
    return FANFaceLandmarker(
        device="auto",
        confidence_threshold=policy.min_detection_confidence,
    )


@st.cache_resource(show_spinner=False)
def _startup_warmup(include_visual: bool) -> bool:
    """Load reusable model and optional visual backend before the first run."""
    _load_cached_model_assets()
    if include_visual:
        _load_cached_face_landmarker()
    return True


def _prepare_media(
    src: Path,
    clip_seconds: float,
    *,
    max_width: int,
    audio_only: bool = False,
    fast: bool = False,
) -> Path:
    if audio_only:
        return _prepare_media_fast_audio(src, clip_seconds)
    ffmpeg = _tool("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg was not found. Activate the viavsr conda env and retry."
        )
    dest = UPLOAD_DIR / f"{src.stem}_prep.mp4"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    video_codec = (
        ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p"]
        if fast
        else ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    )
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-ss",
            "0",
            "-t",
            str(clip_seconds),
            "-i",
            str(src),
            "-vf",
            f"scale='min({max_width},iw)':-2",
            "-r",
            "25",
            *video_codec,
            "-c:a",
            "aac",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(dest),
        ],
        capture_output=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0 or not dest.is_file():
        detail = result.stderr.decode("utf-8", errors="replace")[-400:]
        raise RuntimeError(f"Could not trim/scale the video.\n{detail}")
    return dest


def _prepare_media_fast_audio(src: Path, clip_seconds: float) -> Path:
    """Trim clip with stream copy when possible; minimal re-encode otherwise."""
    ffmpeg = _tool("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg was not found. Activate the viavsr conda env and retry."
        )
    dest = UPLOAD_DIR / f"{src.stem}_fast.mp4"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    copy_cmd = [
        ffmpeg,
        "-y",
        "-ss",
        "0",
        "-t",
        str(clip_seconds),
        "-i",
        str(src),
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(dest),
    ]
    result = subprocess.run(
        copy_cmd,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode == 0 and dest.is_file():
        return dest

    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-ss",
            "0",
            "-t",
            str(clip_seconds),
            "-i",
            str(src),
            "-map",
            "0:v:0?",
            "-map",
            "0:a:0",
            "-vf",
            "scale=96:96:force_original_aspect_ratio=decrease,"
            "pad=96:96:(ow-iw)/2:(oh-ih)/2",
            "-r",
            "25",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-crf",
            "30",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-shortest",
            str(dest),
        ],
        capture_output=True,
        check=False,
        timeout=45,
    )
    if result.returncode != 0 or not dest.is_file():
        detail = result.stderr.decode("utf-8", errors="replace")[-400:]
        raise RuntimeError(f"Could not prepare fast audio media.\n{detail}")
    return dest


def _import_demo_runner():
    """Always use demo.py from this repository (not a stale pip install)."""
    _ensure_repo_viavsr()
    demo_path = (SRC / "viavsr" / "demo.py").resolve()
    if not demo_path.is_file():
        from viavsr.demo import run_end_to_end_demo

        return run_end_to_end_demo

    import importlib.util

    if "viavsr" not in sys.modules:
        importlib.import_module("viavsr")

    spec = importlib.util.spec_from_file_location(
        "viavsr.demo",
        demo_path,
        submodule_search_locations=[str(SRC / "viavsr")],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Không load được {demo_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["viavsr.demo"] = module
    spec.loader.exec_module(module)
    run_end_to_end_demo = module.run_end_to_end_demo

    if "skip_face_tracking" not in inspect.signature(run_end_to_end_demo).parameters:
        raise RuntimeError(
            f"File {demo_path} thiếu tham số skip_face_tracking. "
            "Chạy `pip install -e .` trong thư mục dự án rồi khởi động lại Streamlit."
        )
    return run_end_to_end_demo


def _call_run_end_to_end_demo(
    *,
    audio_only: bool,
    preloaded_assets: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    run_end_to_end_demo = _import_demo_runner()
    params = inspect.signature(run_end_to_end_demo).parameters
    call_kwargs = {
        **kwargs,
        "skip_face_tracking": audio_only,
        "preloaded_assets": preloaded_assets,
    }
    supported = {key: value for key, value in call_kwargs.items() if key in params}
    if audio_only and "skip_face_tracking" not in params:
        raise RuntimeError(
            "demo.py trên máy chưa có tham số skip_face_tracking. "
            "Cập nhật code (git pull) hoặc chạy `pip install -e .`, "
            "rồi khởi động lại Streamlit."
        )
    return run_end_to_end_demo(**supported)


with st.sidebar:
    st.markdown("### Tuỳ chọn")
    audio_only = st.toggle(
        "Chỉ audio",
        value=ON_CLOUD,
        help="Không dùng hình miệng — nhanh hơn nhiều.",
    )
    visual_fallback_policy = st.selectbox(
        "Xử lý khi mất hình",
        ("whole_utterance", "corrupted_av", "interval_gated"),
        index=0,
        disabled=audio_only,
        format_func=lambda value: {
            "whole_utterance": "Fallback toàn câu về audio",
            "corrupted_av": "Corrupted AV",
            "interval_gated": "Interval-gated AV",
        }[value],
        help="Chọn cách xử lý khi chỉ một số khoảng hình miệng không khả dụng.",
    )
    fast_mode = st.toggle(
        "Rút gọn",
        value=True,
        disabled=audio_only,
        help="Clip 5 giây, encode nhẹ hơn.",
    )
    clip_seconds = float(
        st.slider(
            "Độ dài clip",
            min_value=3,
            max_value=8,
            value=FAST_CLIP_SECONDS if fast_mode else DEFAULT_CLIP_SECONDS,
            disabled=fast_mode and not audio_only,
        )
    )
    max_detection = 256 if fast_mode else 320
    max_width = CLOUD_MAX_WIDTH if ON_CLOUD else LOCAL_MAX_WIDTH
    decoder = st.selectbox(
        "Decoder",
        ("ctc_greedy", "joint_beam_search"),
        index=1,
        format_func=lambda value: {
            "ctc_greedy": "CTC greedy (nhanh)",
            "joint_beam_search": "Joint CTC/Attention (khuyến nghị)",
        }[value],
        help="Joint CTC/Attention là mặc định; CTC greedy là chế độ nhanh.",
    )
    if not _hf_credentials_ready():
        if ON_CLOUD:
            st.warning("Cần **HF_TOKEN** trong Streamlit Secrets (Cloud).")
        else:
            st.warning(
                "Chưa thấy HF token. Set `$env:HF_TOKEN='hf_...'` **trong cùng terminal** "
                "rồi chạy lại Streamlit, hoặc `huggingface-cli login`, "
                "hoặc thêm `HF_TOKEN` vào `.streamlit/secrets.toml`."
            )
    _startup_warmup(include_visual=not audio_only)

input_col, preview_col = st.columns([1, 1], gap="large")

with input_col:
    with st.container(border=True):
        st.markdown(
            '<p class="section-label"><span class="section-num">01</span> Nguồn</p>',
            unsafe_allow_html=True,
        )
        source = st.radio(
            "Chọn nguồn",
            ("Upload", "Webcam", "Sample"),
            horizontal=True,
            label_visibility="collapsed",
        )
        media_path: Path | None = st.session_state.get("media_path")

        if source == "Upload":
            uploaded = st.file_uploader(
                "Video có âm thanh",
                type=["mp4", "webm", "mov"],
                label_visibility="collapsed",
            )
            if uploaded is not None:
                media_path = _save_upload(uploaded.name, uploaded.getvalue())
                st.session_state.media_path = media_path

        elif source == "Webcam":
            recorded = RECORDER(key="webcam")
            if recorded:
                try:
                    media_path = _data_url_to_webm(recorded)
                    st.session_state.media_path = media_path
                except Exception as exc:
                    st.error(f"Không đọc được bản ghi webcam: {exc}")
                    media_path = None

        else:
            samples = sorted(SAMPLE_DIR.glob("vi_*.mp4"))
            if not samples:
                st.info("Chưa có mẫu tiếng Việt. Dùng Upload hoặc Webcam.")
            else:
                choice = st.selectbox(
                    "Mẫu tiếng Việt",
                    samples,
                    format_func=lambda p: p.name,
                    label_visibility="collapsed",
                )
                if choice is not None:
                    media_path = choice
                    st.session_state.media_path = media_path

with preview_col:
    with st.container(border=True):
        st.markdown(
            '<p class="section-label"><span class="section-num">02</span> Xem trước</p>',
            unsafe_allow_html=True,
        )
        if media_path and media_path.is_file():
            duration = _duration_seconds(media_path)
            if duration is not None and duration > clip_seconds:
                st.warning(
                    f"File dài {duration:.1f}s — chỉ dùng {clip_seconds:.0f}s đầu."
                )
            st.video(str(media_path))
        else:
            st.markdown(
                '<div class="viavsr-empty">Chưa có video</div>',
                unsafe_allow_html=True,
            )

_, btn_col, _ = st.columns([1, 1.2, 1])
with btn_col:
    run = st.button(
        "Chạy nhận dạng",
        type="primary",
        use_container_width=True,
        disabled=not (media_path and media_path.is_file()),
    )

if run and media_path and media_path.is_file():
    try:
        with st.status("Đang xử lý…", expanded=True) as status:
            status.write("Chuẩn bị tokenizer…")
            _ensure_tokenizer_assets()
            status.write("Xử lý video…")
            prepared = _prepare_media(
                media_path,
                clip_seconds,
                max_width=max_width,
                audio_only=audio_only,
                fast=fast_mode,
            )
            status.write(
                "Nhận dạng…" if audio_only else "Theo dõi khuôn mặt và nhận dạng…"
            )
            assets = _load_cached_model_assets()
            landmarker = None if audio_only else _load_cached_face_landmarker()

            result = _call_run_end_to_end_demo(
                audio_only=audio_only,
                preloaded_assets=assets,
                preloaded_landmarker=landmarker,
                config_path=CONFIG,
                media_path=prepared,
                output_root=OUTPUT_ROOT,
                tracking_device="auto",
                decoder=decoder,
                max_duration_seconds=float(clip_seconds + DURATION_SLACK),
                max_detection_size=max_detection,
                detection_stride=2 if fast_mode else 1,
                visual_fallback_policy=visual_fallback_policy,
            )
            status.write("Xong.")
            status.update(label="Hoàn tất", state="complete")
        st.session_state.result = result
        gc.collect()
    except Exception as exc:
        st.error(str(exc))

if "result" in st.session_state:
    result = st.session_state.result
    inner = result.get("result") or {}
    modality = result.get("modality_decision") or {}
    artifacts = result.get("artifacts") or {}
    timings = result.get("timings_seconds") or {}
    status = result.get("status", "")
    transcript = inner.get("transcript") or "(trống)"

    with st.container(border=True):
        st.markdown(
            '<p class="section-label"><span class="section-num">03</span> Kết quả</p>',
            unsafe_allow_html=True,
        )

        mode = modality.get("selected_mode", "—")
        total_s = timings.get("total", 0)

        st.markdown(
            f'<div class="viavsr-meta">'
            f"<span>Trạng thái · <strong>{html.escape(status)}</strong></span>"
            f"<span>Chế độ · <strong>{html.escape(mode.replace('_', ' '))}</strong></span>"
            f"<span>{total_s:.1f}s</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="viavsr-transcript-wrap">'
            f'<p class="viavsr-transcript">{html.escape(transcript)}</p>'
            f"</div>",
            unsafe_allow_html=True,
        )

        metric_cols = st.columns(4)
        metric_cols[0].metric("Tổng", f"{total_s:.1f}s")
        metric_cols[1].metric("Modality", mode.replace("_", " "))
        metric_cols[2].metric("Model", f"{timings.get('model_loading', 0):.1f}s")
        metric_cols[3].metric("Inference", f"{timings.get('inference', 0):.1f}s")

        step_keys = ("face_tracking", "mouth_roi", "model_loading", "inference")
        step_parts = [
            f"{key.replace('_', ' ')}: {timings[key]:.1f}s"
            for key in step_keys
            if key in timings and timings[key]
        ]
        if step_parts:
            st.caption(" · ".join(step_parts))

        mouth = artifacts.get("mouth_roi")
        if mouth and Path(mouth).is_file():
            with st.expander("Video miệng"):
                st.video(mouth)

        warnings = result.get("warnings") or []
        if warnings:
            st.warning("\n".join(str(item) for item in warnings))
        if status != "passed" and result.get("error"):
            st.error(result["error"].get("message", "Inference failed"))

        with st.expander("Báo cáo JSON"):
            st.json(result)
