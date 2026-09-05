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
    /* Keep the header mounted so the collapsed sidebar can be reopened. */
    header[data-testid="stHeader"] {
        visibility: visible !important;
        height: 3rem !important;
        min-height: 3rem !important;
        background: transparent !important;
    }
    footer,
    [data-testid="stDecoration"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        min-height: 0 !important;
    }
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        display: flex !important;
        visibility: visible !important;
    }
    .stApp {
        background-color: #ffffff;
        background-image: radial-gradient(#e8e6e1 0.65px, transparent 0.65px);
        background-size: 18px 18px;
    }
    .block-container {
        padding-top: 2.25rem !important;
        padding-bottom: 4rem !important;
        max-width: 1560px;
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
        margin-bottom: 1.25rem;
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
        font-size: clamp(2.25rem, 5vw, 3.5rem);
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

    /* Panels - Streamlit bordered containers */
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
        font-size: clamp(1.2rem, 1.7vw, 1.65rem);
        font-weight: 400;
        line-height: 1.48;
        color: #1a1a1a;
        padding: 1.25rem;
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

    .viavsr-details { margin: 1rem 0 0; }
    .viavsr-detail {
        display: flex; justify-content: space-between; gap: 1rem;
        padding: 0.45rem 0; border-bottom: 1px solid #eeece8;
        font-size: 0.83rem; line-height: 1.5;
    }
    .viavsr-detail dt { color: #6b6b68; }
    .viavsr-detail dd {
        margin: 0; text-align: right; color: #1a1a1a;
        font-weight: 500; overflow-wrap: anywhere;
    }
    [data-testid="stVideo"] video {
        width: 100%; height: 280px; object-fit: contain; background: #151515;
    }
    .viavsr-empty { min-height: 280px; }
    @media (max-width: 900px) {
        [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
        [data-testid="stColumn"] { min-width: min(100%, 320px); flex: 1 1 320px; }
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
    if not path.is_file() or path.read_bytes() != data:
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


@st.cache_resource(show_spinner="Loading model...")
def _load_cached_model_assets():
    """Keep one model instance in memory across reruns (saves ~1.7 GB per inference)."""
    _ensure_repo_viavsr()
    _configure_torch_speed()
    from viavsr.inference import load_model_assets_config, load_vietnamese_avsr_assets

    _ensure_tokenizer_assets()
    config = load_model_assets_config(CONFIG)
    return load_vietnamese_avsr_assets(config)


@st.cache_resource(show_spinner="Loading face tracker...")
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
        raise RuntimeError(f"Could not load {demo_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["viavsr.demo"] = module
    spec.loader.exec_module(module)
    run_end_to_end_demo = module.run_end_to_end_demo

    if "skip_face_tracking" not in inspect.signature(run_end_to_end_demo).parameters:
        raise RuntimeError(
            f"File {demo_path} does not support skip_face_tracking. "
            "Run `pip install -e .` from the project directory, then restart Streamlit."
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
            "The local demo.py does not support skip_face_tracking. "
            "Update the code with `git pull` or run `pip install -e .`, "
            "then restart Streamlit."
        )
    return run_end_to_end_demo(**supported)


with st.sidebar:
    st.markdown("### Options")
    audio_only = st.toggle(
        "Audio only",
        value=ON_CLOUD,
        help="Skip mouth-video processing for a faster run.",
    )
    visual_fallback_policy = st.selectbox(
        "Missing-visual strategy",
        ("whole_utterance", "corrupted_av", "interval_gated"),
        index=0,
        disabled=audio_only,
        format_func=lambda value: {
            "whole_utterance": "Audio-only fallback for the full utterance",
            "corrupted_av": "Corrupted AV",
            "interval_gated": "Interval-gated AV",
        }[value],
        help="Choose how to handle intervals where mouth frames are unavailable.",
    )
    fast_mode = st.toggle(
        "Fast mode",
        value=True,
        disabled=audio_only,
        help="Use a 5-second clip and lighter encoding.",
    )
    clip_seconds = float(
        st.slider(
            "Clip duration",
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
            "ctc_greedy": "CTC greedy (fast)",
            "joint_beam_search": "Joint CTC/Attention",
        }[value],
        help="Joint CTC/Attention is the default; CTC greedy is faster.",
    )
    if not _hf_credentials_ready():
        if ON_CLOUD:
            st.warning("Add **HF_TOKEN** to Streamlit Secrets on Cloud.")
        else:
            st.warning(
                "No Hugging Face token was found. Set `export HF_TOKEN='hf_...'` "
                "in the same terminal and restart Streamlit, run "
                "`huggingface-cli login`, or add `HF_TOKEN` to "
                "`.streamlit/secrets.toml`."
            )
    _startup_warmup(include_visual=not audio_only)

def _details(rows: list[tuple[str, str]]) -> None:
    """Render compact, escaped metadata underneath each panel."""
    entries = "".join(
        '<div class="viavsr-detail">'
        f"<dt>{html.escape(label)}</dt><dd>{html.escape(value)}</dd></div>"
        for label, value in rows
    )
    st.markdown(
        f'<dl class="viavsr-details">{entries}</dl>', unsafe_allow_html=True
    )


def _number(value: Any, unit: str = "", digits: int = 1) -> str:
    return "Not available" if value is None else f"{float(value):.{digits}f}{unit}"


def _placeholder(message: str) -> None:
    st.markdown(
        f'<div class="viavsr-empty">{html.escape(message)}</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _source_metadata(path: str, modified_ns: int, size: int) -> dict[str, Any]:
    """Cache metadata per file version without loading recognition models."""
    from viavsr.preprocessing import probe_av_media

    return probe_av_media(path).to_dict()


with st.sidebar:
    st.divider()
    st.markdown("### Input")
    source = st.radio("Input source", ("Upload", "Webcam", "Sample"), horizontal=True)
    media_path: Path | None = None
    if source == "Upload":
        uploaded = st.file_uploader(
            "Video with audio", type=["mp4", "webm", "mov"]
        )
        if uploaded is not None:
            media_path = _save_upload(uploaded.name, uploaded.getvalue())
    elif source == "Webcam":
        recorded = RECORDER(key="webcam")
        if recorded:
            try:
                media_path = _data_url_to_webm(recorded)
            except Exception as exc:
                st.error(f"Could not read the webcam recording: {exc}")
    else:
        samples = sorted(SAMPLE_DIR.glob("*.mp4"))
        if samples:
            media_path = st.selectbox(
                "Vietnamese sample", samples, format_func=lambda p: p.name
            )
        else:
            st.info("No samples are available. Use Upload or Webcam.")
    reference = st.text_area(
        "Reference transcript (optional)",
        help="Enter the exact Vietnamese words in the processed clip to calculate "
        "WER and CER. Include Vietnamese diacritics.",
    )
    st.caption("The reference must match the selected clip duration.")

source_metadata: dict[str, Any] = {}
request_key = None
if media_path and media_path.is_file():
    file_stat = media_path.stat()
    request_key = (
        str(media_path.resolve()), file_stat.st_mtime_ns, file_stat.st_size,
        audio_only, visual_fallback_policy, fast_mode, clip_seconds, decoder,
        reference.strip(),
    )
    try:
        source_metadata = _source_metadata(
            str(media_path), file_stat.st_mtime_ns, file_stat.st_size
        )
    except Exception as exc:
        st.warning(f"Could not inspect the input media: {exc}")

# Reserve the three cards before the action area; fill them with this run's data.
input_col, mouth_col, result_col = st.columns([1, 1, 1], gap="medium")
input_panel = input_col.container(border=True)
mouth_panel = mouth_col.container(border=True)
result_panel = result_col.container(border=True)

run = st.button(
    "Run recognition",
    type="primary",
    disabled=request_key is None,
)
if run and request_key is not None:
    # Invalidate the old result before any work, including failures during upload prep.
    st.session_state.pop("result", None)
    st.session_state.pop("result_request_key", None)
    try:
        with st.status("Processing...", expanded=True) as progress:
            progress.write("Preparing tokenizer...")
            _ensure_tokenizer_assets()
            progress.write("Preprocessing media...")
            prepared = _prepare_media(
                media_path, clip_seconds, max_width=max_width,
                audio_only=audio_only, fast=fast_mode,
            )
            progress.write(
                "Recognizing..." if audio_only else "Tracking face and recognizing..."
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
                reference_text=reference.strip() or None,
                max_duration_seconds=float(clip_seconds + DURATION_SLACK),
                max_detection_size=max_detection,
                detection_stride=2 if fast_mode else 1,
                visual_fallback_policy=visual_fallback_policy,
            )
            passed = result.get("status") == "passed"
            progress.update(
                label="Completed" if passed else "Failed",
                state="complete" if passed else "error",
            )
        st.session_state.result = result
        st.session_state.result_request_key = request_key
        gc.collect()
    except Exception as exc:
        st.error(f"Recognition failed: {exc}")

# A changed file, reference or option must not show a previous recording's metrics.
result = (
    st.session_state.get("result")
    if request_key is not None
    and st.session_state.get("result_request_key") == request_key
    else None
)
report = result or {}
inner = report.get("result") or {}
tracking = report.get("face_tracking") or {}
display = report.get("mouth_roi_display") or {}
availability = display.get("visual_availability") or tracking.get("visual_availability") or {}
timings = report.get("timings_seconds") or {}
mode = (report.get("modality_decision") or {}).get("selected_mode")
mode_labels = {
    "audio_visual": "Audio + Video",
    "audio_visual_corrupted": "Corrupted AV",
    "audio_visual_interval_gated": "Interval-gated AV",
    "audio_only_fallback": "Audio-only fallback",
    "audio_only_experimental": "Audio only",
}

with input_panel:
    st.markdown("#### 1. Original Input")
    if media_path and media_path.is_file():
        st.video(str(media_path))
        _details([
            ("File", media_path.name),
            ("Duration", _number(source_metadata.get("duration_seconds"), " s", 2)),
            ("Audio sample rate", _number(source_metadata.get("audio_sample_rate"), " Hz", 0)),
            ("Video frame rate", _number(source_metadata.get("frame_rate"), " FPS", 2)),
            ("Resolution", f"{source_metadata['video_width']} x {source_metadata['video_height']}"
             if source_metadata else "Not available"),
            ("Source", source),
            ("Clip limit", f"First {clip_seconds:.0f} s"),
        ])
        duration = source_metadata.get("duration_seconds")
        if duration and duration > clip_seconds:
            st.caption(f"Recognition uses only the first {clip_seconds:.0f} seconds.")
    else:
        _placeholder("Upload, record or select a video in the sidebar.")

with mouth_panel:
    st.markdown("#### 2. Processed Mouth ROI")
    mouth_path = (report.get("artifacts") or {}).get("mouth_roi")
    if mouth_path and Path(mouth_path).is_file():
        st.video(mouth_path)
        st.caption("Missing visual intervals are marked as NO VISUAL SIGNAL.")
    else:
        _placeholder(
            "Mouth ROI is unavailable for this run."
            if result else "The processed mouth video will appear after recognition."
        )
    if result:
        count = availability.get("frame_count")
        valid = availability.get("valid_frames")
        coverage = availability.get("coverage")
        valid_label = (
            f"{valid} / {count} ({coverage:.1%})"
            if count is not None and valid is not None and coverage is not None
            else "Not available"
        )
        output_media = display.get("output_media") or {}
        _details([
            ("Mouth crop", f"{display['mouth_roi_size']} x {display['mouth_roi_size']}"
             if display.get("mouth_roi_size") else "Not available"),
            ("Frames with visual signal", valid_label),
            ("Missing frames", str(availability.get("missing_frames", "Not available"))),
            ("Tracking quality", str(tracking.get("quality_status", "Not available")).capitalize()),
            ("ROI frame rate", _number(output_media.get("frame_rate"), " FPS", 1)),
            ("Face tracking", _number(timings.get("face_tracking"), " s", 2)),
            ("ROI export", _number(timings.get("mouth_roi"), " s", 2)),
        ])
        st.caption("Model input: grayscale 96 x 96 ROI, center crop to 88 x 88, normalization.")
        if mode and mode.startswith("audio_only"):
            st.caption("This run uses audio only; any ROI shown is for inspection.")
        intervals = availability.get("missing_intervals") or []
        if intervals:
            with st.expander("Missing visual intervals"):
                for interval in intervals:
                    st.write(
                        f"{interval['start_seconds']:.2f} - "
                        f"{interval['end_seconds']:.2f} s "
                        f"({interval['frame_count']} frames)"
                    )

with result_panel:
    st.markdown("#### 3. Recognition Result")
    if not result:
        _placeholder("Run recognition to see the transcript and scores.")
        if st.session_state.get("result"):
            st.caption("Input or settings changed. Run again to update the results.")
    else:
        if report.get("status") == "passed":
            st.success("Completed")
        else:
            st.error((report.get("error") or {}).get("message", "Recognition failed."))
        if inner:
            st.caption("Transcript")
            transcript = inner.get("transcript") or "(empty)"
            st.markdown(
                '<div class="viavsr-transcript-wrap">'
                f'<p class="viavsr-transcript">{html.escape(transcript)}</p></div>',
                unsafe_allow_html=True,
            )
        st.markdown("**Runtime & decoding**")
        preprocessing_values = [
            value for key, value in timings.items()
            if key in {
                "media_preflight", "face_tracking_backend", "face_tracking",
                "mouth_roi_display", "mouth_roi", "av_preprocessing", "audio_preprocessing",
            }
        ]
        _details([
            ("Mode used", mode_labels.get(mode, mode or "Not available")),
            ("Decoder", {"joint_beam_search": "Joint CTC/Attention",
                         "ctc_greedy": "CTC greedy"}.get(inner.get("decoder"), "Not available")),
            ("Preprocessing", _number(sum(preprocessing_values) if preprocessing_values else None, " s", 2)),
            ("Model loading", _number(timings.get("model_loading"), " s", 2)),
            ("Inference", _number(timings.get("inference"), " s", 2)),
            ("Pipeline total", _number(timings.get("total"), " s", 2)),
            ("Device", inner.get("device", "Not available")),
        ])
        st.caption("Pipeline time excludes upload preparation and initial model warm-up.")
        evaluation = report.get("evaluation")
        if evaluation:
            wer_col, cer_col = st.columns(2)
            wer_col.metric("WER", f"{evaluation['wer']:.2%}")
            cer_col.metric("CER", f"{evaluation['cer']:.2%}")
            st.caption("Compared with the supplied reference transcript.")
        elif inner:
            st.caption("Add a reference transcript before running to calculate WER and CER.")
        if inner.get("hypothesis_score") is not None:
            with st.expander("Decoder details"):
                _details([
                    ("Beam size", str(inner.get("beam_size", "Not available"))),
                    ("CTC weight", _number(inner.get("ctc_weight"), digits=2)),
                    ("Hypothesis score", _number(inner["hypothesis_score"], digits=3)),
                ])
                st.caption("The decoder score is not a calibrated confidence probability.")
        warnings = report.get("warnings") or []
        if warnings:
            with st.expander("Run notes"):
                for warning in warnings:
                    st.write(str(warning).replace("_", " "))
        st.download_button(
            "Export result (JSON)",
            data=json.dumps(report, ensure_ascii=False, indent=2),
            file_name=f"{media_path.stem}_result.json",
            mime="application/json",
        )
        with st.expander("Full JSON report"):
            st.json(report)
