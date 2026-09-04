#!/usr/bin/env python3
"""Compare corrupted, interval-gated, and audio-only webcam inference."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from viavsr.evaluation.webcam_comparison import run_paired_webcam_comparison
from viavsr.inference import DEFAULT_BEAM_SIZE, DEFAULT_CTC_WEIGHT
from viavsr.preprocessing.face_tracking import DEFAULT_DETECTION_MAX_SIZE
from viavsr.preprocessing.media import TARGET_FRAME_RATE

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "experiments" / "paired_webcam"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Track and preprocess one webcam recording once, then compare "
            "corrupted AV, interval-gated AV, and audio-only inference."
        )
    )
    parser.add_argument("--media", required=True, type=Path)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--tracking-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--decoder",
        choices=("ctc_greedy", "joint_beam_search"),
        default="joint_beam_search",
    )
    parser.add_argument("--beam-size", type=int, default=DEFAULT_BEAM_SIZE)
    parser.add_argument("--ctc-weight", type=float, default=DEFAULT_CTC_WEIGHT)
    parser.add_argument("--max-duration", type=float, default=15.0)
    parser.add_argument("--frame-rate", type=int, default=TARGET_FRAME_RATE)
    parser.add_argument(
        "--max-detection-size",
        type=int,
        default=DEFAULT_DETECTION_MAX_SIZE,
    )
    parser.add_argument("--confidence-threshold", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_paired_webcam_comparison(
        config_path=args.config,
        media_path=args.media,
        output_root=args.output_root,
        reference_text=args.reference_text,
        tracking_device=args.tracking_device,
        decoder=args.decoder,
        beam_size=args.beam_size,
        ctc_weight=args.ctc_weight,
        max_duration_seconds=args.max_duration,
        frame_rate=args.frame_rate,
        max_detection_size=args.max_detection_size,
        confidence_threshold=args.confidence_threshold,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
