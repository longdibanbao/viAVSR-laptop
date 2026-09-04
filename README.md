# viAVSR-laptop

Vietnamese audio-visual speech recognition for short laptop webcam recordings.

The project combines synchronized speech audio and a tracked mouth region, encodes both modalities with a released Vietnamese AV-HuBERT model, and decodes Vietnamese text with either CTC greedy decoding or joint CTC/attention beam search.

## Overview

```text
webcam video with embedded audio
    → media validation
    → face detection and 68-point landmark tracking
    → affine face alignment
    → 96×96 mouth-region extraction
    → synchronized audio/video feature preparation
    → AV-HuBERT encoding
    → CTC greedy or joint CTC/attention decoding
    → Vietnamese transcript
    → optional WER/CER evaluation
```

The implementation targets Linux. Windows users should run it inside WSL2 Ubuntu so FFmpeg, Conda, CUDA, and POSIX paths behave consistently.

## Current validated status

The backend is ready for glass-box UI integration. The remaining product work is
primarily presentation, recording controls, asynchronous execution, and
report visualization, not a second implementation of AVSR preprocessing or
inference.

| Capability | Current status |
| --- | --- |
| Vietnamese checkpoint and tokenizer | Pinned, checksum-validated, vocabulary-compatible |
| Raw webcam preprocessing | Face tracking, quality gates, alignment, 96x96 mouth ROI, synchronized audio |
| Missing visual intervals | Display-ready `No visual signal` frames plus experimental interval-gated inference |
| Unusable visual stream | Whole-utterance audio-only fallback still produces a transcript |
| Decoding and evaluation | CTC greedy, joint CTC/attention, Vietnamese WER/CER |
| End-to-end integration | One library call and one CLI command, with schema-versioned JSON |
| Generated artifacts | Final report plus optional mouth-ROI display video; transient work is removed |
| Automated validation | 138 tests passing |
| Local webcam evidence | 10/10 recordings completed; corpus WER 12.73%, CER 7.31% |
| ViCocktail robustness smoke | 5/5 samples and 145/145 condition records passed |

These results establish integration readiness, not production-level accuracy.
The five-sample robustness run is a smoke test; use the resumable full-split
command below for final experimental conclusions.

## Features

- Vietnamese AV-HuBERT checkpoint loading from Hugging Face.
- Verified Vietnamese SentencePiece `unigram2048` tokenizer.
- Configurable CPU or CUDA model placement.
- Model/tokenizer vocabulary compatibility validation.
- Raw webcam media inspection.
- RetinaFace detection and FAN 68-point facial landmark tracking.
- Temporal landmark interpolation and smoothing.
- Affine face alignment and 96×96 mouth-ROI export.
- Synchronized 25 fps video and 16 kHz mono audio preprocessing.
- CTC greedy and joint CTC/attention beam-search decoding.
- Vietnamese-aware WER and CER that preserve diacritics.
- JSON reports for model assets, preprocessing, inference, and evaluation.

## Model and tokenizer

| Asset | Pinned source |
| --- | --- |
| Model | [`nguyenvulebinh/AV-HuBERT-CTC-Attention-VI`](https://huggingface.co/nguyenvulebinh/AV-HuBERT-CTC-Attention-VI) |
| Model revision | `b8a1fa5d6079701b3f8f791bfd601057fbd23de3` |
| Dataset | [`nguyenvulebinh/ViCocktail`](https://huggingface.co/datasets/nguyenvulebinh/ViCocktail) |
| Tokenizer repository | `nguyenvulebinh/viCocktail` |
| Tokenizer revision | `ad644a77e8e3177aa7422510302c11de5282fa26` |
| SentencePiece model | `unigram2048.model` |
| Token units | `unigram2048_units.txt` |

Expected tokenizer checksums:

| File | SHA-256 |
| --- | --- |
| `unigram2048.model` | `21ca39e799b64044d75edccd9016fac0315e64f89bdd43fbd3089607dceb9d64` |
| `unigram2048_units.txt` | `ea7b25e67a302305ffdb59909419c08822b3607a6b03871adef2bcb9f6ebec25` |

Tokenizer binaries, the checkpoint, and Hugging Face caches are downloaded locally and are not committed.

## Requirements

- Linux or WSL2 Ubuntu.
- Conda or Miniconda.
- Python 3.11.
- FFmpeg and FFprobe.
- A Hugging Face account with access to the released checkpoint.
- An NVIDIA GPU is recommended for full inference.

The pipeline has been validated on a GeForce GTX 1660 Ti with 6 GB of VRAM. CPU placement is supported but considerably slower.

## Installation

```bash
git clone <repository-url>
cd viAVSR-laptop

conda env create -f environment/environment.yml
conda activate viavsr
pip install -e ".[dev]"
```

For an existing environment:

```bash
conda env update -f environment/environment.yml --prune
conda activate viavsr
pip install -e ".[dev]"
```

Confirm the package and CUDA runtime:

```bash
python -c 'import torch, viavsr; print("viavsr:", viavsr.__file__); print("torch:", torch.__version__); print("CUDA:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")'
```

## Configuration

The default runtime configuration is [`configs/config.yaml`](configs/config.yaml).

```yaml
model:
  repository_id: nguyenvulebinh/AV-HuBERT-CTC-Attention-VI
  revision: b8a1fa5d6079701b3f8f791bfd601057fbd23de3
  cache_dir: .cache/models
  device: cuda
  dtype: float32

tokenizer:
  model_path: assets/tokenizers/vi/unigram2048.model
  units_path: assets/tokenizers/vi/unigram2048_units.txt
```

Set `model.device` to `cpu` for an intentional CPU run. The loader does not silently fall back from CUDA to CPU. Do not put credentials or machine-specific absolute paths in YAML.

## Hugging Face authentication

Accept the model repository access conditions, then export a personal read-only token without writing it into project files:

```bash
read -rsp "Hugging Face token: " HF_TOKEN
export HF_TOKEN
echo
```

Confirm that the variable exists without printing its value:

```bash
python -c 'import os; print("HF_TOKEN configured:", bool(os.environ.get("HF_TOKEN")))'
```

Each team member should use their own token. Never commit tokens, `.env` files, or commands containing a real credential.

## Download and validate model assets

Download and checksum the Vietnamese tokenizer:

```bash
python scripts/fetch_tokenizer_assets.py \
  --config configs/config.yaml
```

Load the checkpoint, validate model/tokenizer vocabulary dimensions, and run Vietnamese tokenizer round trips:

```bash
python scripts/check_model_assets.py \
  --config configs/config.yaml
```

Expected successful output includes:

```text
Model loaded: PASSED
Tokenizer loaded: PASSED
SentencePiece vocabulary: 2048
ASR tokenizer vocabulary: 2057
Model output vocabulary: 2057
Vocabulary compatibility: PASSED
Vietnamese round-trip: PASSED
Device: cuda
```

Metadata and logs are generated under `outputs/model_assets/`.

## Recording input

Record one Vietnamese utterance per file.

- Prefer 3–5 seconds; the default maximum is 15 seconds.
- Keep the speaker frontal and well lit.
- Keep the full face visible, including chin and forehead.
- Avoid mouth occlusion, rapid head turns, and strong backlighting.
- Keep the mouth sufficiently large in the frame.
- Store audio and video in the same container.
- Prefer MP4 or WebM.
- Save the exact spoken sentence if WER/CER will be calculated.

FFmpeg normalizes source frame rate, sample rate, and channel count later.

Local recordings belong under `samples/webcam/` or `recordings/` and are ignored:

```bash
mkdir -p samples/webcam
```

Suggested local naming:

```text
samples/webcam/webcam_001.mp4
samples/webcam/webcam_001_reference.txt
```

## End-to-end webcam inference

### One-command raw-video workflow

For the normal demo path, give the runner a raw webcam video that contains both
video and audio:

~~~bash
python scripts/run_avsr_demo.py \
  --config configs/config.yaml \
  --media samples/webcam/webcam_001.mp4 \
  --tracking-device auto \
  --decoder joint_beam_search \
  --beam-size 3 \
  --ctc-weight 0.1
~~~

Add `--reference-text "the exact sentence spoken in the video"` to calculate
WER/CER. The model device and dtype come from `configs/config.yaml`; the
tracking device is selected independently with `--tracking-device`.

By default, each run retains only the final UI mouth-ROI video (when available)
and consolidated JSON report:

~~~text
outputs/demo/webcam_001/
  mouth_roi.mp4
  report.json
~~~

The report contains raw-media metadata, face-quality diagnostics, preprocessed
tensor shapes, model/tokenizer metadata, modality decisions, decoder parameters,
transcript, timings, and optional WER/CER. If visual preprocessing is unusable,
the default `whole_utterance` policy transcribes from audio only.

Two experimental partial-visual policies are available for recordings where
the mouth is unavailable only during parts of the utterance. To reproduce the
benchmark's corrupted-AV condition on a webcam recording:

~~~bash
python scripts/run_avsr_demo.py \
  --config configs/config.yaml \
  --media samples/webcam/webcam_001.mp4 \
  --visual-fallback-policy corrupted_av
~~~

`corrupted_av` neutral-fills unavailable normalized mouth frames while keeping
all aligned audio, then runs the checkpoint's ordinary audio-visual path. It
does not suppress visual features explicitly.

To apply interval gating instead:

~~~bash
python scripts/run_avsr_demo.py \
  --config configs/config.yaml \
  --media samples/webcam/webcam_001.mp4 \
  --visual-fallback-policy interval_gated
~~~

`interval_gated` uses the same neutral-filled input but also zeros unavailable
visual features before audio/visual fusion. Both modes preserve usable visual
intervals and expose the same `No visual signal` intervals in the UI artifact.
Compare their WER/CER on the same recording and reference text.

For a strict paired comparison, use one command to share the face track,
prepared tensors, availability mask, and loaded model across corrupted AV,
interval-gated AV, and audio-only inference:

~~~bash
python scripts/compare_webcam_modes.py \
  --config configs/config.yaml \
  --media samples/webcam/webcam_008.mp4 \
  --tracking-device cuda \
  --decoder joint_beam_search \
  --beam-size 3 \
  --ctc-weight 0.1 \
  --reference-text "ơ sau một ngày mệt nhoài thì ờ cuối cùng tôi cũng đã về lại ký túc xá"
~~~

The comparison report includes tensor and availability-mask fingerprints,
per-mode transcripts and WER/CER, paired metric deltas, model metadata, and
timings. Only two compact artifacts are retained:

~~~text
outputs/experiments/paired_webcam/webcam_008/
  comparison_report.json
  mouth_roi.mp4
~~~

For the normal demo runner, add `--keep-intermediates` when debugging.
Transient face tracks, stage reports,
and the inference-only ROI are then retained under `outputs/demo/<stem>/.work/`.
Without that flag they are deleted after the consolidated report is written.

## Glass-box UI handoff

The UI should integrate with the reusable Python function, not reproduce shell
commands or preprocessing logic:

```python
from pathlib import Path

from viavsr.demo import run_end_to_end_demo

payload = run_end_to_end_demo(
    config_path=Path("configs/config.yaml"),
    media_path=Path("recordings/current.mp4"),
    output_root=Path("outputs/demo"),
    tracking_device="auto",
    decoder="joint_beam_search",
    beam_size=3,
    ctc_weight=0.1,
    visual_fallback_policy="whole_utterance",
)
```

This call is blocking, so run it in a worker process or background thread and
disable duplicate Record/Infer actions until it finishes. It returns the same
dictionary written to `outputs/demo/<media-stem>/report.json`. The current
report schema is version 3; reject unknown future major versions rather than
silently interpreting changed fields.

### UI data contract

| UI area | Source |
| --- | --- |
| Original recording | `request.media_path` and `raw_media` |
| Processed mouth video | `artifacts.mouth_roi`, when present |
| Visual availability | `face_tracking.visual_availability` |
| Missing intervals | `face_tracking.visual_availability.missing_intervals` |
| Tracking quality | `face_tracking.quality_status`, `detection_rate`, and `visual_coverage` |
| Selected modality | `modality_decision.selected_mode` and `visual_input_used` |
| Transcript | `result.transcript` |
| Decoder settings | `result.decoder`, `beam_size`, and `ctc_weight` |
| Runtime | `timings_seconds`, especially `total` and `inference` |
| Optional accuracy | `evaluation.wer` and `evaluation.cer` |
| Warning state | `warnings` and `modality_decision.fallback_reason` |
| Failure state | `status`, `stage`, and `error.message` |

The exported mouth video is specifically for UI playback. Its missing intervals
already contain `No visual signal` frames synchronized to the original
timeline. If `artifacts.mouth_roi` is absent, show a persistent no-visual
placeholder while still displaying an audio-only transcript when the run
succeeds.

Do not read `.work/`, face-track NPZ files, or console text from the UI. Those
are transient debugging details. Do not invent word confidence, token
timestamps, or top-N hypotheses: the current backend exposes only the best
transcript and a sequence-level hypothesis score. Such features require a
separate backend contract before they can be shown truthfully.

### Recommended first UI milestone

1. Record or select one MP4/WebM file containing camera video and microphone
   audio, limited to 15 seconds.
2. Show the original recording and its duration/audio/video metadata.
3. Run `run_end_to_end_demo` asynchronously and present a clear busy state.
4. Show the processed mouth video, visual coverage, missing intervals, and
   selected modality.
5. Show the best transcript, decoder settings, total runtime, and optional
   WER/CER when reference text was entered.
6. Handle `passed`, audio-only fallback, warning, and failed states without
   crashing or leaving stale output from a previous run.
7. Allow exporting the final `report.json`.

Use `whole_utterance` as the default fallback policy. The UI may expose
`corrupted_av` and `interval_gated` behind an **Experimental** control; the
released checkpoint was not trained specifically for these real-webcam
missing-interval policies. Keep the record/infer flow single-utterance and
single-speaker for this milestone. Multi-speaker selection,
live streaming, top-N hypotheses, word-level confidence, and model fine-tuning
remain separate backend work.

### Manual stage-by-stage workflow

#### 1. Inspect the raw recording

```bash
python scripts/check_media_input.py \
  --media samples/webcam/webcam_001.mp4 \
  --output outputs/inference/webcam_001_preflight.json
```

Typical `next_stage` values:

- `face_alignment_and_mouth_roi_extraction_required` for ordinary webcam video.
- `ready_for_inference` for a prepared 96×96 mouth video.
- `split_or_record_a_shorter_clip` when the file exceeds the duration limit.

A raw full-frame video normally reports `has_mouth_roi_resolution: false`. This describes the current file and is not a tracking failure.

#### 2. Track faces and landmarks

One recording:

```bash
python scripts/track_webcam_faces.py \
  --device auto \
  --media samples/webcam/webcam_001.mp4
```

Several recordings:

```bash
python scripts/track_webcam_faces.py \
  --device auto \
  --media \
    samples/webcam/webcam_001.mp4 \
    samples/webcam/webcam_002.mp4 \
    samples/webcam/webcam_003.mp4 \
    samples/webcam/webcam_004.mp4
```

Generated artifacts:

```text
outputs/preprocessing/face_tracks/<stem>_face_track.npz
outputs/preprocessing/face_tracks/<stem>_face_track.json
```

The NPZ contains landmarks, face boxes, confidence scores, detection masks, and
a versioned VIAVSR-7 quality result. Load it with
`numpy.load(path, allow_pickle=False)`. Quality thresholds live under
`face_tracking` in `configs/config.yaml`; `--config` selects another policy
and `--confidence-threshold` overrides only its detection-confidence threshold.
Low-confidence, low-landmark-confidence, too-small, mostly out-of-frame, and
geometrically inconsistent face candidates are excluded. This is temporal
association, not biometric face recognition. The command exits non-zero when the resulting track
fails clip-level detection-rate, internal/edge missing-run, minimum-frame, or
ambiguity thresholds. Inspect `quality_issues` and `quality_thresholds` in the JSON
before retrying.

#### 3. Align faces and extract mouth regions

```bash
python scripts/prepare_webcam_mouth_roi.py \
  --media \
    samples/webcam/webcam_001.mp4 \
    samples/webcam/webcam_002.mp4 \
    samples/webcam/webcam_003.mp4 \
    samples/webcam/webcam_004.mp4
```

Generated artifacts:

```text
outputs/preprocessing/mouth_roi/<stem>_mouth96.mp4
outputs/preprocessing/mouth_roi/<stem>_mouth96.json
```

Each MP4 contains a 96×96 aligned grayscale mouth track at 25 fps and synchronized 16 kHz mono audio.

#### 4. Run model inference

Joint CTC/attention decoding:

```bash
python scripts/run_media_inference.py \
  --config configs/config.yaml \
  --media outputs/preprocessing/mouth_roi/webcam_001_mouth96.mp4 \
  --decoder joint_beam_search \
  --beam-size 3 \
  --ctc-weight 0.1 \
  --reference-text "hôm nay thời tiết bên ngoài rất đẹp" \
  --output outputs/inference/webcam_001_joint_beam.json
```

Omit `--reference-text` when no reference is available. The transcript and token IDs are still produced, but WER/CER are omitted.

CTC greedy decoding:

```bash
python scripts/run_media_inference.py \
  --config configs/config.yaml \
  --media outputs/preprocessing/mouth_roi/webcam_001_mouth96.mp4 \
  --decoder ctc_greedy \
  --output outputs/inference/webcam_001_greedy.json
```

Released joint-decoder settings:

- Beam size: 3.
- Attention weight: 0.9.
- CTC-prefix weight: 0.1.
- Language-model weight: 0.
- Length-bonus weight: 0.

The inference report includes media metadata, tensor shapes, token IDs, transcript, decoder parameters, device, dtype, timing, and optional evaluation.

#### 5. Evaluate an existing transcript

```bash
python scripts/evaluate_transcripts.py \
  --reference-text "hôm nay trời đẹp" \
  --prediction-text "hôm nay trời lạnh" \
  --output outputs/evaluation/metrics.json
```

Normalization applies Unicode NFC, lowercase, a consistent punctuation policy, and whitespace collapse while preserving Vietnamese diacritics. Raw model predictions are evaluated without LLM rewriting.

## Preprocessing contract

| Stage | Contract |
| --- | --- |
| Prepared video | 25 fps, grayscale, 96×96 mouth frames |
| Video transform | divide by 255, center-crop to 88×88, normalize with mean 0.421 and standard deviation 0.165 |
| Video tensor | `[1, 1, T, 88, 88]` |
| Prepared audio | mono, 16 kHz |
| Synchronization | trim or pad to 640 audio samples per video frame |
| Audio transform | 26-bin log filterbanks, stack four frames, layer normalization |
| Audio tensor | `[1, 104, T]` |

Inference currently processes one utterance per call.

## Webcam evaluation results

Ten local webcam recordings were processed from raw full-frame video through
tracking, mouth extraction, joint CTC/attention inference, and WER/CER
evaluation. The interval-gated policy was enabled and activated only when the
visual stream contained missing intervals.

### Latest joint-decoder run

| Sample | Duration | Inference mode | Visual coverage | WER | CER |
| --- | ---: | --- | ---: | ---: | ---: |
| `webcam_001` | 4.45 s | Audio + visual | 100.00% | 0.00% | 0.00% |
| `webcam_002` | 4.10 s | Audio + visual | 100.00% | 0.00% | 0.00% |
| `webcam_003` | 5.10 s | Audio + visual | 100.00% | 20.00% | 10.53% |
| `webcam_004` | 7.28 s | Audio + visual | 100.00% | 25.00% | 12.24% |
| `webcam_005` | 9.47 s | Interval-gated AV | 97.03% | 32.35% | 20.00% |
| `webcam_006` | 7.59 s | Interval-gated AV | 40.96% | 0.00% | 0.00% |
| `webcam_007` | 8.64 s | Interval-gated AV | 21.50% | 4.76% | 2.38% |
| `webcam_008` | 8.58 s | Interval-gated AV | 67.45% | 22.22% | 11.59% |
| `webcam_009` | 9.19 s | Interval-gated AV | 72.81% | 0.00% | 0.00% |
| `webcam_010` | 8.85 s | Interval-gated AV | 96.35% | 6.67% | 4.92% |

Across all ten recordings, the corpus result was 21 word errors over 165
reference words (12.73% WER) and 49 character errors over 670 reference
characters (7.31% CER). Full transcripts and diagnostics are stored in each
sample's consolidated `report.json`.


A separate ten-sample compatibility run on the official ViCocktail clean test split produced:

| Metric | Result |
| --- | ---: |
| Successful samples | 10 / 10 |
| Evaluated duration | 69.50 s |
| Corpus WER | 5.71% |
| Corpus CER | 3.20% |
| Inference real-time factor | 0.994 |

This deterministic prefix is also a smoke check rather than a full benchmark. Reproduce it with:

```bash
python scripts/run_official_benchmark.py \
  --config configs/config.yaml \
  --split test \
  --offset 0 \
  --count 10 \
  --beam-size 3 \
  --ctc-weight 0.1
```

By default the benchmark retains only:

```text
outputs/official_benchmark/
  benchmark_report.json
  execution.log
```

Downloaded media and duplicate per-sample reports are transient and are removed
after their contents have been consolidated into the benchmark report. Add
`--keep-intermediates` only when debugging; those files are then retained under
`outputs/official_benchmark/.work/`. All generated benchmark artifacts are
ignored by Git.

## Robustness evaluation

### Visual-dropout robustness benchmark

The robustness runner compares the same ViCocktail samples under:

- clean audio and video;
- complete audio with deterministic contiguous visual dropout at 10%, 30%, and
  50%;
- the same corrupted input without interval gating;
- interval-gated audio-visual inference;
- whole-utterance audio-only inference; and
- automatic routing between interval-gated and audio-only inference.

Three seeds are used by default. Each seed produces the same mask for all paired
conditions, making their WER/CER differences directly comparable. Run a
five-sample local smoke check with:

```bash
python scripts/run_visual_dropout_benchmark.py \
  --config configs/config.yaml \
  --split test \
  --offset 0 \
  --count 5 \
  --beam-size 3 \
  --ctc-weight 0.1 \
  --output-dir outputs/visual_dropout_smoke_5
```

Set `--count 0` for the complete remaining split. The defaults are equivalent
to `--dropout-levels 0.1 0.3 0.5 --seeds 17 29 43`. Work is recorded after
every condition, so rerunning the exact command safely skips completed records.
If the dataset revision, model, tokenizer, decoder, dropout protocol, or routing
threshold changes, use a new output directory.

The verified five-sample smoke run completed all 145 condition records without
failure. Corpus results were:

| Condition | Visual dropout | WER | CER |
| --- | ---: | ---: | ---: |
| Clean AV | 0% | 5.58% | 2.72% |
| Audio-only | Full visual removal | 4.06% | 1.85% |
| Corrupted AV | 10% | 5.75% | 2.63% |
| Interval-gated AV | 10% | 5.41% | 2.43% |
| Automatic routing | 10% | 5.41% | 2.43% |
| Corrupted AV | 30% | 6.09% | 2.96% |
| Interval-gated AV | 30% | 5.75% | 2.76% |
| Automatic routing | 30% | 5.75% | 2.76% |
| Corrupted AV | 50% | 6.43% | 2.92% |
| Interval-gated AV | 50% | 5.58% | 2.59% |
| Automatic routing | 50% | 4.06% | 1.85% |

Automatic routing selected interval-gated AV for 10% and 30% dropout and
audio-only at 50% dropout. This small sample is evidence that the benchmark and
routing mechanics work; it is not sufficient to claim general accuracy gains.

Only three final artifacts are retained:

```text
outputs/visual_dropout_benchmark/
  benchmark_report.json
  execution.log
  results.jsonl
```

The JSON report contains corpus and macro WER/CER by condition and dropout
level, paired deltas from clean AV, routing counts, decoder settings, and
model/tokenizer metadata. The JSONL file is the append-only resume ledger.
Downloaded media remains transient under `.work/` and is removed after each
invocation.

## Project structure

```text
assets/                   tokenizer manifest and downloaded local assets
configs/                  model and runtime configuration
environment/              Conda environment definition
samples/                  local media or lightweight sample metadata
scripts/                  command-line entry points
src/viavsr/
  evaluation/             Vietnamese normalization and WER/CER
  inference/              model loading, decoding, and recognition
  preprocessing/          media probing, face tracking, and mouth extraction
  utils/                  small shared utilities
tests/                    automated tests
outputs/                  generated reports, media, and predictions
```

Reusable implementation belongs under `src/viavsr/`. Scripts should primarily parse arguments and orchestrate library functions.

## Testing

```bash
python -m pytest
```

Current validated result:

```text
138 passed
```

When Ruff is installed:

```bash
ruff check scripts src tests
ruff format --check scripts src tests
```

The vendored AVSRCocktail implementation under `src/viavsr/inference/vendor/` is excluded from Ruff to preserve upstream behavior.

## Generated files and Git policy

Keep these files local:

- Raw or processed audio/video.
- Downloaded datasets.
- Checkpoints and Hugging Face caches.
- Tokenizer binaries.
- Face-track NPZ files.
- Generated reports, predictions, and logs.
- Conda or virtual environments.
- Experiment-tracking directories.
- UI dependency folders, generated bundles, and browser-test reports.
- Local UI recordings and framework secret files.
- Credentials and `.env` files.

Commit UI source code, reusable components, tests, lock files, and example
configuration alongside the existing Python source, YAML configuration,
tokenizer manifests, and lightweight sample metadata.

Before committing:

```bash
git status --short
git diff --cached
```

Do not force-add ignored media or model artifacts.

## Remote GPU usage

The same workflow applies to a lab server or rented Linux GPU:

1. Clone the repository on the remote Linux filesystem.
2. Create the `viavsr` Conda environment.
3. Export a personal `HF_TOKEN`.
4. Keep `model.device: cuda`.
5. Run tokenizer and model-asset validation.
6. Copy only required input media to the server.
7. Run preprocessing and inference.
8. Copy back small JSON reports and transcripts.

Use persistent storage for `.cache/models/` when available so the checkpoint is not downloaded for every session.

## Troubleshooting

### `has_mouth_roi_resolution` is false

This is expected for ordinary webcam video. Run tracking and mouth extraction before inference.

### CUDA was requested but is unavailable

Check `nvidia-smi`, the CUDA-enabled PyTorch installation, and WSL GPU support. Set `model.device: cpu` only for an intentional CPU run.

### Model access fails

Accept the model terms and verify `HF_TOKEN`. Never put the token in `configs/config.yaml`.

### The prepared mouth video does not exist

Run `track_webcam_faces.py`, then `prepare_webcam_mouth_roi.py`. Inference consumes `<stem>_mouth96.mp4`, not the raw recording.

### The shell reports unrecognized arguments or command not found

A line-continuation backslash must be the final character on its line. Do not put spaces after it.

### Face detection is unstable

Improve frontal pose, lighting, face size, and mouth visibility. Inspect the tracking JSON for detection rate, interpolated frames, confidence, and maximum missing runs.
Mouth-ROI extraction intentionally rejects a failed track, unsupported artifact
versions, and legacy NPZ files without VIAVSR-7 quality metadata. Rerun `track_webcam_faces.py` after improving
the recording; do not bypass the gate by editing the artifact.

## Current limitations

- The checkpoint is Vietnamese-focused and unreliable for English–Vietnamese code switching.
- Numeric text may produce unknown tokenizer pieces.
- Webcam accuracy depends on alignment, visibility, synchronization, and recording quality.
- The CLI processes one utterance at a time.
- Long recordings should be segmented before inference.
- Beam parameters have not been tuned on a large local development set.
- No language model is included in the released joint decoder.
- The repository provides command-line workflows rather than a recording UI.

## Third-party implementation

Model and decoder reproduction includes code derived from AVSRCocktail. See `src/viavsr/inference/vendor/avsrcocktail/LICENSE` and `src/viavsr/inference/vendor/avsrcocktail/NOTICE.md` for third-party terms and provenance.
