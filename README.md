# viAVSR-laptop

**Vietnamese audio-visual speech recognition for short webcam recordings.**

A record-and-infer application built around the released Vietnamese **AV-HuBERT + joint CTC/Attention** checkpoint. It combines raw-video preprocessing, missing-visual handling, transcript evaluation, and a Streamlit demo in one reproducible Linux workflow.

This project evaluates and integrates an existing model; it does not introduce a new pretrained checkpoint.

## What the project provides

- **End-to-end inference:** face tracking, aligned mouth-ROI extraction, synchronized audio/video features, and Vietnamese decoding.
- **Missing-visual evaluation:** paired comparisons of Corrupted AV, interval-gated AV, and audio-only inference.
- **Web demo:** original video, processed mouth ROI, transcript, runtime, optional WER/CER, and downloadable JSON reports.
- **Reproducible assets:** pinned checkpoint revision and checksum-verified Vietnamese SentencePiece tokenizer.

```text
Recording with audio
  -> face tracking and aligned mouth ROI
  -> synchronized audio/video features
  -> AV-HuBERT -> joint CTC/Attention -> Vietnamese transcript
```

## Quick start

### 1. Install

Use **Linux or WSL2 Ubuntu**, with Conda/Miniconda. The Conda environment uses Python 3.11 and includes FFmpeg/FFprobe. An NVIDIA GPU is recommended; CPU inference is supported but slower.

```bash
git clone --branch feat/ui https://github.com/longdibanbao/viAVSR-laptop.git
cd viAVSR-laptop

conda env create -f environment/environment.yml
conda activate viavsr
python -m pip install -e ".[dev]"

python -c "import torch, viavsr; print('CUDA:', torch.cuda.is_available()); print('Package:', viavsr.__file__)"
```

The command above selects the current demo branch, `feat/ui`. Dependencies pin PyTorch **2.7.1**, torchvision **0.22.1**, and transformers **4.52.4**. See [pyproject.toml](pyproject.toml) and [environment.yml](environment/environment.yml).

If Hugging Face requires authentication, use `huggingface-cli login` with your own read token. Never commit credentials or paste them into command examples.

### 2. Launch the web app

```bash
python -m streamlit run src/viavsr/ui/app.py \
  --server.address=127.0.0.1 \
  --server.port=8501
```

Open **http://localhost:8501**, then upload, record, or select a video. For the Corrupted AV demo, turn **Audio only** off, select **Corrupted AV**, and use **Joint CTC/Attention**.

- The app loads the tokenizer, checkpoint, and face tracker on first use, then caches model resources within the server process. Initial loading is slower than subsequent runs.
- The UI processes the first **3–8 seconds** selected by **Clip duration**. Fast mode uses a 5-second clip and lighter preprocessing.
- A reference transcript is optional and must match the processed clip to produce meaningful WER/CER.
- For timing comparisons, keep the video, clip duration, decoder, and Fast mode setting identical. Report cold-start and warmed-up timings separately.

### 3. Run from the terminal

For CLI use, download the tokenizer once and optionally validate all model assets:

```bash
python scripts/fetch_tokenizer_assets.py --config configs/config.yaml
python scripts/check_model_assets.py --config configs/config.yaml
```

Run a raw video with embedded audio; replace the media path with your own file:

```bash
python scripts/run_avsr_demo.py \
  --config configs/config.yaml \
  --media samples/webcam/webcam_002.mp4 \
  --tracking-device auto \
  --visual-fallback-policy corrupted_av \
  --decoder joint_beam_search \
  --beam-size 3 \
  --ctc-weight 0.1
```

Add `--reference-text "the exact spoken transcript"` for WER/CER. The CLI defaults to a 15-second maximum; local webcam recordings are not included in the repository.

Outputs are saved under `outputs/demo/<video-stem>/`: `report.json` and, when available, `mouth_roi.mp4`. Add `--keep-intermediates` for preprocessing diagnostics. The reusable entry point is `viavsr.demo.run_end_to_end_demo`.

## Missing-visual strategies

| Strategy | Behavior |
| --- | --- |
| **Corrupted AV** (`corrupted_av`) | Neutral-fill missing visual frames and let the visual encoder process the sequence without explicit feature gating. |
| **Interval-gated AV** (`interval_gated`) | Use the same corrupted input, then zero visual features at unavailable time steps before fusion. |
| **Whole-utterance fallback** (`whole_utterance`) | Use ordinary AV when visual preprocessing passes the quality gate; otherwise use audio only for the entire utterance. This is the default policy. |

The UI's **Audio only** toggle explicitly skips face tracking and discards visual input. It differs from choosing a fallback policy. Partial-visual policies are experimental and can still fall back when visual preprocessing is unusable; inspect the actual selected mode in the report.

These are inference-time strategies, not additional training or a rerun of AV-HuBERT masked prediction. The display ROI marks missing intervals with **No visual signal**; the model consumes a separate prepared input.

## Evaluation

### Visual-dropout benchmark

Start with a five-sample smoke test:

```bash
python scripts/run_visual_dropout_benchmark.py \
  --config configs/config.yaml \
  --split test --count 5 \
  --dropout-levels 0.1 0.3 0.5 \
  --seeds 17 29 43 \
  --beam-size 3 --ctc-weight 0.1 \
  --output-dir outputs/visual_dropout_smoke
```

For the full split, use `--count 0` and a new output directory. Audio remains unchanged; each sample/ratio/seed combination shares the same deterministic visual mask across paired modes. The runner also records clean AV, audio-only, and an experimental automatic-routing condition.

Runs produce `benchmark_report.json`, `results.jsonl`, and `execution.log`. Reusing the same command and output directory resumes completed work. Use a new directory whenever the configuration changes.

### Recorded full-test results

Corpus WER (%) on **1,167 ViCocktail test utterances**, using three seeds, beam size 3, and CTC weight 0.1. Dropout results pool the three seeds; clean AV and audio-only are evaluated once per utterance.

| Mode | Clean | 10% visual loss | 30% visual loss | 50% visual loss |
| --- | ---: | ---: | ---: | ---: |
| Clean AV | 9.41 | — | — | — |
| Corrupted AV | — | **9.44** | **10.02** | **11.05** |
| Interval-gated AV | — | 9.59 | 10.29 | 11.11 |

**Audio-only reference: 13.64% WER.** Corrupted AV has the lowest observed WER among the tested missing-visual strategies. These controlled results do not establish statistical significance or guarantee performance on arbitrary webcam or conferencing recordings.

Provenance: local artifact `outputs/runpod_vicocktail_test_full_610f2a6/benchmark_report.json` (generated reports are not committed). The full ledger includes the automatic-routing experiment, omitted from this summary.

WER/CER evaluation uses Unicode NFC, lowercase, a documented punctuation policy, and whitespace normalization while **preserving Vietnamese diacritics**. Predictions are scored without LLM rewriting. Use `scripts/evaluate_transcripts.py` for existing transcripts and `scripts/compare_webcam_modes.py` for a paired raw-webcam comparison; both expose `--help`.

## Configuration and deployment

Runtime settings live in [configs/config.yaml](configs/config.yaml).

| Setting | Default |
| --- | --- |
| Checkpoint | [nguyenvulebinh/AV-HuBERT-CTC-Attention-VI](https://huggingface.co/nguyenvulebinh/AV-HuBERT-CTC-Attention-VI) |
| Revision | `b8a1fa5d6079701b3f8f791bfd601057fbd23de3` |
| Tokenizer | `unigram2048.model` and `unigram2048_units.txt` |
| Model device | `auto` (CUDA if available, otherwise CPU) |
| Decoder | Joint CTC/Attention; beam 3, CTC weight 0.1 |

Use `model.device: cuda` to require GPU inference. Face tracking has its own device setting (`--tracking-device` in the CLI). Tokenizer provenance and checksums are in [manifest.json](assets/tokenizers/vi/manifest.json). The model uses 25 fps video, a 96×96 exported mouth ROI with an 88×88 inference crop, and 16 kHz mono audio.

<details>
<summary><strong>Remote GPU / Runpod</strong></summary>

Run the entire application on the GPU server; teammates only need a browser. A [Dockerfile](Dockerfile) is provided with PyTorch 2.7.1 CUDA 12.8 wheels. Use it with a compatible NVIDIA driver and GPU-enabled container runtime.

For an existing Runpod environment, retain the repository, virtual environment, and caches under `/workspace`. Expose **HTTP port 8501** and replace the example domain below with the URL shown in your Pod's Connect tab:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/.cache/torch

python -m streamlit run src/viavsr/ui/app.py \
  --server.address=0.0.0.0 --server.port=8501 \
  --server.headless=true \
  --browser.serverAddress=YOUR-POD-ID-8501.proxy.runpod.net \
  --browser.serverPort=443 \
  --server.enableCORS=true --server.enableXsrfProtection=true
```

Keep the process running, for example in a `tmux` session. The external domain setting avoids WebSocket origin rejection behind the HTTPS proxy. Disabling `hf_transfer` avoids an inherited fast-download setting when that optional package is absent.

The app has **no built-in authentication** and concurrent users are not validated. Add access control before public deployment and run demo requests sequentially. Stopping a Runpod Pod preserves its volume but still incurs storage charges; terminating it deletes its volume disk. Back up reports first and verify GPU availability after restarting.

</details>

## Development and scope

```text
configs/            Runtime and model settings
environment/        Conda environment
assets/             Tokenizer manifest and local assets
scripts/            CLI entry points
src/viavsr/         Pipeline, preprocessing, inference, evaluation, and UI
tests/              Automated tests
outputs/            Generated reports and media (ignored by Git)
```

Run the lightweight suite without downloading a dataset:

```bash
python -m pytest
```

**Limits:** record-and-infer rather than streaming transcription; one utterance per pipeline call; accuracy depends on visual tracking and audio/video synchronization. Visible-speaker tracking does not establish reliable silent-target attribution. Hypothesis scores are not calibrated word-level confidence, and pipeline timing is not identical to browser click-to-result latency.

**Future work:** a Vietnamese video-conferencing benchmark with paired local/received recordings across Zoom, Google Meet, and Microsoft Teams, followed by robustness fine-tuning and evaluation on unseen speakers and platforms. These are research plans, not released capabilities.

Keep recordings, datasets, checkpoints, caches, generated reports, and secrets out of Git. Third-party code retains its upstream [license](src/viavsr/inference/vendor/avsrcocktail/LICENSE) and [provenance notice](src/viavsr/inference/vendor/avsrcocktail/NOTICE.md).

## References

- [AV-HuBERT — Shi et al., ICLR 2022](https://arxiv.org/abs/2201.02184)
- [ViCocktail — Nguyen et al., Interspeech 2025](https://www.isca-archive.org/interspeech_2025/nguyen25d_interspeech.html)
- [Dropout-induced modality bias — Dai et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Dai_A_Study_of_Dropout-Induced_Modality_Bias_on_Robustness_to_Missing_CVPR_2024_paper.html)
- [AVSR in video conferencing — Huang et al., CVPR 2026](https://arxiv.org/abs/2603.22915)
