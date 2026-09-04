from types import SimpleNamespace

import pytest
import torch
from torch import nn

from viavsr.inference import recognition
from viavsr.inference.decoding import JointBeamSearchHypothesis
from viavsr.inference.errors import InferenceError
from viavsr.inference.recognition import (
    collapse_ctc_predictions,
    recognize_prepared_av,
)
from viavsr.preprocessing.media import MediaMetadata, PreparedAVInput


class _FakeEncoder:
    def __init__(self) -> None:
        self.seen_video: torch.Tensor | None = torch.empty(0)
        self.seen_visual_availability: torch.Tensor | None = None

    def __call__(
        self, *, input_features: torch.Tensor, video: torch.Tensor | None
    ) -> SimpleNamespace:
        del input_features
        self.seen_video = video
        return SimpleNamespace(last_hidden_state=torch.zeros((1, 6, 4)))

    def extract_finetune(
        self,
        source: dict[str, torch.Tensor | None],
        padding_mask: torch.Tensor | None = None,
        visual_availability: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        del padding_mask
        self.seen_video = source["video"]
        self.seen_visual_availability = visual_availability
        return torch.zeros((1, 6, 4)), None


class _FakeCTC:
    def log_softmax(self, features: torch.Tensor) -> torch.Tensor:
        assert features.shape == (1, 6, 4)
        frame_ids = torch.tensor([0, 2, 2, 0, 2, 3])
        logits = torch.full((1, 6, 5), -10.0)
        logits[0, torch.arange(6), frame_ids] = 10.0
        return logits


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.avsr = SimpleNamespace(encoder=_FakeEncoder(), ctc=_FakeCTC())


class _FakeTokenizer:
    def __init__(self) -> None:
        self.decoded_ids: list[int] | None = None
        self.token_list = ["<blank>", "<unk>", "xin", "chào", "<eos>"]

    def decode(self, token_ids: list[int]) -> str:
        self.decoded_ids = token_ids
        return "xin chào"


def _prepared(frames: int = 6) -> PreparedAVInput:
    return PreparedAVInput(
        videos=torch.zeros((1, 1, frames, 88, 88)),
        audios=torch.zeros((1, 104, frames)),
        video_lengths=torch.tensor([frames]),
        audio_lengths=torch.tensor([frames]),
        metadata=MediaMetadata(
            path="/tmp/sample.mp4",
            duration_seconds=frames / 25,
            video_width=96,
            video_height=96,
            frame_rate=25.0,
            audio_sample_rate=16_000,
            audio_channels=1,
        ),
    )


def test_collapse_ctc_predictions_keeps_repeat_separated_by_blank() -> None:
    assert collapse_ctc_predictions([0, 2, 2, 0, 2, 3, 3]) == [2, 2, 3]


def test_recognize_prepared_av_runs_encoder_ctc_and_tokenizer() -> None:
    tokenizer = _FakeTokenizer()
    assets = SimpleNamespace(model=_FakeModel(), tokenizer=tokenizer)

    result = recognize_prepared_av(assets, _prepared())  # type: ignore[arg-type]

    assert result.transcript == "xin chào"
    assert result.token_ids == [2, 2, 3]
    assert result.decoder == "ctc_greedy"
    assert result.input_video_frames == 6
    assert result.encoder_frames == 6
    assert result.device == "cpu"
    assert result.dtype == "float32"
    assert tokenizer.decoded_ids == [2, 2, 3]


def test_recognize_prepared_av_runs_joint_beam_search(monkeypatch) -> None:
    tokenizer = _FakeTokenizer()
    assets = SimpleNamespace(model=_FakeModel(), tokenizer=tokenizer)

    def fake_decode(
        model,
        features,
        token_list,
        *,
        beam_size,
        ctc_weight,
    ) -> JointBeamSearchHypothesis:
        assert model is assets.model.avsr
        assert features.shape == (6, 4)
        assert token_list is tokenizer.token_list
        assert beam_size == 5
        assert ctc_weight == pytest.approx(0.2)
        return JointBeamSearchHypothesis(token_ids=[2, 3], score=-4.5)

    monkeypatch.setattr(
        recognition,
        "decode_joint_ctc_attention",
        fake_decode,
    )

    result = recognize_prepared_av(
        assets,  # type: ignore[arg-type]
        _prepared(),
        decoder="joint_beam_search",
        beam_size=5,
        ctc_weight=0.2,
    )

    assert result.transcript == "xin chào"
    assert result.token_ids == [2, 3]
    assert result.decoder == "joint_beam_search"
    assert result.beam_size == 5
    assert result.ctc_weight == pytest.approx(0.2)
    assert result.hypothesis_score == pytest.approx(-4.5)
    assert result.to_dict()["beam_size"] == 5


def test_recognize_prepared_av_runs_experimental_audio_only() -> None:
    model = _FakeModel()
    assets = SimpleNamespace(model=model, tokenizer=_FakeTokenizer())

    result = recognize_prepared_av(
        assets,  # type: ignore[arg-type]
        _prepared(),
        inference_mode="audio_only_experimental",
    )

    assert model.avsr.encoder.seen_video is None
    assert result.transcript == "xin ch\u00e0o"
    assert result.inference_mode == "audio_only_experimental"
    assert result.visual_input_used is False


def test_recognize_prepared_av_runs_audio_only_fallback() -> None:
    model = _FakeModel()
    assets = SimpleNamespace(model=model, tokenizer=_FakeTokenizer())

    result = recognize_prepared_av(
        assets,  # type: ignore[arg-type]
        _prepared(),
        inference_mode="audio_only_fallback",
    )

    assert model.avsr.encoder.seen_video is None
    assert result.inference_mode == "audio_only_fallback"
    assert result.visual_input_used is False


def test_recognize_prepared_av_applies_corrupted_visual_mask_without_gate() -> None:
    model = _FakeModel()
    assets = SimpleNamespace(model=model, tokenizer=_FakeTokenizer())
    mask = torch.tensor([[True, True, False, False, True, True]])
    base = _prepared()
    prepared = PreparedAVInput(
        videos=torch.ones_like(base.videos),
        audios=base.audios,
        video_lengths=base.video_lengths,
        audio_lengths=base.audio_lengths,
        metadata=base.metadata,
        visual_availability=mask,
    )

    result = recognize_prepared_av(
        assets,  # type: ignore[arg-type]
        prepared,
        inference_mode="audio_visual_corrupted",
    )

    seen_video = model.avsr.encoder.seen_video
    assert seen_video is not None
    assert torch.all(seen_video[:, :, :2] == 1)
    assert torch.all(seen_video[:, :, 2:4] == 0)
    assert torch.all(seen_video[:, :, 4:] == 1)
    assert model.avsr.encoder.seen_visual_availability is None
    assert result.inference_mode == "audio_visual_corrupted"
    assert result.visual_input_used is True
    assert result.visual_coverage == pytest.approx(4 / 6)
    assert result.visual_masked_frames == 2


def test_recognize_prepared_av_applies_interval_visual_gate() -> None:
    model = _FakeModel()
    assets = SimpleNamespace(model=model, tokenizer=_FakeTokenizer())
    mask = torch.tensor([[True, True, False, False, True, True]])
    base = _prepared()
    prepared = PreparedAVInput(
        videos=torch.ones_like(base.videos),
        audios=base.audios,
        video_lengths=base.video_lengths,
        audio_lengths=base.audio_lengths,
        metadata=base.metadata,
        visual_availability=mask,
    )

    result = recognize_prepared_av(
        assets,  # type: ignore[arg-type]
        prepared,
        inference_mode="audio_visual_interval_gated",
    )

    seen_video = model.avsr.encoder.seen_video
    assert seen_video is not None
    assert torch.all(seen_video[:, :, :2] == 1)
    assert torch.all(seen_video[:, :, 2:4] == 0)
    assert torch.all(seen_video[:, :, 4:] == 1)
    assert torch.equal(model.avsr.encoder.seen_visual_availability, mask)
    assert result.inference_mode == "audio_visual_interval_gated"
    assert result.visual_coverage == pytest.approx(4 / 6)
    assert result.visual_masked_frames == 2


@pytest.mark.parametrize(
    "inference_mode",
    ["audio_visual_corrupted", "audio_visual_interval_gated"],
)
def test_masked_visual_modes_require_availability_mask(
    inference_mode: str,
) -> None:
    assets = SimpleNamespace(model=_FakeModel(), tokenizer=_FakeTokenizer())

    with pytest.raises(InferenceError, match="requires visual_availability"):
        recognize_prepared_av(
            assets,  # type: ignore[arg-type]
            _prepared(),
            inference_mode=inference_mode,  # type: ignore[arg-type]
        )


def test_recognize_prepared_av_rejects_feature_length_mismatch() -> None:
    prepared = _prepared()
    prepared = PreparedAVInput(
        videos=prepared.videos,
        audios=torch.zeros((1, 104, 5)),
        video_lengths=prepared.video_lengths,
        audio_lengths=torch.tensor([5]),
        metadata=prepared.metadata,
    )
    assets = SimpleNamespace(model=_FakeModel(), tokenizer=_FakeTokenizer())

    with pytest.raises(InferenceError, match="feature lengths differ"):
        recognize_prepared_av(assets, prepared)  # type: ignore[arg-type]


def test_recognize_prepared_av_wraps_tokenizer_failure() -> None:
    class BrokenTokenizer:
        def decode(self, token_ids: list[int]) -> str:
            del token_ids
            raise ValueError("invalid token")

    assets = SimpleNamespace(model=_FakeModel(), tokenizer=BrokenTokenizer())

    with pytest.raises(InferenceError, match="tokenizer decoding failed") as caught:
        recognize_prepared_av(assets, _prepared())  # type: ignore[arg-type]

    assert caught.value.stage == "decoding"
