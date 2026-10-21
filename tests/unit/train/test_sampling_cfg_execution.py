# pyright: reportPrivateUsage=false
"""Focused tests for the shared TrainingSampler CFG prepare/execute helpers.

Exercises both wrappers (12/24 variant and N/2N raw) on masked CPU with a
stub composite/Qwen/VAE; the deep variant diagnostics path is stubbed at the
method seam, and bit-equality against the pre-refactor code is covered by
the external actual-model oracle.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from sakuramoon.data.caption import CaptionPlan, empty_caption_dropout_hits
from sakuramoon.data.serialize import (
    MAIN_SUFFIX,
    SYSTEM_PREFIX,
    FramingContract,
    SerializedCaption,
)
from sakuramoon.sampling import sampler as sampling_sampler
from sakuramoon.sampling.profiles import SamplingProfile
from sakuramoon.train import sampling
from sakuramoon.train.sampling import (
    TrainingSampleItem,
    TrainingSampler,
    VariantName,
)
from sakuramoon.train.tensor_metrics import TrainingSamplingError

DEVICE = torch.device("cpu")
PAD = 99
VARIANTS: tuple[VariantName, ...] = tuple(
    cast(VariantName, name) for name in sampling._VARIANT_NAMES[:12]
)
sampler_cls = TrainingSampler


class _Tokenizer:
    pad_token_id = PAD

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if text == SYSTEM_PREFIX:
            return list(range(100, 134))
        if text == MAIN_SUFFIX:
            return list(range(200, 205))
        return [1000 + index for index, _character in enumerate(text)]


class _QwenEncoder:
    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        return types.SimpleNamespace(hidden_states=torch.zeros(*input_ids.shape, 8))


class _Qwen:
    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()
        self.encoder = _QwenEncoder()


class _VAE:
    def __init__(self, decoded: Any) -> None:
        self.decoded = decoded

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if callable(self.decoded):
            return cast(torch.Tensor, self.decoded(latent))
        return self.decoded


class _Composite:
    def __init__(self, predicted: int = 2) -> None:
        self.predicted_count = predicted

    def forward_conditioning(self, inputs: Any) -> Any:
        self.conditioning_branches = len(inputs.latents)
        return (types.SimpleNamespace(tokens=None), types.SimpleNamespace(tokens=None))

    def forward_dit(self, inputs: Any, conditioning: Any) -> list[torch.Tensor]:
        self.dit_branches = len(inputs.latents)
        per_branch = [latent.to(torch.float32) for latent in inputs.latents]
        return [
            per_branch[index % len(per_branch)].clone() for index in range(self.predicted_count)
        ]


def _profile() -> SamplingProfile:
    return SamplingProfile(name="test", solver="euler", steps=2, time_schedule="linear")


def _caption() -> SerializedCaption:
    ids = tuple(range(100, 134)) + tuple(range(200, 205))
    return SerializedCaption(
        plan=CaptionPlan(
            tags=(),
            condition=None,
            nl_text=None,
            selected_nl=None,
            all_condition_dropped=True,
            dropout_hits=empty_caption_dropout_hits(all_condition=True),
        ),
        text="",
        input_ids=ids,
        attention_mask=(True,) * len(ids),
        main_token_indices=tuple(range(len(ids))),
        main_mask=(True,) * len(ids),
        condition_token_indices=(),
        condition_mask=(),
        use_null_condition=False,
        condition_source=None,
        condition_role=None,
        all_condition_dropped=True,
        dropout_hits=empty_caption_dropout_hits(all_condition=True),
        selected_nl=None,
        body="",
        condition_text="",
        condition_tokens=0,
        condition_bucket=64,
        dense_length=64,
        truncated=False,
    )


def _item(
    variant: VariantName, ordinal: int, *, height: int = 32, width: int = 32
) -> TrainingSampleItem:
    return TrainingSampleItem(
        ordinal=ordinal,
        variant=variant,
        main_source="A",
        condition_sources=(),
        sample_id=f"s{ordinal}",
        caption=_caption(),
        plan=CaptionPlan(
            tags=(),
            condition=None,
            nl_text=None,
            selected_nl=None,
            all_condition_dropped=True,
            dropout_hits=empty_caption_dropout_hits(all_condition=True),
        ),
        height=height,
        width=width,
        zoom=1.0,
        virtual_canvas_size=(height * 2, width * 2),
        crop_box=(0, 0, height, width),
        coordinate_type="canonical_full_canvas",
    )


def _sampler(tmp_path: Path, *, vae: _VAE, composite: _Composite) -> TrainingSampler:
    sampler = cast(Any, object.__new__(TrainingSampler))
    sampler.config = types.SimpleNamespace(
        timestep=types.SimpleNamespace(t_eps=0.05),
        cfg=types.SimpleNamespace(scale=1.0),
        sampling=types.SimpleNamespace(selected=_profile()),
        train=types.SimpleNamespace(resolution=32),
    )
    sampler.repository_root = tmp_path
    sampler.composite = composite
    sampler.qwen = _Qwen()
    sampler.vae = vae
    sampler.device = DEVICE
    sampler.growth_alpha = 0.0
    return cast(TrainingSampler, sampler)


def _diagnostics_stub(payload: dict[str, float]) -> Any:
    def stub(
        self: TrainingSampler,
        inputs: Any,
        conditioning: Any,
        noise: torch.Tensor,
    ) -> dict[str, float]:
        return payload

    return stub


def _variant_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    payload: dict[str, float] | None = None,
    decoded: Any = None,
    predictions: int = 24,
) -> tuple[tuple[TrainingSampleItem, ...], TrainingSampler]:
    monkeypatch.setattr(
        sampler_cls, "_condition_diagnostics",
        _diagnostics_stub({} if payload is None else payload),
    )
    items = tuple(_item(name, index) for index, name in enumerate(VARIANTS))
    sampler = _sampler(
        tmp_path,
        vae=_VAE(torch.zeros(12, 3, 32, 32) if decoded is None else decoded),
        composite=_Composite(predictions),
    )
    return items, sampler


def _bad_profile(velocity: Any, noise: torch.Tensor, *, profile: SamplingProfile) -> Any:
    return types.SimpleNamespace(state=noise[:, :1])


def _bad_euler(velocity: Any, noise: torch.Tensor, *, steps: int) -> Any:
    velocity(noise, torch.zeros(1, dtype=torch.float32))
    raise AssertionError("euler must not continue after the guard fault")


def _diagnostics_explode(
    self: TrainingSampler, *args: Any, **kwargs: Any
) -> dict[str, float]:
    raise AssertionError("variant diagnostics must not run on the raw path")


def _framing() -> FramingContract:
    return FramingContract(34, 5, PAD)


def _raw_items(count: int = 2) -> tuple[TrainingSampleItem, ...]:
    return tuple(_item("raw", index) for index in range(count))


def test_raw_batch_happy_path_n2n(tmp_path: Path) -> None:
    items = _raw_items(2)
    composite = _Composite(4)
    sampler = _sampler(tmp_path, vae=_VAE(torch.zeros(2, 3, 32, 32)), composite=composite)
    images = sampler._generate_raw_batch(items, shared_seed=7, framing=_framing())
    assert images.shape == (2, 3, 32, 32)
    assert images.dtype == torch.uint8
    assert composite.dit_branches == 4
    assert composite.conditioning_branches == 4


def test_variant_batch_happy_path_12_24(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    items, sampler = _variant_setup(tmp_path, monkeypatch, payload={"text_token_rms": 0.0})
    images, diagnostics = sampler._generate_batch(
        items, shared_seed=7, framing=_framing(), include_fixed_diagnostics=False
    )
    assert images.shape == (12, 3, 32, 32)
    assert images.dtype == torch.uint8
    assert diagnostics == {"text_token_rms": 0.0}
    composite = sampler.composite
    assert composite.conditioning_branches == 24
    assert composite.dit_branches == 24


def test_raw_noise_is_shared_and_seed_deterministic(tmp_path: Path) -> None:
    items = _raw_items(3)

    def shared_probe(latent: torch.Tensor) -> torch.Tensor:
        # The shared noise must appear on every branch row; probe that.
        count = latent.shape[0]
        per_row = latent.view(count, -1)[:, :3]
        shared = per_row[0].expand(count, 3)
        mismatch = (per_row != shared).any()
        assert not bool(mismatch), "initial noise is not shared across the batch"
        return per_row.view(count, 3, 1, 1).expand(count, 3, 32, 32)

    first = _sampler(tmp_path, vae=_VAE(shared_probe), composite=_Composite(6))
    second = _sampler(tmp_path, vae=_VAE(shared_probe), composite=_Composite(6))
    a = first._generate_raw_batch(items, shared_seed=11, framing=_framing())
    b = second._generate_raw_batch(items, shared_seed=11, framing=_framing())
    assert torch.equal(a, b)
    c = second._generate_raw_batch(items, shared_seed=12, framing=_framing())
    assert not torch.equal(a, c)


def test_raw_vae_wrong_shape_keeps_original_message(tmp_path: Path) -> None:
    items = _raw_items(2)
    sampler = _sampler(tmp_path, vae=_VAE(torch.zeros(2, 3, 16, 32)), composite=_Composite(4))
    with pytest.raises(TrainingSamplingError, match="VAE returned an invalid raw sample batch"):
        sampler._generate_raw_batch(items, shared_seed=7, framing=_framing())


def test_variant_vae_wrong_shape_keeps_original_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    items, sampler = _variant_setup(tmp_path, monkeypatch, decoded=torch.zeros(11, 3, 32, 32))
    with pytest.raises(
        TrainingSamplingError, match="VAE returned an invalid 12-image batch"
    ):
        sampler._generate_batch(
            items, shared_seed=7, framing=_framing(), include_fixed_diagnostics=False
        )


def test_raw_vae_nonfinite_keeps_original_message(tmp_path: Path) -> None:
    items = _raw_items(2)
    sampler = _sampler(tmp_path, vae=_VAE(torch.full((2, 3, 32, 32), float("nan"))), composite=_Composite(4))
    with pytest.raises(TrainingSamplingError, match="VAE produced nonfinite raw samples"):
        sampler._generate_raw_batch(items, shared_seed=7, framing=_framing())


def test_variant_vae_nonfinite_keeps_original_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    items, sampler = _variant_setup(
        tmp_path, monkeypatch, decoded=torch.full((12, 3, 32, 32), float("nan"))
    )
    with pytest.raises(
        TrainingSamplingError, match="VAE produced nonfinite training samples"
    ):
        sampler._generate_batch(
            items, shared_seed=7, framing=_framing(), include_fixed_diagnostics=False
        )


def test_raw_dit_count_guard_keeps_original_message(tmp_path: Path) -> None:
    items = _raw_items(2)
    sampler = _sampler(tmp_path, vae=_VAE(torch.zeros(2, 3, 32, 32)), composite=_Composite(3))
    with pytest.raises(
        TrainingSamplingError, match="DiT did not return the raw CFG prediction count"
    ):
        sampler._generate_raw_batch(items, shared_seed=7, framing=_framing())


def test_variant_dit_count_guard_keeps_original_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    items, sampler = _variant_setup(tmp_path, monkeypatch, predictions=23)
    with pytest.raises(TrainingSamplingError, match="DiT did not return 24 CFG predictions"):
        sampler._generate_batch(
            items, shared_seed=7, framing=_framing(), include_fixed_diagnostics=False
        )


def test_raw_profile_state_guard_keeps_original_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    items = _raw_items(2)
    sampler = _sampler(tmp_path, vae=_VAE(torch.zeros(2, 3, 32, 32)), composite=_Composite(4))
    monkeypatch.setattr(sampling, "sample_profile", _bad_profile)
    with pytest.raises(
        TrainingSamplingError, match="raw sample profile returned an invalid state batch"
    ):
        sampler._generate_raw_batch(items, shared_seed=7, framing=_framing())


def test_variant_profile_state_guard_keeps_original_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    items, sampler = _variant_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(sampling, "sample_profile", _bad_profile)
    with pytest.raises(
        TrainingSamplingError, match="sample profile returned an invalid state batch"
    ):
        sampler._generate_batch(
            items, shared_seed=7, framing=_framing(), include_fixed_diagnostics=False
        )


def test_variant_timestep_guard_keeps_original_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    items, sampler = _variant_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(sampling_sampler, "euler", _bad_euler)
    with pytest.raises(TrainingSamplingError, match="sampler state or timestep batch is invalid"):
        sampler._generate_batch(
            items, shared_seed=7, framing=_framing(), include_fixed_diagnostics=False
        )


def test_raw_timestep_guard_keeps_original_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    items = _raw_items(2)
    sampler = _sampler(tmp_path, vae=_VAE(torch.zeros(2, 3, 32, 32)), composite=_Composite(4))
    monkeypatch.setattr(sampling_sampler, "euler", _bad_euler)
    with pytest.raises(TrainingSamplingError, match="raw sampler state or timestep batch is invalid"):
        sampler._generate_raw_batch(items, shared_seed=7, framing=_framing())


def test_variant_diagnostics_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def condition_diagnostics(self: Any, inputs: Any, conditioning: Any, noise: Any) -> dict[str, float]:
        calls.append("condition")
        return {"text_token_rms": 1.0}

    def fixed_diagnostics(self: Any, *, framing: Any) -> dict[str, float]:
        calls.append("fixed")
        return {"fixed_marker": 2.0}

    monkeypatch.setattr(sampler_cls, "_condition_diagnostics", condition_diagnostics)
    monkeypatch.setattr(sampler_cls, "_fixed_condition_diagnostics", fixed_diagnostics)
    items = tuple(_item(name, index) for index, name in enumerate(VARIANTS))
    sampler = _sampler(tmp_path, vae=_VAE(torch.zeros(12, 3, 32, 32)), composite=_Composite(24))
    _, diagnostics = sampler._generate_batch(
        items, shared_seed=7, framing=_framing(), include_fixed_diagnostics=True
    )
    assert diagnostics == {"text_token_rms": 1.0, "fixed_marker": 2.0}
    assert calls == ["condition", "fixed"]
    calls.clear()
    _, diagnostics = sampler._generate_batch(
        items, shared_seed=7, framing=_framing(), include_fixed_diagnostics=False
    )
    assert diagnostics == {"text_token_rms": 1.0}
    assert calls == ["condition"]


def test_raw_path_never_runs_variant_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sampler_cls, "_condition_diagnostics", _diagnostics_explode)
    monkeypatch.setattr(sampler_cls, "_fixed_condition_diagnostics", _diagnostics_explode)
    items = _raw_items(2)
    sampler = _sampler(tmp_path, vae=_VAE(torch.zeros(2, 3, 32, 32)), composite=_Composite(4))
    images = sampler._generate_raw_batch(items, shared_seed=7, framing=_framing())
    assert images.shape == (2, 3, 32, 32)
