"""Optimizer-only stochastic-rounding RNG state isolation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StochasticRoundingRNG:
    device: torch.device
    state: torch.Tensor

    @classmethod
    def seeded(cls, device: torch.device, seed: int) -> StochasticRoundingRNG:
        if device.type != "cuda":
            raise ValueError("stochastic-rounding RNG requires a CUDA device")
        if seed < 0:
            raise ValueError("stochastic-rounding seed must be nonnegative")
        training_state = torch.cuda.get_rng_state(device)
        try:
            torch.cuda.manual_seed(seed)
            state = torch.cuda.get_rng_state(device)
        finally:
            torch.cuda.set_rng_state(training_state, device)
        return cls(device=device, state=state)

    def run_step(self, step: object) -> object:
        if not callable(step):
            raise TypeError("optimizer step must be callable")
        training_state = torch.cuda.get_rng_state(self.device)
        torch.cuda.set_rng_state(self.state, self.device)
        try:
            return step()
        finally:
            self.state = torch.cuda.get_rng_state(self.device)
            torch.cuda.set_rng_state(training_state, self.device)

    def state_dict(self) -> dict[str, object]:
        return {
            "device_type": self.device.type,
            "device_index": self.device.index,
            "state": self.state.clone(),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        if state_dict.get("device_type") != "cuda":
            raise ValueError("SR RNG state has the wrong device type")
        if state_dict.get("device_index") != self.device.index:
            raise ValueError("SR RNG state has the wrong CUDA device index")
        state = state_dict.get("state")
        if (
            not isinstance(state, torch.Tensor)
            or state.device.type != "cpu"
            or state.dtype != torch.uint8
            or state.shape != self.state.shape
            or not state.is_contiguous()
        ):
            raise TypeError(
                "SR RNG state must be a contiguous CPU uint8 tensor with the expected shape"
            )
        self.state = state.clone()


__all__ = ["StochasticRoundingRNG"]
