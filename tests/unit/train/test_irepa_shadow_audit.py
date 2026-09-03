"""Phase-4 iREPA shadow-gradient audit contract (isolated diagnostic)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from sakuramoon.train.irepa_diagnostics import shadow_gradient_audit


class _AuditModel(nn.Module):
    """p_shared feeds both graphs; p_main only main; p_irepa only iREPA."""

    def __init__(self) -> None:
        super().__init__()
        self.p_shared = nn.Parameter(torch.randn(8))
        self.p_main = nn.Parameter(torch.randn(8))
        self.p_irepa = nn.Parameter(torch.randn(8))

    def main_loss(self) -> torch.Tensor:
        return torch.stack(
            (
                (2.0 * self.p_shared + self.p_main).square().mean(),
                (-1.0 * self.p_shared + self.p_main).square().mean(),
            )
        )

    def irepa_loss(self) -> torch.Tensor:
        return torch.stack(
            (
                (self.p_shared + self.p_irepa).square().mean(),
                (3.0 * self.p_shared + self.p_irepa).square().mean(),
            )
        )


def test_audit_reports_both_graphs_and_restores_grad_state() -> None:
    torch.manual_seed(5)
    model = _AuditModel()
    before = {name: tensor.clone() for name, tensor in model.named_parameters()}

    facts = shadow_gradient_audit(
        module=model,
        main_per_sample=model.main_loss(),
        irepa_per_sample=model.irepa_loss(),
        lambda_weight=1.0,
    )

    shared = facts["p_shared"]
    assert not shared.main_absent
    assert not shared.irepa_absent
    assert shared.norm_main > 0.0
    assert shared.norm_irepa > 0.0
    assert shared.cosine is not None
    assert -1.0 <= shared.cosine <= 1.0

    main_only = facts["p_main"]
    assert not main_only.main_absent
    assert main_only.irepa_absent
    assert main_only.norm_irepa == 0.0
    assert main_only.cosine is None

    irepa_only = facts["p_irepa"]
    assert irepa_only.main_absent
    assert not irepa_only.irepa_absent
    assert irepa_only.norm_main == 0.0
    assert irepa_only.cosine is None

    # grad state is fully restored; parameters are never mutated
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, tensor in model.named_parameters():
        assert torch.equal(before[name], tensor)


def test_audit_lambda_zero_traverses_the_graph_with_exact_zero_grads() -> None:
    torch.manual_seed(5)
    model = _AuditModel()

    facts = shadow_gradient_audit(
        module=model,
        main_per_sample=model.main_loss(),
        irepa_per_sample=model.irepa_loss(),
        lambda_weight=0.0,
    )

    # the iREPA pass still ran: reached parameters carry an exact zero grad
    # (present, not absent), not a missing one
    for name in ("p_shared", "p_irepa"):
        fact = facts[name]
        assert not fact.irepa_absent
        assert fact.norm_irepa == 0.0
    # present in both graphs -> the zero-norm cosine convention applies
    shared = facts["p_shared"]
    assert not shared.main_absent
    assert shared.cosine == 0.0
    # p_irepa is outside the MAIN graph entirely
    assert facts["p_irepa"].main_absent
    assert facts["p_irepa"].cosine is None
    # p_main is outside the iREPA graph entirely
    assert facts["p_main"].irepa_absent
    assert facts["p_main"].norm_irepa == 0.0
    assert facts["p_main"].cosine is None
    assert all(parameter.grad is None for parameter in model.parameters())


def test_audit_main_only_when_irepa_is_none() -> None:
    torch.manual_seed(5)
    model = _AuditModel()

    facts = shadow_gradient_audit(
        module=model,
        main_per_sample=model.main_loss(),
        irepa_per_sample=None,
    )

    for fact in facts.values():
        assert fact.irepa_absent
        assert fact.norm_irepa == 0.0
        assert fact.cosine is None
    assert not facts["p_shared"].main_absent
    assert all(parameter.grad is None for parameter in model.parameters())


def test_audit_default_parameter_names_cover_every_parameter() -> None:
    torch.manual_seed(5)
    model = _AuditModel()

    facts = shadow_gradient_audit(
        module=model,
        main_per_sample=model.main_loss(),
        irepa_per_sample=model.irepa_loss(),
    )

    assert set(facts) == {"p_shared", "p_main", "p_irepa"}


def test_audit_rejects_invalid_arguments() -> None:
    torch.manual_seed(5)
    model = _AuditModel()
    main = model.main_loss()
    irepa = model.irepa_loss()

    with pytest.raises(ValueError, match="one-dimensional"):
        shadow_gradient_audit(
            module=model,
            main_per_sample=main.unsqueeze(-1),
            irepa_per_sample=irepa,
        )
    with pytest.raises(TypeError, match="float32"):
        shadow_gradient_audit(
            module=model,
            main_per_sample=main.bfloat16(),
            irepa_per_sample=irepa,
        )
    with pytest.raises(ValueError, match="shape differs"):
        shadow_gradient_audit(
            module=model,
            main_per_sample=main,
            irepa_per_sample=irepa[:1],
        )
    with pytest.raises(TypeError, match="float32"):
        shadow_gradient_audit(
            module=model,
            main_per_sample=main,
            irepa_per_sample=irepa.double(),
        )
    with pytest.raises(ValueError, match="lambda_weight"):
        shadow_gradient_audit(
            module=model,
            main_per_sample=main,
            irepa_per_sample=None,
            lambda_weight=1,  # int
        )
    with pytest.raises(ValueError, match="lambda_weight"):
        shadow_gradient_audit(
            module=model,
            main_per_sample=main,
            irepa_per_sample=None,
            lambda_weight=-0.1,
        )
    with pytest.raises(ValueError, match="duplicates"):
        shadow_gradient_audit(
            module=model,
            main_per_sample=main,
            irepa_per_sample=None,
            parameter_names=("p_shared", "p_shared"),
        )
    with pytest.raises(ValueError, match="must not be empty"):
        shadow_gradient_audit(
            module=model,
            main_per_sample=main,
            irepa_per_sample=None,
            parameter_names=(),
        )
    with pytest.raises(KeyError, match="unknown parameter name"):
        shadow_gradient_audit(
            module=model,
            main_per_sample=main,
            irepa_per_sample=None,
            parameter_names=("p_missing",),
        )
    # the failed validations leave grad state clean
    assert all(parameter.grad is None for parameter in model.parameters())
