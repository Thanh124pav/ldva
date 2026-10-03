"""Set-level utility and context encoding (PLAN.md 4.2-4.4, 22)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ldva.models.batch_utility import BatchUtilityConfig, build_batch_utility
from ldva.models.context_encoder import (
    ContextEncoder,
    ContextEncoderConfig,
    SetTransformerContextEncoder,
)


@pytest.fixture
def ctx_cfg():
    return ContextEncoderConfig(latent_dim=6, phi_dim=12, out_dim=5, hidden=(16,))


@pytest.mark.parametrize("pool", ["mean", "sum"])
def test_context_encoder_is_permutation_invariant(ctx_cfg, pool):
    ctx_cfg.pool = pool
    enc = ContextEncoder(ctx_cfg).eval()
    z = torch.randn(3, 6, 6)
    mask = torch.ones(3, 6, dtype=torch.bool)
    perm = torch.randperm(6)
    with torch.no_grad():
        assert torch.allclose(enc(z, mask), enc(z[:, perm], mask[:, perm]), atol=1e-5)


@pytest.mark.parametrize("pool", ["mean", "sum"])
def test_exact_leave_one_out_matches_bruteforce(ctx_cfg, pool):
    """The O(N) LOO decomposition must equal the O(N^2) recompute exactly.

    The readout depends on h_{B\\i} for every member, so an approximation here
    would quietly corrupt every effect label the model is trained on.
    """
    ctx_cfg.pool = pool
    enc = ContextEncoder(ctx_cfg).eval()
    z = torch.randn(4, 6, 6)
    mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]],
        dtype=torch.bool,
    )
    with torch.no_grad():
        fast = enc.leave_one_out(z, mask)
        slow = enc._leave_one_out_bruteforce(z, mask)
    assert torch.allclose(fast, slow, atol=1e-5)


def test_single_member_sees_empty_context(ctx_cfg):
    """A batch of one has no context; it must get the empty-set encoding."""
    enc = ContextEncoder(ctx_cfg).eval()
    z = torch.randn(1, 3, 6)
    mask = torch.tensor([[True, False, False]])
    with torch.no_grad():
        h = enc.leave_one_out(z, mask)
        empty = enc.rho(torch.zeros(1, enc.rho[0].in_features))
    assert torch.allclose(h[0, 0], empty[0], atol=1e-6)


def test_padding_does_not_leak_into_context(ctx_cfg):
    """Changing padded slots must not change any real member's context."""
    enc = ContextEncoder(ctx_cfg).eval()
    z = torch.randn(2, 6, 6)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    z2 = z.clone()
    z2[~mask] = 99.0
    with torch.no_grad():
        a, b = enc(z, mask), enc(z2, mask)
        la, lb = enc.leave_one_out(z, mask), enc.leave_one_out(z2, mask)
    assert torch.allclose(a, b, atol=1e-5)
    assert torch.allclose(la[mask], lb[mask], atol=1e-5)


def test_set_transformer_is_permutation_invariant(ctx_cfg):
    enc = SetTransformerContextEncoder(ctx_cfg).eval()
    z = torch.randn(2, 6, 6)
    mask = torch.ones(2, 6, dtype=torch.bool)
    perm = torch.randperm(6)
    with torch.no_grad():
        assert torch.allclose(enc(z, mask), enc(z[:, perm], mask[:, perm]), atol=1e-4)
    assert enc.exact_loo is False


def test_additive_utility_is_exactly_additive():
    """The ablation control must be incapable of representing interactions."""
    cfg = BatchUtilityConfig(latent_dim=6, hidden=(16,), pool="sum",
                             use_second_moment=False, use_size_feature=False)
    add = build_batch_utility(cfg, "additive").eval()
    z = torch.randn(1, 5, 6)
    full = torch.ones(1, 5, dtype=torch.bool)
    one = torch.ones(1, 1, dtype=torch.bool)
    with torch.no_grad():
        whole = add(z, full).item()
        parts = sum(add(z[:, i : i + 1], one).item() for i in range(5))
    assert abs(whole - parts) < 1e-5


def test_deepsets_utility_can_represent_interactions():
    """The set-level model must NOT be additive, or ablation 3 is vacuous."""
    cfg = BatchUtilityConfig(latent_dim=6, hidden=(16,), pool="sum")
    torch.manual_seed(0)
    ds = build_batch_utility(cfg, "deepsets").eval()
    z = torch.randn(1, 5, 6)
    full = torch.ones(1, 5, dtype=torch.bool)
    one = torch.ones(1, 1, dtype=torch.bool)
    with torch.no_grad():
        whole = ds(z, full).item()
        parts = sum(ds(z[:, i : i + 1], one).item() for i in range(5))
    assert abs(whole - parts) > 1e-3


def test_pairwise_utility_matrix_is_symmetric():
    cfg = BatchUtilityConfig(latent_dim=6, hidden=(16,))
    pw = build_batch_utility(cfg, "pairwise").eval()
    z = torch.randn(2, 4, 6)
    with torch.no_grad():
        K = pw.pairwise_matrix(z)
    assert torch.allclose(K, K.transpose(1, 2), atol=1e-6)


def test_utility_model_is_permutation_invariant():
    cfg = BatchUtilityConfig(latent_dim=6, hidden=(16,))
    for kind in ("deepsets", "additive", "pairwise"):
        m = build_batch_utility(cfg, kind).eval()
        z = torch.randn(1, 5, 6)
        mask = torch.ones(1, 5, dtype=torch.bool)
        perm = torch.randperm(5)
        with torch.no_grad():
            assert torch.allclose(m(z, mask), m(z[:, perm], mask[:, perm]), atol=1e-5), kind


def test_datamodel_scores_hypothetical_latents_without_chunks(datamodel, trained_policy):
    """The planner's entry point works on latents that have no raw data."""
    _, ckpts = trained_policy
    pf = ckpts[0].features
    Z = np.random.default_rng(0).normal(size=(3, 5, datamodel.latent_dim))
    v = datamodel.utility_from_latents(Z, policy_features=pf)
    e = datamodel.effect_from_latents(Z, policy_features=pf)
    assert v.shape == (3,) and e.shape == (3, 5)
    assert torch.isfinite(v).all() and torch.isfinite(e).all()


def test_additivity_diagnostic_distinguishes_additive_from_interacting():
    """The F2 gate must not report perfect additivity on interacting data,
    and must stay honest when the fit is underdetermined."""
    from ldva.training.metrics import additivity_report

    rng = np.random.default_rng(0)
    n, n_ctx = 40, 400
    v = rng.normal(size=n)
    comps = [rng.choice(n, size=6, replace=False) for _ in range(n_ctx)]
    additive = np.array([v[c].sum() for c in comps])
    interacting = np.array([v[c].sum() + 5.0 * np.prod(np.sort(v[c])[-2:]) for c in comps])

    r_add = additivity_report(additive, comps, n)
    r_int = additivity_report(interacting, comps, n)
    assert r_add["additive_r2_heldout"] > 0.95
    assert r_int["additive_r2_heldout"] < r_add["additive_r2_heldout"]

    # underdetermined: in-sample looks perfect, held-out must not
    few = additivity_report(interacting[:20], comps[:20], 200)
    assert few["contexts_per_param"] < 1.0
    assert few["additive_r2_heldout"] < few["additive_r2_insample"]
