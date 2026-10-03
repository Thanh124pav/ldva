"""Guards on the experiment scripts' reporting logic.

These are the places where a result gets *interpreted*, so a bug here turns
noise into a claim. Tested directly rather than by running the experiments,
which are far too slow for a test suite.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def loop_mod():
    return _load("experiments/run_acquisition_loop.py", "ldva_loop_exp")


@pytest.fixture(scope="module")
def stage0_mod():
    return _load("experiments/synthetic/run_stage0.py", "ldva_stage0_exp")


def _fake_summary(finals, policy_noise, n_seeds=1):
    return {
        f"m{i}": {
            "final_utility_mean": f,
            "final_utility_std_across_policies": policy_noise,
            "final_utility_std_across_seeds": 0.0,
            "n_seeds": n_seeds,
        }
        for i, f in enumerate(finals)
    }


def test_resolution_check_rejects_differences_inside_the_noise(loop_mod):
    s = _fake_summary([-1.50, -1.49], policy_noise=0.10)
    r = loop_mod._resolution_check(s, list(s))
    assert r["resolvable"] is False
    assert r["method_spread"] == pytest.approx(0.01, abs=1e-9)


def test_resolution_check_accepts_a_real_difference(loop_mod):
    s = _fake_summary([-1.50, -0.50], policy_noise=0.05)
    r = loop_mod._resolution_check(s, list(s))
    assert r["resolvable"] is True


def test_resolution_check_credits_more_seeds(loop_mod):
    """Averaging over seeds shrinks the standard error, so the same difference
    becomes resolvable with enough of them."""
    finals, noise = [-1.50, -1.40], 0.10
    one = loop_mod._resolution_check(_fake_summary(finals, noise, n_seeds=1), ["m0", "m1"])
    many = loop_mod._resolution_check(_fake_summary(finals, noise, n_seeds=25), ["m0", "m1"])
    assert one["resolvable"] is False
    assert many["resolvable"] is True
    assert many["standard_error_per_method"] < one["standard_error_per_method"]


def test_resolution_check_handles_a_single_method(loop_mod):
    r = loop_mod._resolution_check(_fake_summary([-1.0], 0.1), ["m0"])
    assert r["resolvable"] is False


def test_summarize_separates_seed_noise_from_policy_noise(loop_mod):
    runs = [
        {"method": "a", "seed": 0, "history": [
            {"n_samples": 10, "utility": -1.0, "utility_std": 0.2, "success_rate": 0.1},
            {"n_samples": 20, "utility": -0.8, "utility_std": 0.2, "success_rate": 0.2}]},
        {"method": "a", "seed": 1, "history": [
            {"n_samples": 10, "utility": -1.2, "utility_std": 0.2, "success_rate": 0.1},
            {"n_samples": 20, "utility": -0.6, "utility_std": 0.2, "success_rate": 0.3}]},
    ]
    s = loop_mod._summarize(runs, {})["a"]
    assert s["final_utility_mean"] == pytest.approx(-0.7)
    assert s["final_utility_std_across_seeds"] == pytest.approx(0.1)
    assert s["final_utility_std_across_policies"] == pytest.approx(0.2)
    assert s["total_improvement_mean"] == pytest.approx(0.4)
    assert s["n_seeds"] == 2


def test_methods_list_is_complete(loop_mod):
    """SETUP.md 20's minimum suite, plus the two LDVA planners."""
    assert "ldva_beam" in loop_mod.METHODS and "ldva_greedy" in loop_mod.METHODS
    for m in ("random", "equal", "diversity", "uncertainty", "gradient_alignment",
              "influence_cupid_style"):
        assert m in loop_mod.METHODS, m


def test_every_method_is_dispatchable(loop_mod):
    """A name in METHODS with no planner would fail only mid-experiment."""
    for m in loop_mod.METHODS:
        with pytest.raises(Exception) as e:
            loop_mod._plan(m, None, None, None, 0)
        assert "unknown method" not in str(e.value), m
    with pytest.raises(ValueError, match="unknown method"):
        loop_mod._plan("nope", None, None, None, 0)


def test_stage0_criteria_are_the_six_of_setup_30(stage0_mod):
    report = {
        "datamodel": {"final_val": {
            "effect_mse": 0.1, "effect_gain_over_scalar": 2.0, "gain_mse": 0.01,
            "gain_within_mse": 0.02, "gain_within_r2": 0.8}},
        "ablations": {
            "scalar_readout": {"effect_mse": 0.3},
            "additive_utility": {"gain_mse": 0.02, "gain_within_mse": 0.04,
                                 "gain_within_r2": 0.6}},
        "latent_geometry": {
            "neighbor": {"neighbor_consistency_ratio": 0.4, "frac_probes_consistent": 0.8},
            "additivity": {"additive_r2_heldout": -0.3},
            "gain_signal": {"gain_within_group_share": 0.8}},
        "calibration": {"report": {"spearman": 0.7, "picked_the_best": True,
                                   "regret_normalized": 0.1}},
        "metadata_control": {"direction_cosine_mean": 0.4, "frac_directions_positive": 0.9,
                             "reachability_cosine_mean": 0.9,
                             "jacobian_r2_heldout_mean": 0.3},
    }

    class _R:
        def __init__(self, v):
            self.best_value = v

    solvers = {"exact": _R(1.0), "beam_10": _R(1.0), "beam_1": _R(0.9)}
    crit = stage0_mod._success_criteria(report, solvers, [], {})
    assert len(crit) == 6
    assert all(c["passed"] for c in crit.values()), {k: v["passed"] for k, v in crit.items()}
    assert all("rule" in c for c in crit.values())


def test_stage0_criteria_fail_when_the_evidence_is_absent(stage0_mod):
    """A criterion must not pass on a missing or NaN measurement."""
    report = {
        "datamodel": {"final_val": {
            "effect_mse": 0.3, "effect_gain_over_scalar": 0.5, "gain_mse": 0.04,
            "gain_within_mse": 0.04, "gain_within_r2": 0.1}},
        "ablations": {
            "scalar_readout": {"effect_mse": 0.1},
            "additive_utility": {"gain_mse": 0.02, "gain_within_mse": 0.02,
                                 "gain_within_r2": 0.5}},
        "latent_geometry": {
            "neighbor": {"neighbor_consistency_ratio": 1.4, "frac_probes_consistent": 0.3},
            "additivity": {"additive_r2_heldout": 0.999},
            "gain_signal": {"gain_within_group_share": 0.02}},
        "calibration": {"report": {"spearman": float("nan")}},
        "metadata_control": {"direction_cosine_mean": float("nan"),
                             "frac_directions_positive": 0.1,
                             "reachability_cosine_mean": 0.2,
                             "jacobian_r2_heldout_mean": 0.0},
    }
    crit = stage0_mod._success_criteria(report, {}, [], {})
    assert not any(c["passed"] for c in crit.values())
    # criterion 4 is skipped (no exact solver) and must not count as a pass
    assert crit["4_beam_matches_exact"]["passed"] is False


def test_ablation_registry_covers_plan_18(stage0_mod):
    mod = _load("experiments/synthetic/run_ablations.py", "ldva_abl_exp")
    # ten callable ablations; number 9 is folded into 8 and documented as such
    assert set(mod.ABLATIONS) == {1, 2, 3, 4, 5, 6, 7, 8, 10, 11}
    for num, (name, fn) in mod.ABLATIONS.items():
        assert callable(fn) and isinstance(name, str)
