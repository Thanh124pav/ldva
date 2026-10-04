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
            "method_class": "ldva",
            "final_mean": f,
            "final_std_across_policies": policy_noise,
            "final_std_across_seeds": 0.0,
            "n_seeds": n_seeds,
        }
        for i, f in enumerate(finals)
    }


def test_resolution_check_rejects_differences_inside_the_noise(loop_mod):
    s = _fake_summary([-1.50, -1.49], policy_noise=0.10)
    r = loop_mod._resolution_check(s, list(s), "utility")
    assert r["resolvable"] is False
    assert r["method_spread"] == pytest.approx(0.01, abs=1e-9)


def test_resolution_check_accepts_a_real_difference(loop_mod):
    s = _fake_summary([-1.50, -0.50], policy_noise=0.05)
    r = loop_mod._resolution_check(s, list(s), "utility")
    assert r["resolvable"] is True


def test_resolution_check_credits_more_seeds(loop_mod):
    """Averaging over seeds shrinks the standard error, so the same difference
    becomes resolvable with enough of them."""
    finals, noise = [-1.50, -1.40], 0.10
    one = loop_mod._resolution_check(
        _fake_summary(finals, noise, n_seeds=1), ["m0", "m1"], "utility")
    many = loop_mod._resolution_check(
        _fake_summary(finals, noise, n_seeds=25), ["m0", "m1"], "utility")
    assert one["resolvable"] is False
    assert many["resolvable"] is True
    assert many["standard_error_per_method"] < one["standard_error_per_method"]


def test_resolution_check_handles_a_single_method(loop_mod):
    r = loop_mod._resolution_check(_fake_summary([-1.0], 0.1), ["m0"], "utility")
    assert r["resolvable"] is False


def test_resolution_check_names_the_metric_it_ranked(loop_mod):
    """The report must say WHICH metric a ranking is over: on an adapter with
    no simulator it is the BC proxy, which PLAN.md 18 does not accept as a
    robotics outcome, and that distinction cannot be left implicit."""
    s = _fake_summary([-1.0, -0.5], 0.05)
    assert loop_mod._resolution_check(s, list(s), "rollout_return")["metric"] == (
        "rollout_return")
    assert loop_mod._resolution_check(s, list(s), "utility")["metric"] == "utility"


def test_summarize_separates_seed_noise_from_policy_noise(loop_mod):
    runs = [
        {"method": "a", "method_class": "ldva", "seed": 0, "wall_time_s": 1.0,
         "history": [
            {"n_samples": 10, "utility": -1.0, "utility_std": 0.2},
            {"n_samples": 20, "utility": -0.8, "utility_std": 0.2}]},
        {"method": "a", "method_class": "ldva", "seed": 1, "wall_time_s": 1.0,
         "history": [
            {"n_samples": 10, "utility": -1.2, "utility_std": 0.2},
            {"n_samples": 20, "utility": -0.6, "utility_std": 0.2}]},
    ]
    s = loop_mod._summarize(runs, "utility")["a"]
    assert s["final_mean"] == pytest.approx(-0.7)
    assert s["final_std_across_seeds"] == pytest.approx(0.1)
    assert s["final_std_across_policies"] == pytest.approx(0.2)
    assert s["total_improvement_mean"] == pytest.approx(0.4)
    assert s["n_seeds"] == 2


def test_summarize_ranks_on_the_rollout_metric_when_present(loop_mod):
    """PLAN.md 15 P0.1: the robot metric is the headline, not the BC loss.

    The two move in opposite directions here on purpose - utility worsens while
    return improves - so a summary that silently ranked on utility would be
    caught.
    """
    runs = [
        {"method": "a", "method_class": "ldva", "seed": 0, "wall_time_s": 1.0,
         "history": [
            {"n_samples": 10, "utility": -0.5, "utility_std": 0.1,
             "rollout_return": 10.0, "rollout_return_std": 1.0,
             "rollout_success": 0.1, "rollout_success_std": 0.0},
            {"n_samples": 20, "utility": -0.9, "utility_std": 0.1,
             "rollout_return": 40.0, "rollout_return_std": 1.0,
             "rollout_success": 0.5, "rollout_success_std": 0.0}]},
    ]
    s = loop_mod._summarize(runs, "rollout_return")["a"]
    assert s["final_mean"] == pytest.approx(40.0)
    assert s["total_improvement_mean"] == pytest.approx(30.0)
    assert s["rollout_success_mean"][-1] == pytest.approx(0.5)


def test_methods_list_is_plan_17_e1(loop_mod):
    """PLAN.md 17 E1 names exactly this comparison."""
    assert set(loop_mod.METHODS) == {
        "random", "diversity", "direct_gradient_alignment", "direct_influence",
        "ldva_greedy", "ldva_beam",
    }


def test_ablations_are_not_in_the_default_method_set(loop_mod):
    """PLAN.md 12: the two groups must not be mixed.

    An `abl_`-prefixed rule scores through the LDVA model, so letting one into
    the default set would put an ablation in a baseline table.
    """
    from ldva.acquisition.external_baselines import classify_method

    for m in loop_mod.METHODS:
        assert not m.startswith("abl_"), m
        assert classify_method(m) in ("external", "model_free", "ldva"), m
    for m in loop_mod.ABLATION_METHODS:
        assert m not in loop_mod.METHODS, m
        assert m.startswith("abl_"), m


def test_external_baselines_are_classified_as_external(loop_mod):
    """PLAN.md 12.2: these compute their own scores and are reported as
    independent baselines; the LDVA-scored lookalikes are not."""
    from ldva.acquisition.external_baselines import classify_method

    assert classify_method("direct_gradient_alignment") == "external"
    assert classify_method("direct_influence") == "external"
    assert classify_method("gradient_alignment") == "ldva_ablation"
    assert classify_method("influence_cupid_style") == "ldva_ablation"
    assert classify_method("random") == "model_free"
    assert classify_method("ldva_beam") == "ldva"


def test_every_method_is_dispatchable(loop_mod):
    """A name in METHODS with no planner would fail only mid-experiment."""
    for m in loop_mod.METHODS + loop_mod.ABLATION_METHODS:
        with pytest.raises(Exception) as e:
            loop_mod._plan(m, {})
        assert "unknown method" not in str(e.value), m
    with pytest.raises(ValueError, match="unknown method"):
        loop_mod._plan("nope", {})


def test_main_model_has_no_checkpoint_id_embedding_by_default(loop_mod):
    """PLAN.md 4.2 / 15 P0.4: a checkpoint-ID embedding memorizes the
    checkpoints it saw and says nothing about an unseen future policy, so the
    main result must not use one."""
    args = loop_mod.argparse.Namespace(
        env="synthetic", quick=True, config=None, rounds=None, budget=None,
        eval_policies=None, eval_rollouts=None, n_initial=None,
        ckpt_id_ablation=False, val_split_by=None)
    cfg = loop_mod.build_config(args)
    assert cfg["ckpt_id_ablation"] is False
    # and the validation split must hold out whole checkpoints, not contexts
    assert cfg["val_split_by"] == "checkpoint"

    args.ckpt_id_ablation = True
    assert loop_mod.build_config(args)["ckpt_id_ablation"] is True


def test_stage0_criterion6_rejects_a_high_cosine_in_a_collapsed_space(stage0_mod):
    """A raw cosine is uninterpretable when the latent space is collapsed.

    Measured over 8 seeds, a realized cosine of +0.910 sat only 1.3 standard
    deviations above requesting a *different* direction, because the encoder
    uses ~1 of 32 latent dimensions and every candidate points into the same
    subspace. The old rule (cosine > 0.3) therefore passed degenerate
    representations; the specificity z-score must reject this one.
    """
    report = {
        "datamodel": {"final_val": {"effect_mse": 0.1, "effect_gain_over_scalar": 2.0,
                                    "gain_mse": 0.01, "gain_within_mse": 0.02,
                                    "gain_within_r2": 0.8}},
        "ablations": {"scalar_readout": {"effect_mse": 0.3},
                      "additive_utility": {"gain_mse": 0.02, "gain_within_mse": 0.04,
                                           "gain_within_r2": 0.6}},
        "latent_geometry": {
            "neighbor": {"neighbor_consistency_ratio": 0.4, "frac_probes_consistent": 0.8},
            "additivity": {"additive_r2_heldout": -0.3},
            "gain_signal": {"gain_within_group_share": 0.8}},
        "calibration": {"report": {"spearman": 0.7, "picked_the_best": True,
                                   "regret_normalized": 0.1}},
        # a cosine that looks excellent, from a one-dimensional latent space
        "metadata_control": {"direction_cosine_mean": 0.91,
                             "frac_directions_positive": 1.0,
                             "reachability_cosine_mean": 0.98,
                             "jacobian_r2_heldout_mean": 0.4},
        "direction_specificity": {"z_score_mean": 1.3,
                                  "frac_directions_above_2sd": 0.0,
                                  "null_abs_mean": 0.88},
        "latent_participation_ratio": 1.04,
    }
    crit = stage0_mod._success_criteria(report, {}, [], {})["6_metadata_moves_latents"]
    assert crit["passed"] is False, "a collapsed space must not pass on its raw cosine"
    assert crit["direction_cosine_mean"] == 0.91
    assert crit["specificity_z_score"] == 1.3


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
        # criterion 6 is scored on specificity against the other candidate
        # directions, not on the raw cosine: with a collapsed latent space the
        # old 0.3 threshold sat below chance
        "direction_specificity": {"z_score_mean": 3.1, "frac_directions_above_2sd": 0.8,
                                  "null_abs_mean": 0.2},
        "latent_participation_ratio": 6.0,
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
        "direction_specificity": {"z_score_mean": float("nan"),
                                  "frac_directions_above_2sd": 0.0,
                                  "null_abs_mean": 0.9},
        "latent_participation_ratio": 1.1,
    }
    crit = stage0_mod._success_criteria(report, {}, [], {})
    assert not any(c["passed"] for c in crit.values())
    # criterion 4 is skipped (no exact solver) and must not count as a pass
    assert crit["4_beam_matches_exact"]["passed"] is False


def test_ablation_registry_covers_plan_20(stage0_mod):
    """PLAN.md 20 lists 13 required ablations, keyed by its own numbering.

    Item 13 needs a real rollout and so cannot run in the synthetic script; it
    must be declared as simulator-only rather than quietly missing.
    """
    mod = _load("experiments/synthetic/run_ablations.py", "ldva_abl_exp")
    required = set(range(1, 14))
    covered = set(mod.ABLATIONS) | set(mod.ABLATIONS_REQUIRING_SIMULATOR)
    assert required <= covered, sorted(required - covered)
    assert 13 in mod.ABLATIONS_REQUIRING_SIMULATOR
    for num, (name, fn) in mod.ABLATIONS.items():
        assert callable(fn) and isinstance(name, str)
    for num, (name, why) in mod.ABLATIONS_REQUIRING_SIMULATOR.items():
        assert isinstance(name, str) and isinstance(why, str) and why
