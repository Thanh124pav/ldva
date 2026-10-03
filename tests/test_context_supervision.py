"""Multi-context supervision format and coverage (PLAN.md 3.2, 6; SETUP.md 11).

The non-negotiable property is that a sample appears in many *different*
contexts. These tests guard the format, the coverage guarantees, and the
padding/masking that lets ragged contexts share a forward pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from ldva.data.context_dataset import (
    ContextDataset,
    ContextRecord,
    EffectNormalizer,
    load_records,
    save_records,
)
from ldva.data.effect_profiles import EffectProfileTable
from ldva.data.metadata import MetadataField, MetadataSpec
from ldva.supervision.generate_context_records import ContextBatchSampler, ContextGenConfig


def test_record_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        ContextRecord(0, "c", np.array([1, 2, 3]), np.array([0.1, 0.2]), 0.0)


def test_dataset_rejects_out_of_range_sample_ids(store):
    bad = ContextRecord(0, "c", np.array([len(store) + 5]), np.array([0.1]), 0.0)
    with pytest.raises(ValueError, match="but store holds"):
        ContextDataset(store, [bad])


def test_coverage_meets_the_multi_context_requirement(context_dataset):
    """SETUP.md 11 recommends 20-100 contexts per sample; the fixture targets 10
    to stay fast, so the test asserts the generator hit its own target."""
    rep = context_dataset.coverage_report(min_contexts=10)
    assert rep["uncovered_samples"] == 0
    assert rep["contexts_per_sample_min"] >= 10
    assert rep["n_checkpoints"] >= 2, "contexts must span multiple checkpoints"
    assert rep["checkpoints_per_sample_mean"] > 1.0


def test_batch_compositions_are_diverse(context_dataset):
    """If the same pair always co-occurs, no readout can separate their
    contributions, so the generator rejects near-duplicate compositions."""
    rep = context_dataset.cooccurrence_stats()
    assert rep["cooccurrence_max"] < context_dataset.coverage_report()["n_contexts"]
    compositions = {tuple(sorted(r.batch_sample_ids.tolist())) for r in context_dataset.records}
    assert len(compositions) == len(context_dataset.records), "duplicate compositions emitted"


def test_same_sample_gets_different_effects_in_different_contexts(context_dataset):
    """The premise of the whole project: effects are contextual, not scalar."""
    per_sample: dict[int, list[float]] = {}
    for r in context_dataset.records:
        for sid, eff in zip(r.batch_sample_ids, r.per_sample_effects):
            per_sample.setdefault(int(sid), []).append(float(eff))
    spreads = [np.std(v) for v in per_sample.values() if len(v) > 2]
    assert len(spreads) > 5
    assert np.median(spreads) > 0, "a sample's effect never varies across contexts"


def test_collate_pads_and_masks_correctly(store):
    spec = store.metadata_spec
    rng = np.random.default_rng(0)
    recs = [
        ContextRecord(i, f"c{i%2}", rng.choice(len(store), size=n, replace=False),
                      rng.normal(size=n), float(rng.normal()),
                      policy_features=np.array([0.1, 0.2, 0.3, 0.4]))
        for i, n in enumerate([3, 7, 5])
    ]
    ds = ContextDataset(store, recs)
    batch = ContextDataset.collate([ds[i] for i in range(3)])
    assert batch["obs"].shape == (3, 7, store.chunk_len, store.obs_dim)
    assert batch["mask"].sum(1).tolist() == [3, 7, 5]
    # padded slots must be zero and masked out
    assert batch["obs"][0, 3:].abs().sum() == 0
    assert (batch["sample_ids"][0, 3:] == -1).all()
    assert not batch["mask"][0, 3:].any()


def test_normalizer_standardizes_and_inverts(context_dataset):
    n = context_dataset.normalizer
    gains = np.array([r.batch_gain for r in context_dataset.records])
    z = n.gain(gains)
    assert abs(z.mean()) < 1e-6 and abs(z.std() - 1.0) < 1e-6
    assert np.allclose(n.inverse_gain(z), gains)


def test_split_by_checkpoint_is_disjoint(context_dataset):
    """Holding out whole checkpoints is the honest policy-transfer test."""
    tr, va = context_dataset.split(0.3, seed=0, by="checkpoint")
    assert set(tr.checkpoint_ids).isdisjoint(set(va.checkpoint_ids))
    assert len(tr) + len(va) == len(context_dataset)


def test_split_by_context_partitions_records(context_dataset):
    tr, va = context_dataset.split(0.25, seed=0, by="context")
    ids_tr = {r.context_id for r in tr.records}
    ids_va = {r.context_id for r in va.records}
    assert ids_tr.isdisjoint(ids_va)
    assert len(ids_tr | ids_va) == len(context_dataset)


def test_records_roundtrip_through_disk(context_dataset, tmp_path):
    p = tmp_path / "records.npz"
    save_records(context_dataset.records, p, context_dataset.normalizer)
    loaded, norm = load_records(p)
    assert len(loaded) == len(context_dataset.records)
    a, b = loaded[2], context_dataset.records[2]
    assert np.array_equal(a.batch_sample_ids, b.batch_sample_ids)
    assert np.allclose(a.per_sample_effects, b.per_sample_effects)
    assert a.policy_ckpt_id == b.policy_ckpt_id
    assert isinstance(norm, EffectNormalizer)


def test_effect_profile_distances_use_only_shared_contexts(store):
    """d_effect must average over contexts the pair actually shares."""
    recs = [
        ContextRecord(0, "c0", np.array([0, 1]), np.array([1.0, 3.0]), 0.0),
        ContextRecord(1, "c0", np.array([0, 1]), np.array([2.0, 2.0]), 0.0),
        ContextRecord(2, "c0", np.array([0, 2]), np.array([5.0, 9.0]), 0.0),
    ]
    t = EffectProfileTable(recs, len(store), min_shared=1)
    # pair (0,1) shares contexts 0 and 1: mean(|1-3|, |2-2|) = 1.0
    assert t.distance(0, 1) == pytest.approx(1.0)
    assert t.distance(1, 0) == pytest.approx(1.0)
    # pair (0,2) shares only context 2: |5-9| = 4.0
    assert t.distance(0, 2) == pytest.approx(4.0)
    # pair (1,2) never co-occurs
    assert t.distance(1, 2) is None


def test_effect_profile_min_shared_filters_pairs(store):
    recs = [
        ContextRecord(0, "c0", np.array([0, 1]), np.array([1.0, 3.0]), 0.0),
        ContextRecord(1, "c0", np.array([0, 2]), np.array([1.0, 3.0]), 0.0),
    ]
    assert len(EffectProfileTable(recs, len(store), min_shared=1)) == 2
    assert len(EffectProfileTable(recs, len(store), min_shared=2)) == 0


def test_effect_profile_lookup_batch_reports_missing_pairs(effect_table):
    pairs = np.array([[0, 1], [0, 2], [10_000 % 1, 0]])
    d, found = effect_table.lookup_batch(pairs)
    assert d.shape == (3,) and found.shape == (3,)
    assert found.dtype == bool


def test_batch_sampler_prioritizes_undercovered_samples():
    cfg = ContextGenConfig(contexts_per_sample=5, batch_size=4, batch_size_jitter=0)
    s = ContextBatchSampler(20, cfg, np.random.default_rng(0))
    for _ in range(40):
        s.propose("ckpt")
    assert s.counts.min() > 0
    # coverage should be far more even than random sampling would give
    assert s.counts.max() - s.counts.min() <= 6


def test_batch_sampler_rejects_near_duplicate_compositions():
    cfg = ContextGenConfig(contexts_per_sample=3, batch_size=4, batch_size_jitter=0,
                           max_jaccard=0.5)
    s = ContextBatchSampler(8, cfg, np.random.default_rng(0))
    seen = [tuple(sorted(s.propose("c").tolist())) for _ in range(6)]
    assert len(set(seen)) == len(seen)


def test_metadata_spec_normalization_and_feasibility():
    spec = MetadataSpec([
        MetadataField("x", -1.0, 1.0),
        MetadataField("task", 0.0, 3.0, discrete=True),
        MetadataField("fixed", 0.0, 1.0, controllable=False),
    ])
    rng = np.random.default_rng(0)
    m = spec.sample(50, rng)
    assert spec.is_feasible(m).all()
    assert np.allclose(m[:, 1], np.round(m[:, 1])), "discrete field not rounded"
    mn = spec.normalize(m)
    assert mn.min() >= -1e-9 and mn.max() <= 1 + 1e-9
    assert np.allclose(spec.denormalize(mn), m)
    assert spec.controllable_mask.tolist() == [True, True, False]
    # out-of-box values are projected back in
    assert spec.is_feasible(spec.clip(np.array([[5.0, 9.0, -3.0]]))).all()


def test_sample_store_roundtrip_and_concat(store, tmp_path):
    p = tmp_path / "store.npz"
    store.save(p)
    from ldva.data.samples import SampleStore

    s2 = SampleStore.load(p)
    assert np.allclose(s2.obs, store.obs)
    assert s2.metadata_spec.names == store.metadata_spec.names
    assert len(store.concat(s2)) == 2 * len(store)
    sub = store.subset(np.array([0, 3, 5]))
    assert len(sub) == 3 and np.allclose(sub.obs[1], store.obs[3])
