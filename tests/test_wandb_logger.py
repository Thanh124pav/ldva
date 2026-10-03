"""The logging layer must never be able to break an experiment.

A four-hour acquisition run that dies at round three because a token expired
has destroyed real compute for no scientific reason, so every one of these
tests is about failing soft.
"""

from __future__ import annotations

import numpy as np

from ldva.analysis.wandb_logger import RunLogger, flatten, make_logger


def test_disabled_logger_is_a_working_no_op():
    """`enabled=False` must be a usable object, not None - call sites should
    not need a guard around every log call."""
    log = make_logger(enabled=False, project="p", name="n")
    assert log.active is False
    assert log.url is None
    # every method must be callable and silent
    log.define_steps({"round": None, "*": "round"})
    log.log({"a": 1.0})
    log.summary({"b": 2})
    log.table("t", ["x"], [[1]])
    log.image("i", "/nonexistent/path.png")
    log.finish()


def test_logger_survives_a_broken_backend(monkeypatch):
    """An import or login failure disables logging and keeps going."""
    import builtins

    real_import = builtins.__import__

    def boom(name, *a, **kw):
        if name == "wandb":
            raise ImportError("no wandb here")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", boom)
    log = RunLogger(enabled=True, project="p", name="n").start()
    assert log.active is False
    assert log.enabled is False
    log.log({"a": 1.0})  # must not raise
    log.finish()


def test_log_drops_non_finite_and_non_numeric():
    """NaN is the normal state for a metric that was not measured this round -
    direction control for a method that plans no direction - and wandb would
    draw it as a gap that looks like a failed step."""
    sent = {}

    class _FakeRun:
        summary: dict = {}

        def log(self, row, step=None):
            sent.update(row)

        def finish(self):
            pass

    log = RunLogger(enabled=True)
    log._run = _FakeRun()
    log.log({
        "good": 1.5,
        "int": 3,
        "bool": True,
        "nan": float("nan"),
        "inf": float("inf"),
        "text": "a string",
        "array": np.zeros(3),
    })
    assert sent == {"good": 1.5, "int": 3.0, "bool": 1}


def test_log_applies_a_prefix():
    sent = {}

    class _FakeRun:
        summary: dict = {}

        def log(self, row, step=None):
            sent.update(row)

    log = RunLogger(enabled=True)
    log._run = _FakeRun()
    log.log({"loss": 0.5}, prefix="datamodel/")
    assert sent == {"datamodel/loss": 0.5}


def test_flatten_nests_keys_and_summarizes_lists():
    """A 40-element allocation vector as 40 series would bury the metrics that
    matter, and the full vector is already in the JSON report."""
    out = flatten({"a": {"b": 1, "c": {"d": 2}}, "alloc": [0, 1, 2, 3]})
    assert out == {"a/b": 1, "a/c/d": 2, "alloc_len": 4}


def test_datamodel_attach_does_not_finish_the_parent_run(
    monkeypatch, store, context_dataset, effect_table
):
    """The acquisition loop retrains the data model every round inside one
    per-(method, seed) run. If the trainer owned that run it would call
    `finish()` each round, ending the parent and splitting one acquisition
    curve across a dozen orphan runs.

    Driven through the real trainer rather than asserting on the config, since
    the bug would live in the ownership logic, not in the flag.
    """
    import sys
    import types

    from ldva.models.datamodel import LDVAConfig, LDVADataModel
    from ldva.training.train_datamodel import TrainConfig, train_datamodel

    calls = {"init": 0, "finish": 0, "logged": []}

    class _Run:
        summary: dict = {}

        def log(self, row, step=None):
            calls["logged"].append((row, step))

        def finish(self):
            calls["finish"] += 1

    active = _Run()
    fake = types.ModuleType("wandb")
    fake.run = active

    def _init(**kw):
        calls["init"] += 1
        return active

    fake.init = _init
    fake.define_metric = lambda *a, **k: None
    fake.Table = lambda columns, data: {"columns": columns, "data": data}
    monkeypatch.setitem(sys.modules, "wandb", fake)

    ds = context_dataset
    model = LDVADataModel(LDVAConfig.build(
        obs_dim=store.obs_dim, act_dim=store.act_dim, chunk_len=store.chunk_len,
        meta_dim=store.meta_dim, policy_feat_dim=ds.policy_feat_dim,
        n_checkpoints=0, latent_dim=8, hidden=(16, 16)))
    model.set_dataset_context(np.zeros((4, 8)))
    train_datamodel(
        model, ds, ds,
        TrainConfig(epochs=2, eval_every=2, wandb=True, wandb_attach=True,
                    wandb_prefix="datamodel/"),
        effect_table)

    assert calls["init"] == 0, "attach mode must not create a run"
    assert calls["finish"] == 0, "attach mode must not end the caller's run"
    assert calls["logged"], "attached metrics were not logged at all"
    # attached rows carry the namespace and no explicit step, so the parent's
    # step axis is left alone
    row, step = calls["logged"][0]
    assert step is None
    assert all(k.startswith("datamodel/") for k in row)
