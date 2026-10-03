"""Experiment logging to Weights & Biases (PLAN.md 15 P1, 21).

PLAN.md 15 P1 asks for "structured JSON/CSV/W&B logging" and PLAN.md 21 asks
that every run record its seeds and git SHA. The JSON reports already do the
second; this module adds the first without letting it change what an experiment
measures.

Two rules shape the whole design:

**Logging must never break an experiment.** A four-hour acquisition run that
dies at round three because an API token expired has destroyed real compute for
no scientific reason. Every method here swallows its own exceptions and
degrades to a no-op, and `enabled=False` is a fully functioning object rather
than `None`, so call sites never need a guard.

**One run per (method, seed), grouped by experiment.** This is the unit the
comparison is made over: the wandb UI can then overlay the six methods of
PLAN.md 17 E1 on one chart, average across seeds within a method, and keep E1
separate from E2. Logging one run per *round* would make the acquisition curve
unplottable, and one run for everything would make the methods unseparable.

The step axis is the acquisition **round**, not the optimizer epoch, because
the curve PLAN.md 18 wants is performance against acquired data. The data
model's own per-epoch losses are logged under a `datamodel/` prefix into the
same run via `attach` mode, so they stay inspectable without hijacking the
step axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunLogger:
    """A wandb run, or a no-op object with the same interface.

    `enabled=False`, a missing `wandb` package, and a failed login all produce
    the same thing: an object whose methods do nothing. Call sites stay free of
    `if logger is not None` branches, which is what keeps the logging calls from
    accumulating conditionals in the experiment code.
    """

    enabled: bool = False
    project: str = "ldva"
    name: str | None = None
    group: str | None = None
    job_type: str | None = None
    config: dict = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    notes: str | None = None
    _run: Any = None
    _failed: bool = False

    def start(self) -> "RunLogger":
        if not self.enabled or self._run is not None:
            return self
        try:
            import wandb

            self._run = wandb.init(
                project=self.project,
                name=self.name,
                group=self.group,
                job_type=self.job_type,
                config=self.config,
                tags=list(self.tags) or None,
                notes=self.notes,
                reinit=True,
            )
        except Exception as e:  # noqa: BLE001 - logging must never break a run
            print(f"[ldva] wandb disabled ({type(e).__name__}: {e})", flush=True)
            self._run, self._failed = None, True
            self.enabled = False
        return self

    def define_steps(self, mapping: dict[str, str | None]) -> None:
        """Declare which metric is the x-axis for which metric glob.

        This is not cosmetic - it is what makes the two clocks in one run
        coexist. The loop advances once per acquisition round; the data model
        logs once per epoch, many times within a round. Passing an explicit
        `step=round` for the first while the second logs without a step makes
        wandb's internal counter run ahead, and the next `step=round` call then
        points *backwards* - which wandb silently discards. The symptom is a
        run whose summary is frozen at round 0 while the console shows four
        rounds completing.

        So nothing here passes an explicit step. Each family declares its own
        step metric instead: `round` for the acquisition curve,
        `datamodel/epoch` for the training curves. `mapping` is applied in
        order, so a general glob can be declared first and overridden after.
        """
        if self._run is None:
            return
        try:
            import wandb

            for name, step_metric in mapping.items():
                if step_metric is None:
                    wandb.define_metric(name)
                else:
                    wandb.define_metric(name, step_metric=step_metric)
        except Exception:
            pass

    @property
    def active(self) -> bool:
        return self._run is not None

    @property
    def url(self) -> str | None:
        try:
            return self._run.url if self._run is not None else None
        except Exception:
            return None

    def log(self, row: dict, step: int | None = None, prefix: str = "") -> None:
        """Log scalars. Non-finite and non-numeric values are dropped.

        NaN is the normal state for a metric that was not measured this round -
        direction control for a method that plans no direction, for instance -
        and wandb would otherwise draw it as a gap in a chart that looks like a
        failed step.
        """
        if self._run is None:
            return
        try:
            import math

            clean = {}
            for k, v in row.items():
                key = f"{prefix}{k}" if prefix else k
                if isinstance(v, bool):
                    clean[key] = int(v)
                elif isinstance(v, (int, float)):
                    if math.isfinite(float(v)):
                        clean[key] = float(v)
            if clean:
                self._run.log(clean, step=step)
        except Exception:
            pass

    def summary(self, row: dict) -> None:
        """Final values, which is what the wandb runs table sorts and filters on."""
        if self._run is None:
            return
        try:
            for k, v in row.items():
                if isinstance(v, (int, float, bool, str)):
                    self._run.summary[k] = v
        except Exception:
            pass

    def table(self, name: str, columns: list[str], rows: list[list]) -> None:
        if self._run is None:
            return
        try:
            import wandb

            self._run.log({name: wandb.Table(columns=list(columns), data=[list(r) for r in rows])})
        except Exception:
            pass

    def image(self, name: str, path) -> None:
        if self._run is None:
            return
        try:
            import wandb

            self._run.log({name: wandb.Image(str(path))})
        except Exception:
            pass

    def finish(self) -> None:
        if self._run is None:
            return
        try:
            self._run.finish()
        except Exception:
            pass
        finally:
            self._run = None

    def __enter__(self) -> "RunLogger":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.finish()


def make_logger(
    enabled: bool,
    project: str,
    name: str,
    group: str | None = None,
    job_type: str | None = None,
    config: dict | None = None,
    tags: tuple[str, ...] = (),
) -> RunLogger:
    """Build and start a logger, or a no-op one when logging is off."""
    return RunLogger(
        enabled=bool(enabled),
        project=project,
        name=name,
        group=group,
        job_type=job_type,
        config=dict(config or {}),
        tags=tags,
    ).start()


def flatten(d: dict, prefix: str = "", sep: str = "/") -> dict:
    """Flatten nested dicts into `a/b` keys, which is how wandb groups charts.

    Lists are summarized by length rather than expanded: a 40-element allocation
    vector as 40 separate series would bury the metrics that matter, and the
    full vector is already in the JSON report.
    """
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key, sep))
        elif isinstance(v, (list, tuple)):
            out[f"{key}_len"] = len(v)
        else:
            out[key] = v
    return out
