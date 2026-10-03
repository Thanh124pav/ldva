"""Acquisition metadata (SETUP.md 10, 17, 36; PLAN.md 12).

Metadata is the *actionable* side of acquisition: the planner picks a latent
direction, and the metadata mapper has to turn it into a feasible change of
simulator / vendor collection settings. That only works if the controllable
variables and their feasible ranges are declared explicitly, so a dataset
always carries a `MetadataSpec` alongside the raw vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class MetadataField:
    """One controllable acquisition variable."""

    name: str
    low: float
    high: float
    controllable: bool = True
    discrete: bool = False
    #: monetary cost weight; used by the cost-aware budget of SETUP.md 31.
    cost: float = 0.0

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(f"field {self.name}: high {self.high} < low {self.low}")

    @property
    def span(self) -> float:
        return float(self.high - self.low)


@dataclass
class MetadataSpec:
    fields: list[MetadataField] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.fields)

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.fields]

    @property
    def low(self) -> np.ndarray:
        return np.array([f.low for f in self.fields], dtype=np.float64)

    @property
    def high(self) -> np.ndarray:
        return np.array([f.high for f in self.fields], dtype=np.float64)

    @property
    def span(self) -> np.ndarray:
        return np.maximum(self.high - self.low, 1e-12)

    @property
    def controllable_mask(self) -> np.ndarray:
        return np.array([f.controllable for f in self.fields], dtype=bool)

    @property
    def discrete_mask(self) -> np.ndarray:
        return np.array([f.discrete for f in self.fields], dtype=bool)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def normalize(self, m: np.ndarray) -> np.ndarray:
        """Map raw metadata into [0, 1] per field."""
        return (np.asarray(m, dtype=np.float64) - self.low) / self.span

    def denormalize(self, m_norm: np.ndarray) -> np.ndarray:
        return np.asarray(m_norm, dtype=np.float64) * self.span + self.low

    def clip(self, m: np.ndarray) -> np.ndarray:
        """Project onto the feasible box, rounding discrete fields."""
        out = np.clip(np.asarray(m, dtype=np.float64), self.low, self.high)
        dm = self.discrete_mask
        if dm.any():
            out[..., dm] = np.round(out[..., dm])
            out = np.clip(out, self.low, self.high)
        return out

    def is_feasible(self, m: np.ndarray, atol: float = 1e-8) -> np.ndarray:
        m = np.asarray(m, dtype=np.float64)
        return np.all((m >= self.low - atol) & (m <= self.high + atol), axis=-1)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Uniform sample from the feasible box."""
        raw = rng.uniform(self.low, self.high, size=(n, len(self)))
        return self.clip(raw)

    def to_dict(self) -> dict:
        return {
            "fields": [
                {
                    "name": f.name,
                    "low": f.low,
                    "high": f.high,
                    "controllable": f.controllable,
                    "discrete": f.discrete,
                    "cost": f.cost,
                }
                for f in self.fields
            ]
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MetadataSpec":
        return cls(fields=[MetadataField(**f) for f in d["fields"]])

    @classmethod
    def from_config(cls, entries: list[dict]) -> "MetadataSpec":
        return cls(fields=[MetadataField(**e) for e in entries])
