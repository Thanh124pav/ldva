"""Stage 0 adapter (PLAN.md 14)."""

from __future__ import annotations

import numpy as np
import torch

from ldva.data.metadata import MetadataSpec
from ldva.data.samples import SampleStore
from ldva.envs.base import EnvAdapter, register_adapter
from ldva.envs.synthetic.generator import SyntheticConfig, SyntheticWorld


class SyntheticAdapter(EnvAdapter):
    name = "synthetic"

    def __init__(self, cfg: SyntheticConfig | None = None, seed: int = 0):
        self.world = SyntheticWorld(cfg or SyntheticConfig(seed=seed))

    @property
    def metadata_spec(self) -> MetadataSpec:
        return self.world.metadata_spec

    def collect(self, metadata, rng, round_id: int = 0) -> SampleStore:
        return self.world.build_store(
            0, rng, metadata=np.atleast_2d(metadata), round_id=round_id
        )

    def evaluation_set(self, n: int, rng):
        m = self.world.evaluation_metadata(n, rng)
        obs, act, _ = self.world.generate_chunks(m, rng)
        return torch.from_numpy(obs), torch.from_numpy(act)

    def initial_dataset(self, n: int, rng) -> SampleStore:
        """Two tight modes in one corner: coverage is intentionally incomplete."""
        return self.world.build_store(
            n, rng, regions=self.world.default_initial_regions(), round_id=0
        )


register_adapter("synthetic", SyntheticAdapter)
