"""Stage 1 adapter: PushT low-dimensional (SETUP.md 5)."""

from __future__ import annotations

from ldva.envs.base import NotImplementedAdapter, register_adapter


class PushTAdapter(NotImplementedAdapter):
    name = "pusht"
    stage = "Stage 1"
    setup_section = "5"
    install_hint = "conda create -n robo-pusht ... && pip install gym-pusht"
    todo = (
        "Build the MetadataSpec from configs/env/pusht.yaml (block x/y/theta, "
        "agent x/y, goal id, perturbation).",
        "Implement collect(): reset the env to the requested block/agent/goal "
        "state, roll out the scripted or trained policy, and slice the "
        "trajectory into chunks of chunk_len (test 8 / 16 / 32).",
        "Record the *realized* reset state as metadata, not the requested one.",
        "Implement evaluation_set(): a fixed grid or sampled distribution over "
        "initial states, drawn once before any acquisition (SETUP.md 33).",
        "Use state observations only; SETUP.md 29 says do not begin with vision.",
        "Everything above the adapter is reusable as-is: BCSupervisionTask "
        "takes any SampleStore plus a validation set.",
    )

    def policy_defaults(self) -> dict:
        return {"kind": "mlp_bc", "hidden": (256, 256)}


register_adapter("pusht", PushTAdapter)
