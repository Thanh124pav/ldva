"""Stage 3 adapter: ManiSkill (SETUP.md 7)."""

from __future__ import annotations

from ldva.envs.base import NotImplementedAdapter, register_adapter


class ManiSkillAdapter(NotImplementedAdapter):
    name = "maniskill"
    stage = "Stage 3"
    setup_section = "7"
    install_hint = "the existing robo-maniskill conda env"
    todo = (
        "Use the tasks in configs/env/maniskill.yaml.",
        "Implement collect() using ManiSkill's settable simulator state, which "
        "is what makes metadata-conditioned acquisition clean here.",
        "State observations first; move to RGB only once the acquisition "
        "pipeline works end to end (SETUP.md 7).",
        "Only attempt this after the method is stable on MetaWorld.",
    )

    def policy_defaults(self) -> dict:
        return {"kind": "ppo"}


register_adapter("maniskill", ManiSkillAdapter)
