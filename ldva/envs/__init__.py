"""Environment adapters. Importing this registers every known adapter."""

from ldva.envs.base import (  # noqa: F401
    EnvAdapter,
    NotImplementedAdapter,
    available_adapters,
    get_adapter,
    register_adapter,
)
from ldva.envs.dmc.adapter import DMCAdapter  # noqa: F401
from ldva.envs.maniskill.adapter import ManiSkillAdapter  # noqa: F401
from ldva.envs.metaworld.adapter import MetaWorldAdapter  # noqa: F401
from ldva.envs.pusht.adapter import PushTAdapter  # noqa: F401
from ldva.envs.synthetic.adapter import SyntheticAdapter  # noqa: F401

__all__ = [
    "EnvAdapter",
    "NotImplementedAdapter",
    "available_adapters",
    "get_adapter",
    "register_adapter",
    "SyntheticAdapter",
    "DMCAdapter",
    "PushTAdapter",
    "MetaWorldAdapter",
    "ManiSkillAdapter",
]
