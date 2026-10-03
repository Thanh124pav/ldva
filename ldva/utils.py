"""Small shared helpers: seeding, device selection, config loading, run dirs."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python / numpy / torch together."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SeedBundle:
    """PLAN.md 21: the five seed roles are recorded independently.

    Derived seeds stay stable for a given base seed so that, e.g., changing the
    acquisition search seed does not perturb data generation.
    """

    ROLES = ("env", "policy", "context", "latent", "acquisition")

    def __init__(self, base: int = 0, **overrides: int):
        self.base = int(base)
        self._seeds = {r: self.base + 1000 * i for i, r in enumerate(self.ROLES)}
        for k, v in overrides.items():
            if k not in self._seeds:
                raise KeyError(f"unknown seed role {k!r}, expected one of {self.ROLES}")
            self._seeds[k] = int(v)

    def __getitem__(self, role: str) -> int:
        return self._seeds[role]

    def rng(self, role: str) -> np.random.Generator:
        return np.random.default_rng(self._seeds[role])

    def as_dict(self) -> dict[str, int]:
        return {"base": self.base, **self._seeds}

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"SeedBundle({self.as_dict()})"


def get_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving a single level of `_base_` inheritance."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base is not None:
        parent = load_config(path.parent / base)
        cfg = deep_update(parent, cfg)
    return cfg


def deep_update(base: dict, update: dict) -> dict:
    out = dict(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def make_run_dir(root: str | Path, name: str) -> Path:
    run_dir = Path(root) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_jsonable(obj), f, indent=2, sort_keys=True)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def git_sha(short: bool = False) -> str:
    """Current commit, with `-dirty` when the tree has uncommitted changes.

    PLAN.md 15 P1 and 21 require the SHA on every run. `unknown` rather than an
    exception if this is not a git checkout, so recording provenance can never
    be the thing that kills an experiment.
    """
    import subprocess

    try:
        root = Path(__file__).resolve().parents[1]
        rev = subprocess.run(
            ["git", "rev-parse", "--short" if short else "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=10,
        )
        if rev.returncode != 0:
            return "unknown"
        sha = rev.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root, capture_output=True, text=True, timeout=10,
        )
        return sha + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def run_provenance(extra: dict | None = None) -> dict:
    """Everything needed to say what produced a result (PLAN.md 21).

    Recorded once per run and written into the report next to the numbers, so a
    figure can always be traced back to the code and environment that made it.
    """
    import platform
    import sys

    def _ver(mod: str) -> str:
        try:
            return __import__(mod).__version__
        except Exception:
            return "absent"

    sim = {}
    for m in ("dm_control", "mujoco", "metaworld", "gymnasium"):
        try:
            __import__(m)
            sim[m] = _ver(m)
        except Exception:
            sim[m] = "absent"

    return {
        "git_sha": git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "simulators": sim,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "none"),
        "command": " ".join(sys.argv),
        **(extra or {}),
    }


def config_hash(cfg: dict, *parts: Any) -> str:
    """Stable short hash of a config, for cache keys.

    Sorted JSON so key order cannot change the hash, and `_jsonable` first so
    numpy scalars and tuples hash the same as the plain values they stand for.
    """
    import hashlib

    blob = json.dumps(
        [_jsonable(cfg), [_jsonable(p) for p in parts]], sort_keys=True
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")
