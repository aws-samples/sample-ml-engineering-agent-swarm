"""Project memory — the ``.agenticml.yaml`` CLAUDE.md analog (doc 05 §3.1c; ADR-016).

``.agenticml.yaml`` is the canonical, auto-loaded **ML project memory** the surface
honors on every turn — the spec defaults, steering (hard/soft/thresholds), KPIs/
acceptance thresholds, medallion prefixes, and the ``*_BACKEND`` selectors (doc 05
§8.5). It is the CLAUDE.md analog: version-controlled memory the agent reads so the
user never re-explains their setup (US-CC.01).

**Hierarchy (doc 05 §3.1c — the Claude Code enterprise → user → project → directory
analog).** This module merges, in increasing precedence (later wins):

    org/enterprise  →  user (~/.agenticml/config.yaml)  →  project (./.agenticml.yaml)
                       →  directory (a nested ./sub/.agenticml.yaml)

i.e. the **closer scope wins** (directory beats project beats user beats org). M1
ships the **project + user** layers at minimum (the org layer is accepted as an
optional ``org_config_path``), per the task.

**Precedence vs env (doc 05 §3.1c / §8.5 / env_loader precedence).** Memory provides
**DEFAULTS**; an explicit shell/env ``*_BACKEND`` still wins. ``resolve_selector`` and
``apply_to_env`` honor this: env beats memory, and ``apply_to_env`` only sets a key
that is NOT already present (never clobbering a shell export) — exactly the
``config/env_loader`` root-layer contract.

``init_project_memory`` (T-CTL-02) is the ``/init`` analog: scan repo + data, draft a
``.agenticml.yaml`` from ``.agenticml.yaml.example`` + the detected project state.
``project_init_flow`` wraps it. Pure YAML + filesystem — no AWS, no boto3.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = ["ProjectMemory", "init_project_memory", "project_init_flow"]

PROJECT_MEMORY_FILE = ".agenticml.yaml"
EXAMPLE_FILE = ".agenticml.yaml.example"
DEFAULT_USER_CONFIG = "~/.agenticml/config.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file into a dict; missing/empty/non-dict → ``{}`` (never raises)."""
    try:
        if not path.is_file():
            return {}
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` onto ``base`` (overlay wins); returns a new dict.

    Nested dicts merge key-by-key; any non-dict value (incl. lists) is replaced
    wholesale by the overlay — the closer scope wins for that key.
    """
    out: dict[str, Any] = dict(base)
    for key, val in overlay.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


@dataclass
class ProjectMemory:
    """The merged ``.agenticml.yaml`` project memory (the CLAUDE.md analog).

    ``data`` is the merged dict across the layer hierarchy; ``layers`` records which
    files contributed (for the trace / debugging); ``exists`` is True when at least
    one memory file was found.
    """

    data: dict[str, Any] = field(default_factory=dict)
    layers: list[str] = field(default_factory=list)
    exists: bool = False

    # ── loading + the hierarchy ─────────────────────────────────────────────
    @classmethod
    def load(
        cls,
        *,
        project_dir: str | Path = ".",
        dir_path: str | Path | None = None,
        user_config_path: str | Path | None = None,
        org_config_path: str | Path | None = None,
        use_user_layer: bool = True,
    ) -> ProjectMemory:
        """Load + merge the layer hierarchy (later/closer scope wins).

        Order (increasing precedence): org → user → project → directory.

        Args:
            project_dir: The project root holding ``.agenticml.yaml`` (the project layer).
            dir_path: An optional nested directory whose ``.agenticml.yaml`` overrides
                the project root (the directory layer — closest scope).
            user_config_path: The user layer (default ``~/.agenticml/config.yaml``).
                Only read when ``use_user_layer`` is True.
            org_config_path: An optional org/enterprise layer (lowest precedence).
            use_user_layer: Set False to skip the user layer (hermetic tests).

        Returns:
            A merged :class:`ProjectMemory`.
        """
        merged: dict[str, Any] = {}
        layers: list[str] = []
        found = False

        # 1) org / enterprise (lowest precedence)
        if org_config_path:
            org = _read_yaml(Path(org_config_path).expanduser())
            if org:
                merged = _deep_merge(merged, org)
                layers.append(f"org:{org_config_path}")
                found = True

        # 2) user
        if use_user_layer:
            user_p = Path(user_config_path).expanduser() if user_config_path else Path(DEFAULT_USER_CONFIG).expanduser()
            user = _read_yaml(user_p)
            if user:
                merged = _deep_merge(merged, user)
                layers.append(f"user:{user_p}")
                found = True

        # 3) project (./.agenticml.yaml)
        proj_p = Path(project_dir).expanduser() / PROJECT_MEMORY_FILE
        proj = _read_yaml(proj_p)
        if proj:
            merged = _deep_merge(merged, proj)
            layers.append(f"project:{proj_p}")
            found = True

        # 4) directory (a nested scope) — closest, wins last
        if dir_path is not None:
            dir_p = Path(dir_path).expanduser() / PROJECT_MEMORY_FILE
            # avoid double-applying when dir_path == project_dir
            if dir_p.resolve() != proj_p.resolve():
                d = _read_yaml(dir_p)
                if d:
                    merged = _deep_merge(merged, d)
                    layers.append(f"dir:{dir_p}")
                    found = True

        return cls(data=merged, layers=layers, exists=found)

    # ── accessors ────────────────────────────────────────────────────────────
    def backend_selectors(self) -> dict[str, str]:
        """Return the ``*_BACKEND`` selector map from the ``backend_selectors`` block."""
        sel = self.data.get("backend_selectors")
        if not isinstance(sel, dict):
            return {}
        return {str(k): str(v) for k, v in sel.items()}

    def steering(self) -> dict[str, Any] | None:
        """Return the ``steering`` block (pins/forbids/thresholds) if present."""
        st = self.data.get("steering")
        return st if isinstance(st, dict) else None

    def resolve_selector(self, key: str) -> str | None:
        """Resolve a ``*_BACKEND`` selector: **env wins**, else the memory default.

        Mirrors the existing precedence (shell/env > memory; doc 05 §8.5): memory is a
        DEFAULT the operator's explicit env still overrides.
        """
        env_val = os.environ.get(key)
        if env_val:
            return env_val
        return self.backend_selectors().get(key)

    def apply_to_env(self) -> dict[str, str]:
        """Export the memory's ``*_BACKEND`` selectors as env **defaults**.

        Never clobbers a key already present in ``os.environ`` (a shell export wins) —
        the same root-layer contract as ``config/env_loader._apply_file`` (override
        False). Returns only the keys this call actually set (for the trace).
        """
        applied: dict[str, str] = {}
        for key, val in self.backend_selectors().items():
            if key in os.environ:
                continue  # shell/env value wins — do not clobber
            os.environ[key] = val
            applied[key] = val
        return applied


# ── T-CTL-02 init_project_memory (the /init analog) ─────────────────────────
def _repo_root() -> Path:
    """Repo root = parents[2] of ``src/amlas/control_plane/project_memory.py``."""
    return Path(__file__).resolve().parents[3]


def _find_example(repo_path: Path) -> Path | None:
    """Locate ``.agenticml.yaml.example`` (the target repo first, then THIS repo root)."""
    for cand in (repo_path / EXAMPLE_FILE, _repo_root() / EXAMPLE_FILE):
        if cand.is_file():
            return cand
    return None


_DEFAULT_BACKEND_SELECTORS = {
    "TRAINING_BACKEND": "sagemaker",
    "EVALUATION_BACKEND": "arbiter",
    "LAKE_BACKEND": "s3medallion",
    "ORCHESTRATION_BACKEND": "local_driver",
}


def init_project_memory(*, repo_path: str | Path, data_paths: list[str] | None = None) -> dict[str, Any]:
    """``T-CTL-02`` — scaffold ``.agenticml.yaml`` from the example + detected state.

    The ``/init`` analog (US-CC.01): scans ``repo_path`` (+ the supplied ``data_paths``),
    drafts a ``.agenticml.yaml`` project-memory file seeded from
    ``.agenticml.yaml.example`` and augmented with a ``backend_selectors`` block + the
    detected data paths. Does NOT overwrite an existing memory (idempotent; no
    ``force`` at M1) and never touches AWS.

    Args:
        repo_path: The project root to scan + write ``.agenticml.yaml`` into.
        data_paths: Optional list of dataset paths (relative to ``repo_path``) to record.

    Returns:
        ``{"agenticml_yaml": <path>, "created": bool, "detected": {...}}``.
    """
    repo = Path(repo_path).expanduser()
    target = repo / PROJECT_MEMORY_FILE
    data_paths = list(data_paths or [])

    detected = _detect_project_state(repo, data_paths)

    if target.exists():
        # Idempotent: never clobber an existing memory (the /init analog refuses to
        # overwrite at M1; an explicit re-init/force is a later milestone).
        return {"agenticml_yaml": str(target), "created": False, "detected": detected}

    base: dict[str, Any] = {}
    example = _find_example(repo)
    if example is not None:
        base = _read_yaml(example)

    # Augment with the M1 control-plane blocks: backend_selectors + detected state.
    base.setdefault("backend_selectors", dict(_DEFAULT_BACKEND_SELECTORS))
    base.setdefault("approval", {"mode": "ask"})  # safe-by-default (ADR-014)
    base["project_memory"] = {
        "scaffolded_by": "init_project_memory (T-CTL-02)",
        "detected": detected,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# .agenticml.yaml — ML project memory (the CLAUDE.md analog; ADR-016).\n"
        "# Auto-loaded each session: spec/steering/KPIs, medallion config, and the\n"
        "# *_BACKEND selectors. Drafted by init_project_memory (T-CTL-02 / the /init analog).\n\n"
    )
    target.write_text(header + yaml.safe_dump(base, sort_keys=False, default_flow_style=False), encoding="utf-8")
    return {"agenticml_yaml": str(target), "created": True, "detected": detected}


def _detect_project_state(repo: Path, data_paths: list[str]) -> dict[str, Any]:
    """Detect cheap project signals for the scaffold (data files, specs dir, lake config)."""
    detected: dict[str, Any] = {"data_paths": list(data_paths)}
    # auto-detect a data/ dir if no explicit data_paths supplied
    if not data_paths:
        data_dir = repo / "data"
        if data_dir.is_dir():
            found = [str(p.relative_to(repo)) for p in sorted(data_dir.rglob("*.csv"))][:10]
            if found:
                detected["data_paths"] = found
    detected["has_specs_dir"] = (repo / "specs").is_dir()
    detected["has_data_dir"] = (repo / "data").is_dir()
    return detected


def project_init_flow(*, repo_path: str | Path, data_paths: list[str] | None = None) -> dict[str, Any]:
    """``project_init_flow`` — the cross-cutting flow wrapping ``init_project_memory``.

    Runs the ``/init`` scaffold then loads the resulting memory so the caller gets the
    drafted file path + the merged memory in one step (doc 06 ``project_init_flow`` row).
    """
    result = init_project_memory(repo_path=repo_path, data_paths=data_paths)
    memory = ProjectMemory.load(project_dir=str(repo_path), use_user_layer=False)
    return {
        "flow_id": "project_init_flow",
        "agenticml_yaml": result["agenticml_yaml"],
        "created": result["created"],
        "detected": result["detected"],
        "backend_selectors": memory.backend_selectors(),
    }
