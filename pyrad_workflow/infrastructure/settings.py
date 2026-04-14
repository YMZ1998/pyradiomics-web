from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppSettings:
    workspace: Path
    max_workers: int = 2


def detect_workspace_root(explicit_workspace: Path | None = None) -> Path:
    if explicit_workspace is not None:
        return explicit_workspace.resolve()

    repo_root = Path(__file__).resolve().parents[2]
    expected_paths = (
        repo_root / "configs",
        repo_root / "manifests",
        repo_root / "pyrad_workflow",
    )
    if all(path.exists() for path in expected_paths):
        return repo_root
    return Path.cwd().resolve()


def build_settings(explicit_workspace: Path | None = None) -> AppSettings:
    return AppSettings(workspace=detect_workspace_root(explicit_workspace))
