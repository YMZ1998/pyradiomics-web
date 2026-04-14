from __future__ import annotations

from datetime import datetime
from pathlib import Path


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def resolve_user_path(workspace: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (workspace / candidate).resolve()


def ensure_output_dir(workspace: Path, provided: str | None, default_leaf: str) -> Path:
    chosen = resolve_user_path(workspace, provided)
    if chosen is None:
        chosen = workspace / "outputs" / default_leaf
    chosen.mkdir(parents=True, exist_ok=True)
    return chosen.resolve()


def display_path(path: Path, workspace: Path) -> str:
    try:
        return str(path.relative_to(workspace))
    except ValueError:
        return str(path)


def make_download_listing(output_dir: Path, workspace: Path) -> list[dict[str, str]]:
    files = []
    for file_path in sorted(output_dir.rglob("*")):
        if not file_path.is_file():
            continue
        files.append(
            {
                "name": file_path.name,
                "path": str(file_path),
                "display_path": display_path(file_path, workspace),
            }
        )
    return files
