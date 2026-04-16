from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REQUIRED_MODULES = {
    "classification": "classification package",
    "radiomics": "PyRadiomics",
    "SimpleITK": "SimpleITK",
    "sklearn": "scikit-learn",
    "fastapi": "FastAPI",
    "pandas": "pandas",
}


def main() -> int:
    failures = []
    for module_name, display_name in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module_name)
            print(f"[OK] import {display_name}")
        except Exception as exc:
            failures.append(f"{display_name}: {exc}")

    if failures:
        print("[FAIL] Missing runtime dependencies:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    if shutil.which("uvicorn"):
        print("[OK] command uvicorn")
    else:
        print("[WARN] command uvicorn not found on PATH yet")

    print("[OK] Runtime dependency check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
