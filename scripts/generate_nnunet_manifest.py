from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def build_manifest(dataset_dir: Path) -> pd.DataFrame:
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    if not images_dir.exists() or not labels_dir.exists():
        raise ValueError(f"Expected imagesTr and labelsTr under {dataset_dir}")

    rows = []
    for image_path in sorted(images_dir.glob("*.nii.gz")):
        image_name = image_path.name
        if not image_name.endswith("_0000.nii.gz"):
            continue
        case_id = image_name[: -len("_0000.nii.gz")]
        mask_path = labels_dir / f"{case_id}.nii.gz"
        if not mask_path.exists():
            raise ValueError(f"Missing label for case_id={case_id}: {mask_path}")
        rows.append(
            {
                "case_id": case_id,
                "image_path": image_path.as_posix(),
                "mask_path": mask_path.as_posix(),
                "label": "",
            }
        )

    if not rows:
        raise ValueError(f"No *_0000.nii.gz files found under {images_dir}")
    return pd.DataFrame(rows)


def print_dataset_hint(dataset_dir: Path) -> None:
    dataset_json = dataset_dir / "dataset.json"
    if not dataset_json.exists():
        return
    try:
        payload = json.loads(dataset_json.read_text(encoding="utf-8"))
    except Exception:
        return
    labels = payload.get("labels")
    if labels:
        print("Label map from dataset.json:")
        print(json.dumps(labels, indent=2, ensure_ascii=False))
        print("For PyRadiomics, pick one label value with --label-value or the web page.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a cases manifest from an nnU-Net raw dataset.")
    parser.add_argument("--dataset-dir", required=True, help="Path to nnU-Net raw dataset root.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset_dir).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(dataset_dir)
    manifest.to_csv(output_path, index=False)
    print(f"Wrote {len(manifest)} cases to {output_path}")
    print_dataset_hint(dataset_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
