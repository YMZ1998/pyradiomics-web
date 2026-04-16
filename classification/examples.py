from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ExampleDatasetPaths:
    manifest: Path
    labels: Path
    features: Path
    params: Path


def build_test_data_examples(workspace: Path) -> ExampleDatasetPaths:
    workspace = workspace.resolve()
    candidate_roots = [
        workspace / "classification" / "test_data",
        workspace / "Classification" / "test_data",
    ]
    test_data_root = next((path for path in candidate_roots if path.is_dir()), None)
    if test_data_root is None:
        raise FileNotFoundError(f"Test data directory not found in: {candidate_roots}")

    examples_dir = workspace / "outputs" / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = examples_dir / "test_data_manifest.csv"
    labels_path = examples_dir / "test_data_labels.csv"
    features_path = examples_dir / "test_data_features.csv"
    params_path = workspace / "configs" / "ct_radiomics.yaml"

    manifest_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    feature_frames: list[pd.DataFrame] = []
    label_mapping = {"a": 1, "b": 0}

    for group_name, label_value in label_mapping.items():
        group_dir = test_data_root / group_name
        case_ids = sorted(path.name for path in group_dir.iterdir() if path.is_dir())

        for case_id in case_ids:
            case_dir = group_dir / case_id
            manifest_rows.append(
                {
                    "case_id": case_id,
                    "image_path": str((case_dir / f"{case_id}.nii.gz").resolve()),
                    "mask_path": str((case_dir / f"{case_id}_seg.nii.gz").resolve()),
                    "label": label_value,
                }
            )
            label_rows.append({"case_id": case_id, "label": label_value})

        feature_source = test_data_root / f"{group_name}.csv"
        if feature_source.is_file():
            frame = pd.read_csv(feature_source).copy()
            frame.insert(0, "case_id", case_ids[: len(frame)])
            frame.insert(1, "label", label_value)
            feature_frames.append(frame)

    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    pd.DataFrame(label_rows).to_csv(labels_path, index=False)
    pd.concat(feature_frames, ignore_index=True).to_csv(features_path, index=False)

    return ExampleDatasetPaths(
        manifest=manifest_path,
        labels=labels_path,
        features=features_path,
        params=params_path,
    )
