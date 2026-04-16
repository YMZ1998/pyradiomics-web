# -- coding: utf-8 --
"""Legacy entry script kept for compatibility.

Prefer importing ``classification.select_features`` from application code.
"""

from __future__ import annotations

from pathlib import Path

from classification import build_test_data_examples, select_features


def run_feature_selection_cli(
    features_path: str = "outputs/examples/test_data_features.csv",
    output_dir: str = "outputs/legacy_feature_select",
    labels_path: str = "outputs/examples/test_data_labels.csv",
    top_k: int = 20,
) -> None:
    workspace = Path(__file__).resolve().parents[1]
    examples = build_test_data_examples(workspace)
    artifacts = select_features(
        examples.features if features_path == "outputs/examples/test_data_features.csv" else Path(features_path),
        Path(output_dir),
        labels_path=examples.labels if labels_path == "outputs/examples/test_data_labels.csv" else Path(labels_path),
        top_k=top_k,
    )
    print(f"Selected features saved to: {artifacts.selected_features_path}")
    print(f"Summary saved to: {artifacts.summary_path}")


if __name__ == "__main__":
    run_feature_selection_cli()
