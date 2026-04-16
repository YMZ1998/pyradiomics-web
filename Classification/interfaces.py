from __future__ import annotations

from pathlib import Path

from .examples import ExampleDatasetPaths, build_test_data_examples

NON_FEATURE_COLUMNS = {
    "case_id",
    "image_path",
    "mask_path",
    "label",
}


def _modeling_module():
    import importlib

    return importlib.import_module("pyrad_workflow.modeling")


def load_labels(labels_path: Path):
    return _modeling_module().load_labels(labels_path)


def prepare_training_data(features_path: Path, labels_path: Path | None = None, label_column: str = "label"):
    return _modeling_module().prepare_training_data(features_path, labels_path=labels_path, label_column=label_column)


def select_features(
    features_path: Path,
    output_dir: Path,
    labels_path: Path | None = None,
    label_column: str = "label",
    variance_threshold: float = 0.0,
    correlation_threshold: float = 0.95,
    top_k: int = 20,
    selection_method: str = "anova_top_k",
    progress_callback=None,
):
    return _modeling_module().select_features(
        features_path,
        output_dir,
        labels_path=labels_path,
        label_column=label_column,
        variance_threshold=variance_threshold,
        correlation_threshold=correlation_threshold,
        top_k=top_k,
        selection_method=selection_method,
        progress_callback=progress_callback,
    )


def build_models(random_state: int = 42):
    return _modeling_module().build_models(random_state=random_state)


def available_models():
    return _modeling_module().available_models()


def available_selection_methods():
    return _modeling_module().available_selection_methods()


def train_and_evaluate(
    features_path: Path,
    output_dir: Path,
    labels_path: Path | None = None,
    label_column: str = "label",
    folds: int = 5,
    random_state: int = 42,
    model_names=None,
    progress_callback=None,
):
    return _modeling_module().train_and_evaluate(
        features_path,
        output_dir,
        labels_path=labels_path,
        label_column=label_column,
        folds=folds,
        random_state=random_state,
        model_names=model_names,
        progress_callback=progress_callback,
    )


def predict_and_evaluate(
    features_path: Path,
    model_path: Path,
    output_dir: Path,
    labels_path: Path | None = None,
    label_column: str = "label",
    progress_callback=None,
):
    return _modeling_module().predict_and_evaluate(
        features_path,
        model_path,
        output_dir,
        labels_path=labels_path,
        label_column=label_column,
        progress_callback=progress_callback,
    )


__all__ = [
    "ExampleDatasetPaths",
    "NON_FEATURE_COLUMNS",
    "available_models",
    "available_selection_methods",
    "build_models",
    "build_test_data_examples",
    "load_labels",
    "predict_and_evaluate",
    "prepare_training_data",
    "select_features",
    "train_and_evaluate",
]
