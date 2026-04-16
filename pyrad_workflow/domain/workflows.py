from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from classification import predict_and_evaluate, select_features, train_and_evaluate

from ..constants import (
    DEFAULT_CORRELATION_THRESHOLD,
    DEFAULT_FOLDS,
    DEFAULT_GEOMETRY_TOLERANCE,
    DEFAULT_MASK_LABEL,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SELECTION_METHOD,
    DEFAULT_TOP_K,
    DEFAULT_VARIANCE_THRESHOLD,
)
from ..extraction import extract_features
from ..infrastructure.filesystem import clear_output_dir
from ..validation import validate_manifest

LOGGER = logging.getLogger("pyrad_workflow")
ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class ValidationArtifacts:
    report: pd.DataFrame
    report_path: Path
    output_dir: Path
    invalid_count: int


@dataclass(frozen=True)
class ExtractionArtifacts:
    features: pd.DataFrame
    failures: pd.DataFrame
    features_path: Path
    failures_path: Path
    output_dir: Path


@dataclass(frozen=True)
class TrainingArtifacts:
    metrics: pd.DataFrame
    metrics_path: Path
    output_dir: Path


@dataclass(frozen=True)
class FeatureSelectionArtifacts:
    cleaned_features: pd.DataFrame
    selected_features: pd.DataFrame
    summary: pd.DataFrame
    cleaned_features_path: Path
    selected_features_path: Path
    summary_path: Path
    output_dir: Path


@dataclass(frozen=True)
class FullPipelineArtifacts:
    validation: ValidationArtifacts
    extraction: ExtractionArtifacts
    selection: FeatureSelectionArtifacts
    training: TrainingArtifacts
    output_dir: Path


@dataclass(frozen=True)
class PredictionWorkflowArtifacts:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    predictions_path: Path
    metrics_path: Path | None
    confusion_matrix_path: Path | None
    output_dir: Path
    model_name: str


def run_validation_workflow(
    manifest_path: Path,
    output_dir: Path,
    label_value: int = DEFAULT_MASK_LABEL,
    tolerance: float = DEFAULT_GEOMETRY_TOLERANCE,
    progress_callback: ProgressCallback | None = None,
) -> ValidationArtifacts:
    LOGGER.info("Preparing validation output directory: %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_output_dir(output_dir)
    report = validate_manifest(
        manifest_path,
        label_value=label_value,
        tolerance=tolerance,
        progress_callback=progress_callback,
    )
    report_path = output_dir / "validation_report.csv"
    report.to_csv(report_path, index=False)
    invalid_count = int((~report["is_valid"].astype(bool)).sum())
    LOGGER.info(
        "Validation workflow finished: manifest=%s total_cases=%s invalid_cases=%s report=%s",
        manifest_path,
        len(report),
        invalid_count,
        report_path,
    )
    return ValidationArtifacts(report=report, report_path=report_path, output_dir=output_dir, invalid_count=invalid_count)


def run_extraction_workflow(
    manifest_path: Path,
    params_path: Path,
    output_dir: Path,
    label_value: int = DEFAULT_MASK_LABEL,
    tolerance: float = DEFAULT_GEOMETRY_TOLERANCE,
    progress_callback: ProgressCallback | None = None,
) -> ExtractionArtifacts:
    LOGGER.info("Preparing extraction output directory: %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_output_dir(output_dir)
    features, failures = extract_features(
        manifest_path,
        params_path,
        output_dir,
        label_value=label_value,
        tolerance=tolerance,
        progress_callback=progress_callback,
    )
    features_path = output_dir / "features.csv"
    failures_path = output_dir / "feature_failures.csv"
    features.to_csv(features_path, index=False)
    failures.to_csv(failures_path, index=False)
    LOGGER.info(
        "Extraction workflow finished: manifest=%s features=%s failures=%s features_path=%s failures_path=%s",
        manifest_path,
        len(features),
        len(failures),
        features_path,
        failures_path,
    )
    return ExtractionArtifacts(
        features=features,
        failures=failures,
        features_path=features_path,
        failures_path=failures_path,
        output_dir=output_dir,
    )


def run_training_workflow(
    features_path: Path,
    output_dir: Path,
    labels_path: Path | None = None,
    label_column: str = "label",
    folds: int = DEFAULT_FOLDS,
    random_state: int = DEFAULT_RANDOM_STATE,
    model_names: list[str] | tuple[str, ...] | str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TrainingArtifacts:
    LOGGER.info("Preparing training output directory: %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_output_dir(output_dir)
    metrics = train_and_evaluate(
        features_path,
        output_dir,
        labels_path=labels_path,
        label_column=label_column,
        folds=folds,
        random_state=random_state,
        model_names=model_names,
        progress_callback=progress_callback,
    )
    metrics_path = output_dir / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    LOGGER.info(
        "Training workflow finished: features=%s metrics_rows=%s metrics_path=%s",
        features_path,
        len(metrics),
        metrics_path,
    )
    return TrainingArtifacts(metrics=metrics, metrics_path=metrics_path, output_dir=output_dir)


def run_prediction_workflow(
    features_path: Path,
    model_path: Path,
    output_dir: Path,
    labels_path: Path | None = None,
    label_column: str = "label",
    progress_callback: ProgressCallback | None = None,
) -> PredictionWorkflowArtifacts:
    LOGGER.info("Preparing prediction output directory: %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_output_dir(output_dir)
    artifacts = predict_and_evaluate(
        features_path,
        model_path,
        output_dir,
        labels_path=labels_path,
        label_column=label_column,
        progress_callback=progress_callback,
    )
    LOGGER.info(
        "Prediction workflow finished: features=%s model=%s predictions=%s metrics=%s",
        features_path,
        model_path,
        artifacts.predictions_path,
        artifacts.metrics_path,
    )
    return PredictionWorkflowArtifacts(
        predictions=artifacts.predictions,
        metrics=artifacts.metrics,
        predictions_path=artifacts.predictions_path,
        metrics_path=artifacts.metrics_path,
        confusion_matrix_path=artifacts.confusion_matrix_path,
        output_dir=artifacts.output_dir,
        model_name=artifacts.model_name,
    )


def run_feature_selection_workflow(
    features_path: Path,
    output_dir: Path,
    labels_path: Path | None = None,
    label_column: str = "label",
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    selection_method: str = DEFAULT_SELECTION_METHOD,
    progress_callback: ProgressCallback | None = None,
) -> FeatureSelectionArtifacts:
    LOGGER.info("Preparing feature-selection output directory: %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_output_dir(output_dir)
    artifacts = select_features(
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
    LOGGER.info(
        "Feature selection finished: features=%s cleaned=%s selected=%s summary=%s",
        features_path,
        artifacts.cleaned_features_path,
        artifacts.selected_features_path,
        artifacts.summary_path,
    )
    return FeatureSelectionArtifacts(
        cleaned_features=artifacts.cleaned_features,
        selected_features=artifacts.selected_features,
        summary=artifacts.summary,
        cleaned_features_path=artifacts.cleaned_features_path,
        selected_features_path=artifacts.selected_features_path,
        summary_path=artifacts.summary_path,
        output_dir=output_dir,
    )


def run_full_pipeline_workflow(
    manifest_path: Path,
    params_path: Path,
    output_dir: Path,
    labels_path: Path | None = None,
    label_column: str = "label",
    label_value: int = DEFAULT_MASK_LABEL,
    tolerance: float = DEFAULT_GEOMETRY_TOLERANCE,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    selection_method: str = DEFAULT_SELECTION_METHOD,
    folds: int = DEFAULT_FOLDS,
    random_state: int = DEFAULT_RANDOM_STATE,
    model_names: list[str] | tuple[str, ...] | str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> FullPipelineArtifacts:
    LOGGER.info(
        "Starting full pipeline: manifest=%s params=%s output_dir=%s label_value=%s tolerance=%s top_k=%s folds=%s",
        manifest_path,
        params_path,
        output_dir,
        label_value,
        tolerance,
        top_k,
        folds,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_output_dir(output_dir)
    validation_output_dir = output_dir / "01_validation"
    extraction_output_dir = output_dir / "02_features"
    selection_output_dir = output_dir / "03_selection"
    training_output_dir = output_dir / "04_model"

    def stage_callback(start: float, end: float, prefix: str) -> ProgressCallback:
        def _callback(progress: float, detail: str) -> None:
            if progress_callback is None:
                return
            mapped = start + ((end - start) * (progress / 100.0))
            progress_callback(mapped, f"{prefix}: {detail}")

        return _callback

    validation_artifacts = run_validation_workflow(
        manifest_path,
        validation_output_dir,
        label_value=label_value,
        tolerance=tolerance,
        progress_callback=stage_callback(0.0, 20.0, "Validation"),
    )
    extraction_artifacts = run_extraction_workflow(
        manifest_path,
        params_path,
        extraction_output_dir,
        label_value=label_value,
        tolerance=tolerance,
        progress_callback=stage_callback(20.0, 68.0, "Extraction"),
    )
    selection_artifacts = run_feature_selection_workflow(
        extraction_artifacts.features_path,
        selection_output_dir,
        labels_path=labels_path,
        label_column=label_column,
        variance_threshold=variance_threshold,
        correlation_threshold=correlation_threshold,
        top_k=top_k,
        selection_method=selection_method,
        progress_callback=stage_callback(68.0, 84.0, "Selection"),
    )
    training_artifacts = run_training_workflow(
        selection_artifacts.selected_features_path,
        training_output_dir,
        labels_path=None,
        label_column=label_column,
        folds=folds,
        random_state=random_state,
        model_names=model_names,
        progress_callback=stage_callback(84.0, 100.0, "Training"),
    )
    LOGGER.info("Full pipeline finished successfully: output_dir=%s", output_dir)
    return FullPipelineArtifacts(
        validation=validation_artifacts,
        extraction=extraction_artifacts,
        selection=selection_artifacts,
        training=training_artifacts,
        output_dir=output_dir,
    )
