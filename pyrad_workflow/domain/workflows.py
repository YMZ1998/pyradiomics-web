from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..constants import (
    DEFAULT_CORRELATION_THRESHOLD,
    DEFAULT_FOLDS,
    DEFAULT_GEOMETRY_TOLERANCE,
    DEFAULT_MASK_LABEL,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TOP_K,
    DEFAULT_VARIANCE_THRESHOLD,
)
from ..extraction import extract_features
from ..modeling import select_features, train_and_evaluate
from ..validation import validate_manifest

LOGGER = logging.getLogger("pyrad_workflow")


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


def run_validation_workflow(
    manifest_path: Path,
    output_dir: Path,
    label_value: int = DEFAULT_MASK_LABEL,
    tolerance: float = DEFAULT_GEOMETRY_TOLERANCE,
) -> ValidationArtifacts:
    LOGGER.info("Preparing validation output directory: %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = validate_manifest(manifest_path, label_value=label_value, tolerance=tolerance)
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
) -> ExtractionArtifacts:
    LOGGER.info("Preparing extraction output directory: %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features, failures = extract_features(
        manifest_path,
        params_path,
        output_dir,
        label_value=label_value,
        tolerance=tolerance,
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
) -> TrainingArtifacts:
    LOGGER.info("Preparing training output directory: %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = train_and_evaluate(
        features_path,
        output_dir,
        labels_path=labels_path,
        label_column=label_column,
        folds=folds,
        random_state=random_state,
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


def run_feature_selection_workflow(
    features_path: Path,
    output_dir: Path,
    labels_path: Path | None = None,
    label_column: str = "label",
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
) -> FeatureSelectionArtifacts:
    LOGGER.info("Preparing feature-selection output directory: %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = select_features(
        features_path,
        output_dir,
        labels_path=labels_path,
        label_column=label_column,
        variance_threshold=variance_threshold,
        correlation_threshold=correlation_threshold,
        top_k=top_k,
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
    folds: int = DEFAULT_FOLDS,
    random_state: int = DEFAULT_RANDOM_STATE,
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
    validation_output_dir = output_dir / "01_validation"
    extraction_output_dir = output_dir / "02_features"
    selection_output_dir = output_dir / "03_selection"
    training_output_dir = output_dir / "04_model"

    validation_artifacts = run_validation_workflow(
        manifest_path,
        validation_output_dir,
        label_value=label_value,
        tolerance=tolerance,
    )
    extraction_artifacts = run_extraction_workflow(
        manifest_path,
        params_path,
        extraction_output_dir,
        label_value=label_value,
        tolerance=tolerance,
    )
    selection_artifacts = run_feature_selection_workflow(
        extraction_artifacts.features_path,
        selection_output_dir,
        labels_path=labels_path,
        label_column=label_column,
        variance_threshold=variance_threshold,
        correlation_threshold=correlation_threshold,
        top_k=top_k,
    )
    training_artifacts = run_training_workflow(
        selection_artifacts.selected_features_path,
        training_output_dir,
        labels_path=None,
        label_column=label_column,
        folds=folds,
        random_state=random_state,
    )
    LOGGER.info("Full pipeline finished successfully: output_dir=%s", output_dir)
    return FullPipelineArtifacts(
        validation=validation_artifacts,
        extraction=extraction_artifacts,
        selection=selection_artifacts,
        training=training_artifacts,
        output_dir=output_dir,
    )
