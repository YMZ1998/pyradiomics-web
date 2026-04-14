from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

LOGGER = logging.getLogger("pyrad_workflow")


NON_FEATURE_COLUMNS = {
    "case_id",
    "image_path",
    "mask_path",
    "label",
}


class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.95) -> None:
        self.threshold = threshold
        self.selected_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        frame = pd.DataFrame(X).copy()
        correlation = frame.corr(numeric_only=True).abs()
        upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
        self.selected_columns_ = [column for column in frame.columns if column not in to_drop]
        return self

    def transform(self, X: pd.DataFrame):
        frame = pd.DataFrame(X).copy()
        return frame.loc[:, self.selected_columns_]


@dataclass(frozen=True)
class TrainingArtifacts:
    case_ids: pd.Series
    features: pd.DataFrame
    labels: pd.Series
    label_column: str


@dataclass(frozen=True)
class SelectionArtifacts:
    cleaned_features: pd.DataFrame
    selected_features: pd.DataFrame
    summary: pd.DataFrame
    cleaned_features_path: Path
    selected_features_path: Path
    summary_path: Path


def load_labels(labels_path: Path) -> pd.DataFrame:
    LOGGER.info("Loading labels file: %s", labels_path)
    labels = pd.read_csv(labels_path)
    required = {"case_id", "label"}
    missing = required - set(labels.columns)
    if missing:
        LOGGER.error("Labels file missing required columns: %s", ", ".join(sorted(missing)))
        raise ValueError(f"Labels file missing required columns: {', '.join(sorted(missing))}")
    LOGGER.info("Labels file loaded successfully with %s rows", len(labels))
    return labels


def prepare_training_data(
    features_path: Path,
    labels_path: Path | None = None,
    label_column: str = "label",
) -> TrainingArtifacts:
    LOGGER.info("Loading training features: %s", features_path)
    features = pd.read_csv(features_path)
    if "case_id" not in features.columns:
        LOGGER.error("Features file is missing case_id column: %s", features_path)
        raise ValueError("Features file must contain case_id column.")

    merged = features.copy()
    if labels_path is not None:
        LOGGER.info("Merging external labels into feature table")
        labels = load_labels(labels_path)
        merged = merged.drop(columns=["label"], errors="ignore").merge(labels, on="case_id", how="left")
        label_column = "label"

    if label_column not in merged.columns:
        LOGGER.error("No label column found in training data: label_column=%s", label_column)
        raise ValueError(
            "No label column found. Provide --labels with columns case_id,label "
            "or include a non-empty label column in the features file."
        )

    merged[label_column] = merged[label_column].replace("", np.nan)
    if merged[label_column].isna().all():
        LOGGER.error("Training labels are entirely empty")
        raise ValueError(
            "Labels are empty. Provide a labels CSV with case_id,label or populate the label column before training."
        )
    if merged[label_column].isna().any():
        missing_cases = merged.loc[merged[label_column].isna(), "case_id"].astype(str).tolist()
        LOGGER.error("Training labels are missing for cases: %s", ", ".join(missing_cases))
        raise ValueError(
            "Labels are missing for some cases: "
            + ", ".join(missing_cases)
            + ". Ensure every case_id has a label before training."
        )

    numeric_frame = merged.copy()
    for column in list(numeric_frame.columns):
        if column in NON_FEATURE_COLUMNS:
            continue
        numeric_frame[column] = pd.to_numeric(numeric_frame[column], errors="coerce")

    feature_columns = [
        column
        for column in numeric_frame.columns
        if column not in NON_FEATURE_COLUMNS and np.issubdtype(numeric_frame[column].dtype, np.number)
    ]
    if not feature_columns:
        LOGGER.error("No numeric feature columns found in %s", features_path)
        raise ValueError("No numeric feature columns found in features file.")

    x = numeric_frame.loc[:, feature_columns]
    y = merged[label_column].astype(str)
    LOGGER.info(
        "Prepared training data: cases=%s numeric_features=%s classes=%s",
        len(merged),
        len(feature_columns),
        sorted(y.unique()),
    )
    return TrainingArtifacts(case_ids=merged["case_id"].astype(str), features=x, labels=y, label_column=label_column)


def _build_feature_table(case_ids: pd.Series, labels: pd.Series, features: pd.DataFrame, label_column: str) -> pd.DataFrame:
    frame = features.copy()
    frame.insert(0, "case_id", case_ids.reset_index(drop=True))
    frame.insert(1, label_column, labels.reset_index(drop=True))
    return frame


def select_features(
    features_path: Path,
    output_dir: Path,
    labels_path: Path | None = None,
    label_column: str = "label",
    variance_threshold: float = 0.0,
    correlation_threshold: float = 0.95,
    top_k: int = 20,
) -> SelectionArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_training_data(features_path, labels_path=labels_path, label_column=label_column)
    LOGGER.info(
        "Starting feature selection: raw_numeric_features=%s variance_threshold=%s correlation_threshold=%s top_k=%s",
        len(prepared.features.columns),
        variance_threshold,
        correlation_threshold,
        top_k,
    )

    imputer = SimpleImputer(strategy="median")
    imputed = pd.DataFrame(
        imputer.fit_transform(prepared.features),
        columns=prepared.features.columns,
        index=prepared.features.index,
    )

    variance_selector = VarianceThreshold(threshold=variance_threshold)
    variance_array = variance_selector.fit_transform(imputed)
    variance_columns = imputed.columns[variance_selector.get_support()].tolist()
    variance_frame = pd.DataFrame(variance_array, columns=variance_columns, index=imputed.index)

    corr_filter = CorrelationFilter(threshold=correlation_threshold)
    corr_frame = corr_filter.fit_transform(variance_frame)

    selection_mode = "correlation_only"
    selected_frame = corr_frame.copy()
    if top_k > 0 and len(corr_frame.columns) > top_k:
        selector = SelectKBest(score_func=f_classif, k=top_k)
        selected_array = selector.fit_transform(corr_frame, prepared.labels)
        selected_columns = corr_frame.columns[selector.get_support()].tolist()
        selected_frame = pd.DataFrame(selected_array, columns=selected_columns, index=corr_frame.index)
        selection_mode = "anova_top_k"

    cleaned_table = _build_feature_table(prepared.case_ids, prepared.labels, corr_frame, prepared.label_column)
    selected_table = _build_feature_table(prepared.case_ids, prepared.labels, selected_frame, prepared.label_column)

    summary = pd.DataFrame(
        [
            {"step": "raw_numeric", "feature_count": len(prepared.features.columns)},
            {"step": "after_variance", "feature_count": len(variance_frame.columns)},
            {"step": "after_correlation", "feature_count": len(corr_frame.columns)},
            {"step": "after_selection", "feature_count": len(selected_frame.columns)},
            {"step": "selection_mode", "feature_count": selection_mode},
        ]
    )

    cleaned_features_path = output_dir / "cleaned_features.csv"
    selected_features_path = output_dir / "selected_features.csv"
    summary_path = output_dir / "feature_selection_summary.csv"

    cleaned_table.to_csv(cleaned_features_path, index=False)
    selected_table.to_csv(selected_features_path, index=False)
    summary.to_csv(summary_path, index=False)
    LOGGER.info(
        "Feature selection summary: after_variance=%s after_correlation=%s after_selection=%s mode=%s",
        len(variance_frame.columns),
        len(corr_frame.columns),
        len(selected_frame.columns),
        selection_mode,
    )

    return SelectionArtifacts(
        cleaned_features=cleaned_table,
        selected_features=selected_table,
        summary=summary,
        cleaned_features_path=cleaned_features_path,
        selected_features_path=selected_features_path,
        summary_path=summary_path,
    )


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    base_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("corr", CorrelationFilter(threshold=0.95)),
    ]
    return {
        "logistic_regression": Pipeline(
            base_steps
            + [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=2000, random_state=random_state),
                ),
            ]
        ),
        "random_forest": Pipeline(
            base_steps
            + [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=random_state,
                        class_weight="balanced",
                    ),
                )
            ]
        ),
        "svm": Pipeline(
            base_steps
            + [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", probability=False, class_weight="balanced", random_state=random_state)),
            ]
        ),
    }


def train_and_evaluate(
    features_path: Path,
    output_dir: Path,
    labels_path: Path | None = None,
    label_column: str = "label",
    folds: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_training_data(features_path, labels_path=labels_path, label_column=label_column)
    prepared_frame = _build_feature_table(prepared.case_ids, prepared.labels, prepared.features, prepared.label_column)
    prepared_frame.to_csv(output_dir / "prepared_features.csv", index=False)
    LOGGER.info("Prepared feature table written to %s", output_dir / "prepared_features.csv")

    class_counts = prepared.labels.value_counts()
    min_class_count = int(class_counts.min())
    LOGGER.info("Training class distribution: %s", class_counts.to_dict())
    if min_class_count < 2:
        LOGGER.error("At least one class has fewer than 2 samples")
        raise ValueError("Each class must contain at least 2 samples for cross-validation.")
    effective_folds = min(folds, min_class_count)
    if effective_folds < 2:
        LOGGER.error("Unable to build a valid stratified cross-validation split")
        raise ValueError("Unable to create a valid stratified cross-validation split.")

    cv = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=random_state)
    LOGGER.info("Starting model evaluation with folds=%s random_state=%s", effective_folds, random_state)

    metrics_rows = []
    label_order = sorted(prepared.labels.unique())
    models = build_models(random_state=random_state)

    for model_name, pipeline in models.items():
        LOGGER.info("Evaluating model: %s", model_name)
        predictions = cross_val_predict(pipeline, prepared.features, prepared.labels, cv=cv)
        accuracy = accuracy_score(prepared.labels, predictions)
        f1_macro = f1_score(prepared.labels, predictions, average="macro")
        f1_weighted = f1_score(prepared.labels, predictions, average="weighted")
        matrix = confusion_matrix(prepared.labels, predictions, labels=label_order)

        confusion_path = output_dir / f"confusion_matrix_{model_name}.csv"
        pd.DataFrame(matrix, index=label_order, columns=label_order).to_csv(confusion_path)
        LOGGER.info(
            "Model %s finished: accuracy=%.4f f1_macro=%.4f f1_weighted=%.4f confusion_matrix=%s",
            model_name,
            accuracy,
            f1_macro,
            f1_weighted,
            confusion_path,
        )
        metrics_rows.append(
            {
                "model": model_name,
                "folds": effective_folds,
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "classes": json.dumps(label_order, ensure_ascii=True),
            }
        )

    metrics_frame = pd.DataFrame(metrics_rows).sort_values(by="f1_macro", ascending=False).reset_index(drop=True)
    if not metrics_frame.empty:
        best_model = metrics_frame.iloc[0]
        LOGGER.info(
            "Training summary: best_model=%s best_f1_macro=%.4f evaluated_models=%s",
            best_model["model"],
            float(best_model["f1_macro"]),
            len(metrics_frame),
        )
    return metrics_frame
