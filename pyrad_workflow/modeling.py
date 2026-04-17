from __future__ import annotations

import json
import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from classification import mean_roc_plot

LOGGER = logging.getLogger("pyrad_workflow")
ProgressCallback = Callable[[float, str], None]


NON_FEATURE_COLUMNS = {
    "case_id",
    "image_path",
    "mask_path",
    "label",
}

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "svm": "Support Vector Machine",
    "knn": "K-Nearest Neighbors",
    "naive_bayes": "Naive Bayes",
}

SELECTION_METHOD_LABELS = {
    "anova_top_k": "ANOVA Top-K",
    "mutual_info_top_k": "Mutual Information Top-K",
    "lasso": "L1 Logistic Selection",
    "correlation_only": "Correlation Only",
    "mutual_info_top_k_fallback_anova": "Mutual Information Top-K (Fallback to ANOVA)",
}

DEFAULT_SAFE_MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "svm",
    "naive_bayes",
]


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


@dataclass(frozen=True)
class PredictionArtifacts:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    predictions_path: Path
    metrics_path: Path | None
    confusion_matrix_path: Path | None
    confusion_matrix_image_path: Path | None
    roc_curve_path: Path | None
    output_dir: Path
    model_name: str


@dataclass(frozen=True)
class ShapArtifacts:
    summary_plot_path: Path | None
    waterfall_plot_path: Path | None
    importance_csv_path: Path | None
    top_feature_names: list[str]


def load_labels(labels_path: Path) -> pd.DataFrame:
    LOGGER.info("Loading labels file: %s", labels_path)
    labels = pd.read_csv(labels_path)
    required = {"case_id", "label"}
    missing = required - set(labels.columns)
    if missing:
        LOGGER.error("Labels file missing required columns: %s", ", ".join(sorted(missing)))
        raise ValueError(f"Labels file missing required columns: {', '.join(sorted(missing))}")
    labels["case_id"] = normalize_case_id_series(labels["case_id"])
    LOGGER.info(
        "Labels file loaded successfully: rows=%s classes=%s preview=%s",
        len(labels),
        labels["label"].astype(str).value_counts().to_dict(),
        labels.head(3).to_dict(orient="records"),
    )
    return labels


def emit_progress(progress_callback: ProgressCallback | None, percent: float, detail: str) -> None:
    if progress_callback is None:
        return
    progress_callback(max(0.0, min(100.0, float(percent))), detail)


def normalize_case_id_series(series: pd.Series) -> pd.Series:
    def _normalize(value) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip()
        if text == "":
            return ""
        if re.fullmatch(r"[+-]?\d+(?:\.0+)?", text):
            return str(int(float(text)))
        return text

    return series.map(_normalize).astype(str)


def available_models() -> list[dict[str, str]]:
    return [{"name": name, "label": label} for name, label in MODEL_DISPLAY_NAMES.items()]


def available_selection_methods() -> list[dict[str, str]]:
    return [{"name": name, "label": label} for name, label in SELECTION_METHOD_LABELS.items()]


def resolve_selection_method(selection_method: str | None) -> str:
    method = str(selection_method or "anova_top_k").strip()
    if method not in SELECTION_METHOD_LABELS:
        raise ValueError(f"Unsupported selection method: {method}")
    return method


def resolve_model_names(model_names: list[str] | tuple[str, ...] | str | None) -> list[str]:
    if model_names is None:
        return list(MODEL_DISPLAY_NAMES.keys())
    if isinstance(model_names, str):
        requested = [part.strip() for part in model_names.split(",") if part.strip()]
    else:
        requested = [str(part).strip() for part in model_names if str(part).strip()]
    if not requested:
        return list(MODEL_DISPLAY_NAMES.keys())
    invalid = sorted(set(requested) - set(MODEL_DISPLAY_NAMES))
    if invalid:
        raise ValueError(f"Unsupported models: {', '.join(invalid)}")
    return list(dict.fromkeys(requested))


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
    features["case_id"] = normalize_case_id_series(features["case_id"])
    LOGGER.info(
        "Training feature table loaded: rows=%s columns=%s preview_case_ids=%s",
        len(features),
        len(features.columns),
        features["case_id"].head(5).tolist(),
    )

    merged = features.copy()
    if labels_path is not None:
        LOGGER.info("Merging external labels into feature table")
        labels = load_labels(labels_path)
        merged = merged.drop(columns=["label"], errors="ignore").merge(labels, on="case_id", how="left")
        label_column = "label"
        LOGGER.info("Merged features with labels: rows=%s columns=%s", len(merged), len(merged.columns))

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
    # Drop columns that became entirely missing after numeric coercion, such as
    # diagnostic hash/config fields from example feature tables.
    x = x.dropna(axis=1, how="all")
    feature_columns = x.columns.tolist()
    if not feature_columns:
        LOGGER.error("No usable numeric feature columns remain in %s", features_path)
        raise ValueError("No usable numeric feature columns remain after numeric conversion.")
    y = merged[label_column].astype(str)
    LOGGER.info(
        "Prepared training data: cases=%s numeric_features=%s classes=%s",
        len(merged),
        len(feature_columns),
        sorted(y.unique()),
    )
    LOGGER.info("Prepared numeric feature preview: %s", feature_columns[:10])
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
    selection_method: str = "anova_top_k",
    progress_callback: ProgressCallback | None = None,
) -> SelectionArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    emit_progress(progress_callback, 5, "Loading feature table")
    prepared = prepare_training_data(features_path, labels_path=labels_path, label_column=label_column)
    selection_method = resolve_selection_method(selection_method)
    LOGGER.info(
        "Starting feature selection: raw_numeric_features=%s variance_threshold=%s correlation_threshold=%s top_k=%s method=%s",
        len(prepared.features.columns),
        variance_threshold,
        correlation_threshold,
        top_k,
        selection_method,
    )

    emit_progress(progress_callback, 20, "Imputing missing values")
    imputer = SimpleImputer(strategy="median")
    imputed = pd.DataFrame(
        imputer.fit_transform(prepared.features),
        columns=prepared.features.columns,
        index=prepared.features.index,
    )

    emit_progress(progress_callback, 40, "Applying variance filter")
    variance_selector = VarianceThreshold(threshold=variance_threshold)
    variance_array = variance_selector.fit_transform(imputed)
    variance_columns = imputed.columns[variance_selector.get_support()].tolist()
    variance_frame = pd.DataFrame(variance_array, columns=variance_columns, index=imputed.index)

    emit_progress(progress_callback, 62, "Removing correlated features")
    corr_filter = CorrelationFilter(threshold=correlation_threshold)
    corr_frame = corr_filter.fit_transform(variance_frame)

    selection_mode = selection_method
    selected_frame = corr_frame.copy()
    if selection_method == "anova_top_k" and top_k > 0 and len(corr_frame.columns) > top_k:
        emit_progress(progress_callback, 82, f"Selecting top {top_k} features with ANOVA")
        selector = SelectKBest(score_func=f_classif, k=top_k)
        selected_array = selector.fit_transform(corr_frame, prepared.labels)
        selected_columns = corr_frame.columns[selector.get_support()].tolist()
        selected_frame = pd.DataFrame(selected_array, columns=selected_columns, index=corr_frame.index)
    elif selection_method == "mutual_info_top_k" and top_k > 0 and len(corr_frame.columns) > top_k:
        emit_progress(progress_callback, 82, f"Selecting top {top_k} features with mutual information")
        try:
            selector = SelectKBest(score_func=mutual_info_classif, k=top_k)
            selected_array = selector.fit_transform(corr_frame, prepared.labels)
            selected_columns = corr_frame.columns[selector.get_support()].tolist()
            selected_frame = pd.DataFrame(selected_array, columns=selected_columns, index=corr_frame.index)
        except Exception as exc:
            LOGGER.warning("Mutual information selection failed, falling back to ANOVA: %s", exc)
            selector = SelectKBest(score_func=f_classif, k=top_k)
            selected_array = selector.fit_transform(corr_frame, prepared.labels)
            selected_columns = corr_frame.columns[selector.get_support()].tolist()
            selected_frame = pd.DataFrame(selected_array, columns=selected_columns, index=corr_frame.index)
            selection_mode = "mutual_info_top_k_fallback_anova"
    elif selection_method == "lasso":
        emit_progress(progress_callback, 82, "Selecting features with L1 logistic model")
        classes = sorted(prepared.labels.unique().tolist())
        solver = "liblinear" if len(classes) <= 2 else "saga"
        estimator = LogisticRegression(
            penalty="l1",
            solver=solver,
            C=0.1,
            max_iter=5000,
            random_state=42,
            class_weight="balanced",
        )
        selector = SelectFromModel(estimator)
        selector.fit(corr_frame, prepared.labels)
        support = selector.get_support()
        selected_columns = corr_frame.columns[support].tolist()
        if not selected_columns:
            selected_columns = corr_frame.columns.tolist()
        if top_k > 0 and len(selected_columns) > top_k:
            fitted_estimator = selector.estimator_
            coef = np.abs(np.asarray(fitted_estimator.coef_))
            if coef.ndim > 1:
                importance = coef.max(axis=0)
            else:
                importance = coef
            ranking = pd.Series(importance, index=corr_frame.columns).sort_values(ascending=False)
            selected_columns = [column for column in ranking.index if column in selected_columns][:top_k]
        selected_frame = corr_frame.loc[:, selected_columns].copy()
    else:
        emit_progress(progress_callback, 82, "Keeping correlation-filtered features")

    cleaned_table = _build_feature_table(prepared.case_ids, prepared.labels, corr_frame, prepared.label_column)
    selected_table = _build_feature_table(prepared.case_ids, prepared.labels, selected_frame, prepared.label_column)

    summary = pd.DataFrame(
        [
            {"step": "raw_numeric", "feature_count": len(prepared.features.columns)},
            {"step": "after_variance", "feature_count": len(variance_frame.columns)},
            {"step": "after_correlation", "feature_count": len(corr_frame.columns)},
            {"step": "after_selection", "feature_count": len(selected_frame.columns)},
            {"step": "selection_method", "feature_count": selection_mode},
            {"step": "selection_method_label", "feature_count": SELECTION_METHOD_LABELS[selection_mode]},
        ]
    )

    cleaned_features_path = output_dir / "cleaned_features.csv"
    selected_features_path = output_dir / "selected_features.csv"
    summary_path = output_dir / "feature_selection_summary.csv"

    emit_progress(progress_callback, 94, "Saving selection artifacts")
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
    LOGGER.info("Selected feature preview: %s", selected_frame.columns[: min(20, len(selected_frame.columns))].tolist())
    emit_progress(progress_callback, 100, "Feature selection complete")

    return SelectionArtifacts(
        cleaned_features=cleaned_table,
        selected_features=selected_table,
        summary=summary,
        cleaned_features_path=cleaned_features_path,
        selected_features_path=selected_features_path,
        summary_path=summary_path,
    )


def build_models(random_state: int = 42, selected_models: list[str] | tuple[str, ...] | str | None = None) -> dict[str, Pipeline]:
    base_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("corr", CorrelationFilter(threshold=0.95)),
    ]
    models = {
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
                ("model", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state)),
            ]
        ),
        "knn": Pipeline(
            base_steps
            + [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=3)),
            ]
        ),
        "naive_bayes": Pipeline(
            base_steps
            + [
                ("model", GaussianNB()),
            ]
        ),
    }
    names = resolve_model_names(selected_models)
    return {name: models[name] for name in names}


def save_model_bundle(
    output_dir: Path,
    model_name: str,
    pipeline: Pipeline,
    feature_columns: list[str],
    label_column: str,
    classes: list[str],
) -> Path:
    bundle = {
        "model_name": model_name,
        "model_label": MODEL_DISPLAY_NAMES.get(model_name, model_name),
        "pipeline": pipeline,
        "feature_columns": feature_columns,
        "label_column": label_column,
        "classes": classes,
    }
    model_path = output_dir / f"trained_model_{model_name}.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(bundle, handle)
    return model_path


def load_model_bundle(model_path: Path) -> dict[str, object]:
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def transformed_feature_names(pipeline: Pipeline, original_columns: list[str]) -> list[str]:
    names = list(original_columns)
    for step_name, step in pipeline.steps[:-1]:
        _ = step_name
        if hasattr(step, "selected_columns_"):
            names = list(step.selected_columns_)
    return names


def transform_feature_frame(pipeline: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    preprocess = pipeline[:-1]
    transformed = preprocess.transform(features)
    names = transformed_feature_names(pipeline, features.columns.tolist())
    return pd.DataFrame(transformed, columns=names, index=features.index)


def _plot_confusion_matrix_image(matrix_frame: pd.DataFrame, output_path: Path, title: str) -> Path:
    figure, axis = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(matrix_frame, annot=True, fmt="g", cmap="Blues", cbar=False, ax=axis, linewidths=0.5)
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _plot_roc_curve_image(
    truth: pd.Series,
    score_frame: pd.DataFrame,
    output_path: Path,
    title: str,
    tprs=None,
    aucs=None,
    mean_fpr=None,
) -> Path | None:
    classes = sorted(truth.astype(str).dropna().unique().tolist())
    if len(classes) < 2:
        return None

    if len(classes) == 2:
        positive_label = classes[-1]
        score_column = f"prob_{positive_label}"
        if score_column not in score_frame.columns:
            return None
        if mean_fpr is None:
            mean_fpr = np.linspace(0, 1, 100)
        if tprs is None or aucs is None:
            binary_truth = (truth.astype(str) == positive_label).astype(int)
            fpr, tpr, _ = roc_curve(binary_truth, score_frame[score_column])
            interpolated = np.interp(mean_fpr, fpr, tpr)
            interpolated[0] = 0.0
            tprs = [interpolated]
            aucs = [auc(fpr, tpr)]

        figure, axis = plt.subplots(figsize=(6.5, 5.2))
        mean_roc_plot(axis, tprs, aucs, mean_fpr, title, output_path=output_path)
        plt.close(figure)
        return output_path

    figure, axis = plt.subplots(figsize=(6.5, 5.2))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#90a4b8", linewidth=1.2, label="Baseline")
    truth_matrix = label_binarize(truth.astype(str), classes=classes)
    drew_curve = False
    colors = ["#1f5f95", "#2b8c6b", "#b5473a", "#8a5bd8", "#dd8b2a"]
    for index, class_name in enumerate(classes):
        score_column = f"prob_{class_name}"
        if score_column not in score_frame.columns:
            continue
        fpr, tpr, _ = roc_curve(truth_matrix[:, index], score_frame[score_column])
        roc_value = auc(fpr, tpr)
        axis.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{class_name} (AUC={roc_value:.3f})",
            color=colors[index % len(colors)],
        )
        drew_curve = True
    if not drew_curve:
        plt.close(figure)
        return None

    axis.set_title(title)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend(loc="lower right", frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def run_shap_analysis(
    fitted_pipeline: Pipeline,
    feature_frame: pd.DataFrame,
    output_dir: Path,
    title_prefix: str,
    max_background: int = 30,
    max_explain: int = 80,
) -> ShapArtifacts:
    try:
        import shap
    except Exception as exc:
        LOGGER.warning("SHAP is unavailable, skipping explanation analysis: %s", exc)
        return ShapArtifacts(None, None, None, [])

    if feature_frame.empty:
        return ShapArtifacts(None, None, None, [])

    transformed_frame = transform_feature_frame(fitted_pipeline, feature_frame)
    estimator = fitted_pipeline.steps[-1][1]
    background = transformed_frame.iloc[: min(max_background, len(transformed_frame))].copy()
    explain_frame = transformed_frame.iloc[: min(max_explain, len(transformed_frame))].copy()
    required_max_evals = max(2 * transformed_frame.shape[1] + 1, 500)

    try:
        if isinstance(estimator, RandomForestClassifier):
            explainer = shap.TreeExplainer(estimator)
            explanation = explainer(explain_frame)
        elif isinstance(estimator, LogisticRegression):
            explainer = shap.LinearExplainer(estimator, background, feature_perturbation="interventional")
            explanation = explainer(explain_frame)
        else:
            predict_fn = estimator.predict_proba if hasattr(estimator, "predict_proba") else estimator.predict
            explainer = shap.Explainer(predict_fn, background)
            explanation = explainer(explain_frame, max_evals=required_max_evals)
    except Exception as exc:
        LOGGER.warning("Unable to compute SHAP explanation, skipping: %s", exc)
        return ShapArtifacts(None, None, None, [])

    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)
    if values.ndim == 3:
        values = values[:, :, -1]
        if base_values.ndim > 1:
            base_values = base_values[:, -1]

    mean_abs = np.mean(np.abs(values), axis=0)
    ranking = (
        pd.DataFrame({"feature": explain_frame.columns, "mean_abs_shap": mean_abs})
        .sort_values(by="mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    top_feature_names = ranking.head(15)["feature"].tolist()

    importance_csv_path = output_dir / f"{title_prefix}_shap_importance.csv"
    ranking.to_csv(importance_csv_path, index=False)

    summary_plot_path = output_dir / f"{title_prefix}_shap_summary.png"
    figure, axis = plt.subplots(figsize=(8, max(4.5, min(10, len(top_feature_names) * 0.45 + 1.5))))
    top_rows = ranking.head(15).iloc[::-1]
    axis.barh(top_rows["feature"], top_rows["mean_abs_shap"], color="#1f5f95")
    axis.set_xlabel("Mean |SHAP value|")
    axis.set_title(f"{title_prefix.replace('_', ' ').title()} SHAP Importance")
    figure.tight_layout()
    figure.savefig(summary_plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    waterfall_plot_path = output_dir / f"{title_prefix}_shap_waterfall.png"
    try:
        single_base = base_values[0] if np.ndim(base_values) else base_values
        single_explanation = shap.Explanation(
            values=np.asarray(values[0], dtype=float),
            base_values=float(np.asarray(single_base).reshape(-1)[0]),
            data=np.asarray(explain_frame.iloc[0].to_numpy(), dtype=float),
            feature_names=[str(name) for name in explain_frame.columns],
        )
        plt.figure(figsize=(8, 6))
        shap.plots.waterfall(single_explanation, max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(waterfall_plot_path, dpi=180, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        LOGGER.warning("Unable to render SHAP waterfall plot: %s", exc)
        waterfall_plot_path = None

    return ShapArtifacts(summary_plot_path, waterfall_plot_path, importance_csv_path, top_feature_names)


def collect_cv_roc_data(
    pipeline: Pipeline,
    features: pd.DataFrame,
    labels: pd.Series,
    cv: StratifiedKFold,
):
    classes = sorted(labels.astype(str).unique().tolist())
    if len(classes) != 2:
        return None, None, None

    positive_label = classes[-1]
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for train_index, test_index in cv.split(features, labels):
        fitted = clone(pipeline)
        x_train = features.iloc[train_index]
        x_test = features.iloc[test_index]
        y_train = labels.iloc[train_index]
        y_test = labels.iloc[test_index]
        fitted.fit(x_train, y_train)
        score_frame = compute_prediction_score_frame(fitted, x_test)
        score_column = f"prob_{positive_label}"
        if score_column not in score_frame.columns:
            continue
        binary_truth = (y_test.astype(str) == positive_label).astype(int)
        fpr, tpr, _ = roc_curve(binary_truth, score_frame[score_column])
        interpolated = np.interp(mean_fpr, fpr, tpr)
        interpolated[0] = 0.0
        tprs.append(interpolated)
        aucs.append(auc(fpr, tpr))

    if not tprs:
        return None, None, None
    return tprs, aucs, mean_fpr


def prepare_prediction_data(
    features_path: Path,
    feature_columns: list[str],
    labels_path: Path | None = None,
    label_column: str = "label",
):
    features = pd.read_csv(features_path)
    if "case_id" not in features.columns:
        raise ValueError("Features file must contain case_id column.")
    features["case_id"] = normalize_case_id_series(features["case_id"])

    merged = features.copy()
    if labels_path is not None:
        labels = load_labels(labels_path)
        merged = merged.drop(columns=["label"], errors="ignore").merge(labels, on="case_id", how="left")
        label_column = "label"

    numeric_frame = merged.copy()
    for column in list(numeric_frame.columns):
        if column in NON_FEATURE_COLUMNS:
            continue
        numeric_frame[column] = pd.to_numeric(numeric_frame[column], errors="coerce")

    x = numeric_frame.reindex(columns=feature_columns)
    truth = None
    if label_column in merged.columns:
        label_series = merged[label_column].replace("", np.nan)
        if not label_series.isna().all():
            truth = label_series.astype(str)
    return merged["case_id"].astype(str), x, truth, label_column


def compute_prediction_confidence(pipeline: Pipeline, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    model = pipeline
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        classes = [str(item) for item in getattr(model, "classes_", [])]
        probability_frame = pd.DataFrame(probabilities, columns=[f"prob_{label}" for label in classes])
        confidence = probability_frame.max(axis=1)
        return probability_frame, confidence

    if hasattr(model, "decision_function"):
        decision = model.decision_function(features)
        decision_array = np.asarray(decision)
        if decision_array.ndim == 1:
            confidence = pd.Series(1.0 / (1.0 + np.exp(-np.abs(decision_array))), name="confidence")
            probability_frame = pd.DataFrame({"decision_score": decision_array})
        else:
            shifted = decision_array - decision_array.max(axis=1, keepdims=True)
            exp_values = np.exp(shifted)
            probabilities = exp_values / exp_values.sum(axis=1, keepdims=True)
            classes = [str(item) for item in getattr(model, "classes_", range(probabilities.shape[1]))]
            probability_frame = pd.DataFrame(probabilities, columns=[f"score_{label}" for label in classes])
            confidence = probability_frame.max(axis=1)
        return probability_frame, confidence

    return pd.DataFrame(), pd.Series([pd.NA] * len(features), dtype="object", name="confidence")


def compute_prediction_score_frame(pipeline: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    model = pipeline
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        classes = [str(item) for item in getattr(model, "classes_", [])]
        return pd.DataFrame(probabilities, columns=[f"prob_{label}" for label in classes])

    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(features))
        if decision.ndim == 1:
            probabilities = 1.0 / (1.0 + np.exp(-decision))
            classes = [str(item) for item in getattr(model, "classes_", [])]
            if len(classes) == 2:
                return pd.DataFrame(
                    {
                        f"prob_{classes[0]}": 1.0 - probabilities,
                        f"prob_{classes[1]}": probabilities,
                    }
                )
            return pd.DataFrame({"decision_score": probabilities})

        shifted = decision - decision.max(axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        probabilities = exp_values / exp_values.sum(axis=1, keepdims=True)
        classes = [str(item) for item in getattr(model, "classes_", range(probabilities.shape[1]))]
        return pd.DataFrame(probabilities, columns=[f"prob_{label}" for label in classes])

    return pd.DataFrame()


def evaluate_predictions(truth: pd.Series, predictions: np.ndarray | pd.Series) -> tuple[dict[str, object], pd.DataFrame]:
    truth_series = truth.reset_index(drop=True)
    pred_series = pd.Series(predictions, dtype="object").reset_index(drop=True)
    valid_mask = truth_series.notna()
    truth_series = truth_series.loc[valid_mask].astype(str).reset_index(drop=True)
    pred_series = pred_series.loc[valid_mask].astype(str).reset_index(drop=True)
    labels = sorted(pd.concat([truth_series, pred_series], ignore_index=True).dropna().unique().tolist())
    matrix = confusion_matrix(truth_series, pred_series, labels=labels)
    metrics = {
        "cases": len(truth_series),
        "accuracy": accuracy_score(truth_series, pred_series),
        "f1_macro": f1_score(truth_series, pred_series, average="macro"),
        "f1_weighted": f1_score(truth_series, pred_series, average="weighted"),
        "classes": json.dumps(labels, ensure_ascii=True),
    }
    matrix_frame = pd.DataFrame(matrix, index=labels, columns=labels)
    return metrics, matrix_frame


def compute_auc_from_score_frame(truth: pd.Series, score_frame: pd.DataFrame) -> float | None:
    if score_frame.empty:
        return None

    truth_series = truth.reset_index(drop=True)
    valid_mask = truth_series.notna()
    truth_series = truth_series.loc[valid_mask].astype(str).reset_index(drop=True)
    score_frame = score_frame.loc[valid_mask].reset_index(drop=True)
    classes = sorted(truth_series.dropna().unique().tolist())
    probability_columns = [f"prob_{label}" for label in classes]
    if not all(column in score_frame.columns for column in probability_columns):
        return None

    try:
        if len(classes) == 2:
            positive_label = classes[-1]
            return float(roc_auc_score((truth_series == positive_label).astype(int), score_frame[f"prob_{positive_label}"]))

        y_true = label_binarize(truth_series, classes=classes)
        y_score = score_frame.loc[:, probability_columns].to_numpy()
        return float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
    except Exception as exc:
        LOGGER.warning("Unable to compute ROC AUC from score frame: %s", exc)
        return None


def train_and_evaluate(
    features_path: Path,
    output_dir: Path,
    labels_path: Path | None = None,
    label_column: str = "label",
    folds: int = 5,
    random_state: int = 42,
    model_names: list[str] | tuple[str, ...] | str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    emit_progress(progress_callback, 5, "Loading training data")
    prepared = prepare_training_data(features_path, labels_path=labels_path, label_column=label_column)
    prepared_frame = _build_feature_table(prepared.case_ids, prepared.labels, prepared.features, prepared.label_column)
    prepared_frame.to_csv(output_dir / "prepared_features.csv", index=False)
    LOGGER.info("Prepared feature table written to %s", output_dir / "prepared_features.csv")
    emit_progress(progress_callback, 12, "Prepared feature table saved")

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
    models = build_models(random_state=random_state, selected_models=model_names)
    total_models = max(len(models), 1)
    registry_rows: list[dict[str, object]] = []
    failed_models: list[dict[str, str]] = []

    for index, (model_name, pipeline) in enumerate(models.items(), start=1):
        start_progress = 12 + ((index - 1) / total_models) * 60
        emit_progress(progress_callback, start_progress, f"Training {model_name}")
        LOGGER.info("Evaluating model: %s", model_name)
        try:
            predictions = cross_val_predict(pipeline, prepared.features, prepared.labels, cv=cv)
            probability_predictions = cross_val_predict(pipeline, prepared.features, prepared.labels, cv=cv, method="predict_proba")
            accuracy = accuracy_score(prepared.labels, predictions)
            f1_macro = f1_score(prepared.labels, predictions, average="macro")
            f1_weighted = f1_score(prepared.labels, predictions, average="weighted")
            matrix = confusion_matrix(prepared.labels, predictions, labels=label_order)
            cv_probability_frame = pd.DataFrame(
                probability_predictions,
                columns=[f"prob_{label}" for label in label_order],
                index=prepared.features.index,
            )
            roc_auc_value = compute_auc_from_score_frame(prepared.labels.reset_index(drop=True), cv_probability_frame.reset_index(drop=True))
            cv_tprs, cv_aucs, cv_mean_fpr = collect_cv_roc_data(pipeline, prepared.features, prepared.labels, cv)
            fitted_pipeline = pipeline.fit(prepared.features, prepared.labels)
            probability_frame = compute_prediction_score_frame(fitted_pipeline, prepared.features)

            confusion_path = output_dir / f"confusion_matrix_{model_name}.csv"
            confusion_image_path = output_dir / f"confusion_matrix_{model_name}.png"
            roc_curve_path = output_dir / f"roc_curve_{model_name}.png"
            pd.DataFrame(matrix, index=label_order, columns=label_order).to_csv(confusion_path)
            matrix_frame = pd.DataFrame(matrix, index=label_order, columns=label_order)
            _plot_confusion_matrix_image(matrix_frame, confusion_image_path, f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)} Confusion Matrix")
            roc_image = _plot_roc_curve_image(
                prepared.labels.reset_index(drop=True),
                probability_frame.reset_index(drop=True),
                roc_curve_path,
                f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)} ROC Curve",
                tprs=cv_tprs,
                aucs=cv_aucs,
                mean_fpr=cv_mean_fpr,
            )
            model_path = save_model_bundle(
                output_dir,
                model_name,
                fitted_pipeline,
                prepared.features.columns.tolist(),
                prepared.label_column,
                label_order,
            )
            LOGGER.info(
                "Model %s finished: accuracy=%.4f f1_macro=%.4f f1_weighted=%.4f confusion_matrix=%s model=%s",
                model_name,
                accuracy,
                f1_macro,
                f1_weighted,
                confusion_path,
                model_path,
            )
            metrics_rows.append(
                {
                    "model": model_name,
                    "model_label": MODEL_DISPLAY_NAMES.get(model_name, model_name),
                    "folds": effective_folds,
                    "accuracy": accuracy,
                    "roc_auc": roc_auc_value if roc_auc_value is not None else "",
                    "f1_macro": f1_macro,
                    "f1_weighted": f1_weighted,
                    "classes": json.dumps(label_order, ensure_ascii=True),
                    "model_path": str(model_path),
                }
            )
            registry_rows.append(
                {
                    "model": model_name,
                    "model_label": MODEL_DISPLAY_NAMES.get(model_name, model_name),
                    "model_path": str(model_path),
                    "confusion_matrix_path": str(confusion_path),
                    "confusion_matrix_image_path": str(confusion_image_path),
                    "roc_curve_path": str(roc_image) if roc_image is not None else "",
                    "status": "completed",
                    "error": "",
                }
            )
        except Exception as exc:
            LOGGER.exception("Model %s failed during training/evaluation: %s", model_name, exc)
            failed_models.append({"model": model_name, "error": str(exc)})
            registry_rows.append(
                {
                    "model": model_name,
                    "model_label": MODEL_DISPLAY_NAMES.get(model_name, model_name),
                    "model_path": "",
                    "confusion_matrix_path": "",
                    "status": "failed",
                    "error": str(exc),
                }
            )
        end_progress = 12 + (index / total_models) * 60
        emit_progress(progress_callback, end_progress, f"Finished {model_name}")

    if not metrics_rows:
        raise RuntimeError(
            "All selected models failed during training: "
            + "; ".join(f"{item['model']}: {item['error']}" for item in failed_models)
        )

    metrics_frame = pd.DataFrame(metrics_rows).sort_values(by="f1_macro", ascending=False).reset_index(drop=True)
    registry_path = output_dir / "trained_models_manifest.csv"
    pd.DataFrame(registry_rows).to_csv(registry_path, index=False)
    if failed_models:
        pd.DataFrame(failed_models).to_csv(output_dir / "failed_models.csv", index=False)
        LOGGER.warning("Some models failed and were skipped: %s", failed_models)
    if not metrics_frame.empty:
        best_model = metrics_frame.iloc[0]
        LOGGER.info(
            "Training summary: best_model=%s best_f1_macro=%.4f evaluated_models=%s",
            best_model["model"],
            float(best_model["f1_macro"]),
            len(metrics_frame),
        )
        LOGGER.info("Training metrics preview: %s", metrics_frame.to_dict(orient="records"))
        best_model_path = Path(str(best_model["model_path"]))
        best_bundle = load_model_bundle(best_model_path)
        run_shap_analysis(
            best_bundle["pipeline"],
            prepared.features,
            output_dir,
            title_prefix=f"best_model_{best_model['model']}",
        )
    emit_progress(progress_callback, 100, "Model training complete")
    return metrics_frame


def predict_and_evaluate(
    features_path: Path,
    model_path: Path,
    output_dir: Path,
    labels_path: Path | None = None,
    label_column: str = "label",
    progress_callback: ProgressCallback | None = None,
) -> PredictionArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    emit_progress(progress_callback, 5, "Loading trained model")
    bundle = load_model_bundle(model_path)
    emit_progress(progress_callback, 20, "Preparing prediction features")
    case_ids, x, truth, effective_label_column = prepare_prediction_data(
        features_path,
        feature_columns=[str(column) for column in bundle["feature_columns"]],
        labels_path=labels_path,
        label_column=label_column,
    )

    pipeline = bundle["pipeline"]
    model_name = str(bundle["model_name"])
    emit_progress(progress_callback, 60, f"Running predictions with {model_name}")
    predictions = pipeline.predict(x)
    probability_frame, confidence = compute_prediction_confidence(pipeline, x)

    prediction_rows = pd.DataFrame(
        {
            "case_id": case_ids.reset_index(drop=True),
            "predicted_label": pd.Series(predictions).astype(str),
            "confidence": confidence.reset_index(drop=True),
        }
    )
    if not probability_frame.empty:
        prediction_rows = pd.concat([prediction_rows, probability_frame.reset_index(drop=True)], axis=1)
    if truth is not None:
        truth_series = truth.reset_index(drop=True).astype(str)
        prediction_rows.insert(1, effective_label_column, truth_series)
        prediction_rows["is_correct"] = prediction_rows[effective_label_column] == prediction_rows["predicted_label"]

    predictions_path = output_dir / "predictions.csv"
    prediction_rows.to_csv(predictions_path, index=False)

    metrics_path: Path | None = None
    confusion_matrix_path: Path | None = None
    confusion_matrix_image_path: Path | None = None
    roc_curve_path: Path | None = None
    metrics_frame = pd.DataFrame()

    if truth is not None:
        emit_progress(progress_callback, 82, "Calculating evaluation metrics")
        metrics, matrix_frame = evaluate_predictions(truth, predictions)
        roc_auc_value = compute_auc_from_score_frame(truth.reset_index(drop=True), probability_frame.reset_index(drop=True))
        metrics_frame = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "model_label": bundle.get("model_label", model_name),
                    "roc_auc": roc_auc_value if roc_auc_value is not None else "",
                    **metrics,
                }
            ]
        )
        metrics_path = output_dir / "prediction_metrics.csv"
        confusion_matrix_path = output_dir / f"prediction_confusion_matrix_{model_name}.csv"
        confusion_matrix_image_path = output_dir / f"prediction_confusion_matrix_{model_name}.png"
        roc_curve_path = output_dir / f"prediction_roc_curve_{model_name}.png"
        metrics_frame.to_csv(metrics_path, index=False)
        matrix_frame.to_csv(confusion_matrix_path)
        _plot_confusion_matrix_image(matrix_frame, confusion_matrix_image_path, f"{bundle.get('model_label', model_name)} Prediction Confusion Matrix")
        roc_curve_path = _plot_roc_curve_image(
            truth.reset_index(drop=True),
            probability_frame.reset_index(drop=True),
            roc_curve_path,
            f"{bundle.get('model_label', model_name)} Prediction ROC Curve",
        )

    run_shap_analysis(
        pipeline,
        feature_frame=x,
        output_dir=output_dir,
        title_prefix=f"prediction_{model_name}",
    )

    emit_progress(progress_callback, 100, "Prediction complete")
    return PredictionArtifacts(
        predictions=prediction_rows,
        metrics=metrics_frame,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        confusion_matrix_path=confusion_matrix_path,
        confusion_matrix_image_path=confusion_matrix_image_path,
        roc_curve_path=roc_curve_path,
        output_dir=output_dir,
        model_name=model_name,
    )
