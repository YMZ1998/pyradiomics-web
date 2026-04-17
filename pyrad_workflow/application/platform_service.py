from __future__ import annotations

import hashlib
import json
import logging
import re
from concurrent.futures import Executor
from pathlib import Path
from typing import Any

import pandas as pd
from classification import available_models, available_selection_methods, build_test_data_examples
from ..modeling import DEFAULT_SAFE_MODEL_NAMES

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
from ..domain.workflows import (
    run_extraction_workflow,
    run_feature_selection_workflow,
    run_full_pipeline_workflow,
    run_prediction_workflow,
    run_training_workflow,
    run_validation_workflow,
)
from ..infrastructure.filesystem import ensure_output_dir, make_download_listing, resolve_user_path
from ..infrastructure.jobs import InMemoryJobStore, JobRecord
from ..infrastructure.settings import AppSettings

WORKFLOW_NAMES = {"validate", "extract", "select", "train", "predict", "full"}
LOGGER = logging.getLogger("pyrad_workflow")
WORKFLOW_CACHE_VERSION = "2026-04-17-shap-images-2"


class ClinicalRadiomicsPlatformService:
    def __init__(self, settings: AppSettings, job_store: InMemoryJobStore, executor: Executor) -> None:
        self.settings = settings
        self.job_store = job_store
        self.executor = executor

    def default_paths(self, task_name: str | None = None) -> dict[str, Any]:
        task_name = self._normalize_task_name(task_name)
        examples = build_test_data_examples(self.settings.workspace)
        validation_output_dir = self._default_run_dir("validation", task_name=task_name)
        extraction_output_dir = self._default_run_dir("features", task_name=task_name)
        selection_output_dir = self._default_run_dir("selection", task_name=task_name)
        training_output_dir = self._default_run_dir("model", task_name=task_name)
        prediction_output_dir = self._default_run_dir("prediction", task_name=task_name)
        pipeline_output_dir = self._default_run_dir("pipeline", task_name=task_name)
        task_root = str(self._task_root(task_name))
        extraction_features_path = str((Path(extraction_output_dir) / "features.csv").resolve())
        selection_features_path = str((Path(selection_output_dir) / "selected_features.csv").resolve())
        prediction_features_path = selection_features_path
        latest_model = self._latest_trained_model_path(training_output_dir)
        return {
            "task_name": task_name,
            "task_output_root": task_root,
            "manifest": str(examples.manifest),
            "params": str(examples.params),
            "labels": str(examples.labels),
            "label_column": "label",
            "label_value": DEFAULT_MASK_LABEL,
            "tolerance": DEFAULT_GEOMETRY_TOLERANCE,
            "variance_threshold": DEFAULT_VARIANCE_THRESHOLD,
            "correlation_threshold": DEFAULT_CORRELATION_THRESHOLD,
            "top_k": DEFAULT_TOP_K,
            "selection_method": DEFAULT_SELECTION_METHOD,
            "folds": DEFAULT_FOLDS,
            "random_state": DEFAULT_RANDOM_STATE,
            "validation_output_dir": validation_output_dir,
            "extraction_output_dir": extraction_output_dir,
            "selection_output_dir": selection_output_dir,
            "training_output_dir": training_output_dir,
            "prediction_output_dir": prediction_output_dir,
            "pipeline_output_dir": pipeline_output_dir,
            "extraction_features_path": extraction_features_path,
            "selection_features_path": selection_features_path,
            "prediction_features_path": prediction_features_path,
            "trained_model_path": latest_model,
            "available_models": available_models(),
            "available_selection_methods": available_selection_methods(),
            "default_train_models": DEFAULT_SAFE_MODEL_NAMES,
            "example_features_path": str(examples.features),
        }

    def inspect_data(self, payload: dict[str, Any]) -> dict[str, Any]:
        manifest_path = self._resolve_required_path(payload, "manifest")
        labels_path = self._resolve_required_path(payload, "labels", required=False)
        label_column = str(payload.get("label_column", "label"))

        manifest = pd.read_csv(manifest_path)
        LOGGER.info(
            "Inspect data: manifest=%s rows=%s columns=%s labels=%s label_column=%s",
            manifest_path,
            len(manifest),
            len(manifest.columns),
            labels_path,
            label_column,
        )
        preview_rows = manifest.head(10).fillna("").to_dict(orient="records")
        if labels_path is not None and labels_path.exists():
            label_rows = self._summarize_labels(pd.read_csv(labels_path), "label")
        else:
            label_rows = self._summarize_labels(manifest, label_column)

        return {
            "ok": True,
            "cards": [
                self._metric_card("Cases", len(manifest), "brand", "Rows in manifest"),
                self._metric_card("Columns", len(manifest.columns), "neutral", "Manifest schema"),
                self._metric_card("Label Groups", len(label_rows), "success", "Non-empty labels"),
            ],
            "columns": list(manifest.columns),
            "rows": preview_rows,
            "label_groups": label_rows,
        }

    def inspect_features(self, payload: dict[str, Any]) -> dict[str, Any]:
        features_path = self._resolve_required_path(payload, "features")
        features = pd.read_csv(features_path)
        meta_columns = {"case_id", "image_path", "mask_path", "label"}
        feature_columns = [column for column in features.columns if column not in meta_columns]
        LOGGER.info(
            "Inspect features: features=%s rows=%s feature_columns=%s",
            features_path,
            len(features),
            len(feature_columns),
        )

        groups: dict[str, list[str]] = {}
        for column in feature_columns:
            group_name = self._group_feature_name(column)
            groups.setdefault(group_name, []).append(column)

        group_rows = [
            {
                "group": group_name,
                "count": len(names),
                "examples": ", ".join(names[:3]),
                "features": names,
            }
            for group_name, names in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
        ]
        feature_rows = [{"feature": name, "group": self._group_feature_name(name)} for name in feature_columns[:80]]
        return {
            "ok": True,
            "cards": [
                self._metric_card("Rows", len(features), "brand", "Cases in feature table"),
                self._metric_card("Feature Columns", len(feature_columns), "success", "Usable features"),
                self._metric_card("Groups", len(group_rows), "neutral", "Grouped by radiomics prefix"),
            ],
            "group_rows": group_rows,
            "feature_rows": feature_rows,
        }

    def inspect_models(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        requested_task = None if payload is None else payload.get("task_name")
        root = self._task_root(self._normalize_task_name(requested_task))
        if payload and payload.get("root"):
            resolved = resolve_user_path(self.settings.workspace, str(payload.get("root")))
            if resolved is not None:
                root = resolved
        models = self._discover_trained_models(root)
        LOGGER.info("Inspect models: root=%s count=%s", root, len(models))
        return {"ok": True, "models": models}

    def run_workflow_sync(self, workflow: str, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize_payload(workflow, payload)
        cache_key = self._cache_key(workflow, normalized)
        LOGGER.info("Run workflow sync: workflow=%s payload=%s", workflow, self._loggable_payload(normalized))
        cached = self.job_store.get_cached_result(cache_key)
        if cached is not None:
            LOGGER.info("Workflow cache hit: workflow=%s cache_key=%s", workflow, cache_key[:12])
            return {**cached, "cached": True}
        result = self._execute_workflow(workflow, normalized)
        self.job_store.put_cached_result(cache_key, result)
        LOGGER.info("Workflow sync completed: workflow=%s output_dir=%s", workflow, result.get("output_dir"))
        return result

    def submit_workflow(self, workflow: str, payload: dict[str, Any]) -> JobRecord:
        normalized = self._normalize_payload(workflow, payload)
        cache_key = self._cache_key(workflow, normalized)
        cached = self.job_store.get_cached_result(cache_key)
        LOGGER.info("Submit workflow async: workflow=%s payload=%s", workflow, self._loggable_payload(normalized))

        job = self.job_store.create_job(workflow=workflow, payload=normalized, cache_key=cache_key)
        if cached is not None:
            LOGGER.info("Async workflow cache hit: workflow=%s job_id=%s cache_key=%s", workflow, job.job_id, cache_key[:12])
            self.job_store.update_status(
                job.job_id,
                "completed",
                result={**cached, "cached": True},
                progress=100.0,
                detail="Loaded from cache",
            )
            completed = self.job_store.get_job(job.job_id)
            if completed is None:
                raise RuntimeError("Cached job was not retained.")
            return completed

        self.executor.submit(self._run_job, job.job_id, workflow, normalized, cache_key)
        queued = self.job_store.get_job(job.job_id)
        if queued is None:
            raise RuntimeError("Queued job was not retained.")
        return queued

    def get_job(self, job_id: str) -> JobRecord | None:
        return self.job_store.get_job(job_id)

    def _run_job(self, job_id: str, workflow: str, payload: dict[str, Any], cache_key: str) -> None:
        self.job_store.update_status(job_id, "running", progress=0.0, detail="Starting workflow")
        LOGGER.info("Job started: job_id=%s workflow=%s payload=%s", job_id, workflow, self._loggable_payload(payload))
        try:
            result = self._execute_workflow(
                workflow,
                payload,
                progress_callback=lambda progress, detail: self._handle_job_progress(job_id, workflow, progress, detail),
            )
            self.job_store.put_cached_result(cache_key, result)
            self.job_store.update_status(job_id, "completed", result=result, progress=100.0, detail="Completed")
            LOGGER.info("Job completed: job_id=%s workflow=%s output_dir=%s", job_id, workflow, result.get("output_dir"))
        except Exception as exc:
            self.job_store.update_status(job_id, "failed", error=str(exc), detail=str(exc))
            LOGGER.exception("Job failed: job_id=%s workflow=%s error=%s", job_id, workflow, exc)

    def _execute_workflow(
        self,
        workflow: str,
        payload: dict[str, Any],
        progress_callback=None,
    ) -> dict[str, Any]:
        handlers = {
            "validate": self._run_validation,
            "extract": self._run_extraction,
            "select": self._run_selection,
            "train": self._run_training,
            "predict": self._run_prediction,
            "full": self._run_full_pipeline,
        }
        handler = handlers.get(workflow)
        if handler is None:
            raise ValueError(f"Unsupported workflow: {workflow}")
        return handler(payload, progress_callback=progress_callback)

    def _run_validation(self, payload: dict[str, Any], progress_callback=None) -> dict[str, Any]:
        manifest_path = self._resolve_required_path(payload, "manifest")
        output_dir = self._ensure_output_dir(payload, "output_dir", "validation")
        LOGGER.info(
            "Validation request: manifest=%s output_dir=%s label_value=%s tolerance=%s",
            manifest_path,
            output_dir,
            payload.get("label_value", DEFAULT_MASK_LABEL),
            payload.get("tolerance", DEFAULT_GEOMETRY_TOLERANCE),
        )
        artifacts = run_validation_workflow(
            manifest_path,
            output_dir,
            label_value=int(payload.get("label_value", DEFAULT_MASK_LABEL)),
            tolerance=float(payload.get("tolerance", DEFAULT_GEOMETRY_TOLERANCE)),
            progress_callback=progress_callback,
        )
        valid_count = len(artifacts.report) - artifacts.invalid_count
        return self._workflow_response(
            title="Validation complete",
            summary=f"{len(artifacts.report)} cases checked, invalid={artifacts.invalid_count}",
            output_dir=output_dir,
            insights={
                "cards": [
                    self._metric_card("Total Cases", len(artifacts.report), "brand", "Manifest rows checked"),
                    self._metric_card("Valid Cases", valid_count, "success", "Ready for extraction"),
                    self._metric_card("Invalid Cases", artifacts.invalid_count, "error", "Need geometry or mask fixes"),
                    self._metric_card(
                        "Output Files",
                        len(make_download_listing(output_dir, self.settings.workspace)),
                        "neutral",
                        "CSV artifacts",
                    ),
                ],
                "bar_chart": self._bar_chart(
                    "Validation Status",
                    [
                        {"label": "Valid", "value": valid_count, "accent": "success"},
                        {"label": "Invalid", "value": artifacts.invalid_count, "accent": "error"},
                    ],
                ),
                "stage_view": self._stage_view(
                    [{"step": "validate", "state": "done", "detail": f"{valid_count} valid / {artifacts.invalid_count} invalid"}]
                ),
            },
        )

    def _run_extraction(self, payload: dict[str, Any], progress_callback=None) -> dict[str, Any]:
        manifest_path = self._resolve_required_path(payload, "manifest")
        params_path = self._resolve_required_path(payload, "params")
        output_dir = self._ensure_output_dir(payload, "output_dir", "features")
        LOGGER.info(
            "Extraction request: manifest=%s params=%s output_dir=%s label_value=%s tolerance=%s",
            manifest_path,
            params_path,
            output_dir,
            payload.get("label_value", DEFAULT_MASK_LABEL),
            payload.get("tolerance", DEFAULT_GEOMETRY_TOLERANCE),
        )
        artifacts = run_extraction_workflow(
            manifest_path,
            params_path,
            output_dir,
            label_value=int(payload.get("label_value", DEFAULT_MASK_LABEL)),
            tolerance=float(payload.get("tolerance", DEFAULT_GEOMETRY_TOLERANCE)),
            progress_callback=progress_callback,
        )
        success_count = len(artifacts.features)
        failure_count = len(artifacts.failures)
        return self._workflow_response(
            title="Feature extraction complete",
            summary=f"success={success_count} failed={failure_count}",
            output_dir=output_dir,
            insights={
                "cards": [
                    self._metric_card("Extracted Cases", success_count, "success", "Feature rows generated"),
                    self._metric_card("Failed Cases", failure_count, "error", "See failure log"),
                    self._metric_card(
                        "Per-case CSV",
                        len(list((output_dir / "per_case").glob("*.csv"))),
                        "brand",
                        "Detailed case outputs",
                    ),
                    self._metric_card(
                        "Output Files",
                        len(make_download_listing(output_dir, self.settings.workspace)),
                        "neutral",
                        "Artifacts ready to download",
                    ),
                ],
                "bar_chart": self._bar_chart(
                    "Extraction Outcome",
                    [
                        {"label": "Success", "value": success_count, "accent": "success"},
                        {"label": "Failed", "value": failure_count, "accent": "error"},
                    ],
                ),
                "stage_view": self._stage_view(
                    [{"step": "extract", "state": "done", "detail": f"{success_count} success / {failure_count} failed"}]
                ),
            },
        )

    def _run_selection(self, payload: dict[str, Any], progress_callback=None) -> dict[str, Any]:
        features_path = self._resolve_required_path(payload, "features")
        labels_path = self._resolve_required_path(payload, "labels", required=False)
        output_dir = self._ensure_output_dir(payload, "output_dir", "selection")
        LOGGER.info(
            "Selection request: features=%s labels=%s output_dir=%s variance=%s correlation=%s top_k=%s",
            features_path,
            labels_path,
            output_dir,
            payload.get("variance_threshold", DEFAULT_VARIANCE_THRESHOLD),
            payload.get("correlation_threshold", DEFAULT_CORRELATION_THRESHOLD),
            payload.get("top_k", DEFAULT_TOP_K),
        )
        artifacts = run_feature_selection_workflow(
            features_path,
            output_dir,
            labels_path=labels_path,
            label_column=str(payload.get("label_column", "label")),
            variance_threshold=float(payload.get("variance_threshold", DEFAULT_VARIANCE_THRESHOLD)),
            correlation_threshold=float(payload.get("correlation_threshold", DEFAULT_CORRELATION_THRESHOLD)),
            top_k=int(payload.get("top_k", DEFAULT_TOP_K)),
            selection_method=str(payload.get("selection_method", DEFAULT_SELECTION_METHOD)),
            progress_callback=progress_callback,
        )
        summary_map = {str(row["step"]): row["feature_count"] for row in artifacts.summary.to_dict(orient="records")}
        selection_method = str(summary_map.get("selection_method", payload.get("selection_method", DEFAULT_SELECTION_METHOD)))
        selection_method_label = str(summary_map.get("selection_method_label", selection_method))
        selected_feature_names = [
            column
            for column in artifacts.selected_features.columns
            if column not in {"case_id", str(payload.get("label_column", "label"))}
        ]
        return self._workflow_response(
            title="Feature selection complete",
            summary=f"selected {len(artifacts.selected_features.columns) - 2} feature columns",
            output_dir=output_dir,
            table=artifacts.summary.to_dict(orient="records"),
            selected_features=selected_feature_names,
            insights={
                "cards": [
                    self._metric_card("Raw Numeric", summary_map.get("raw_numeric", 0), "neutral", "Before filtering"),
                    self._metric_card("After Variance", summary_map.get("after_variance", 0), "brand", "Low-variance removed"),
                    self._metric_card(
                        "After Correlation",
                        summary_map.get("after_correlation", 0),
                        "brand",
                        "Collinear features reduced",
                    ),
                    self._metric_card(
                        "After Selection",
                        summary_map.get("after_selection", 0),
                        "success",
                        selection_method_label,
                    ),
                ],
                "bar_chart": self._bar_chart(
                    "Feature Count by Stage",
                    [
                        {"label": "Raw", "value": int(summary_map.get("raw_numeric", 0)), "accent": "neutral"},
                        {"label": "Variance", "value": int(summary_map.get("after_variance", 0)), "accent": "brand"},
                        {"label": "Correlation", "value": int(summary_map.get("after_correlation", 0)), "accent": "brand"},
                        {"label": "Selected", "value": int(summary_map.get("after_selection", 0)), "accent": "success"},
                    ],
                ),
                "stage_view": self._stage_view(
                    [
                        {"step": "clean", "state": "done", "detail": f"{summary_map.get('after_variance', 0)} after variance filter"},
                        {
                            "step": "decorrelate",
                            "state": "done",
                            "detail": f"{summary_map.get('after_correlation', 0)} after correlation filter",
                        },
                        {"step": "select", "state": "done", "detail": f"{summary_map.get('after_selection', 0)} final features"},
                    ]
                ),
            },
        )

    def _run_training(self, payload: dict[str, Any], progress_callback=None) -> dict[str, Any]:
        features_path = self._resolve_required_path(payload, "features")
        labels_path = self._resolve_required_path(payload, "labels", required=False)
        output_dir = self._ensure_output_dir(payload, "output_dir", "model")
        LOGGER.info(
            "Training request: features=%s labels=%s output_dir=%s label_column=%s folds=%s random_state=%s",
            features_path,
            labels_path,
            output_dir,
            payload.get("label_column", "label"),
            payload.get("folds", DEFAULT_FOLDS),
            payload.get("random_state", DEFAULT_RANDOM_STATE),
        )
        artifacts = run_training_workflow(
            features_path,
            output_dir,
            labels_path=labels_path,
            label_column=str(payload.get("label_column", "label")),
            folds=int(payload.get("folds", DEFAULT_FOLDS)),
            random_state=int(payload.get("random_state", DEFAULT_RANDOM_STATE)),
            model_names=self._as_string_list(payload.get("models")),
            progress_callback=progress_callback,
        )
        metric_rows = artifacts.metrics.round(4).to_dict(orient="records")
        best_row = metric_rows[0] if metric_rows else {}
        return self._workflow_response(
            title="Training complete",
            summary=f"evaluated {len(artifacts.metrics)} models",
            output_dir=output_dir,
            table=metric_rows,
            best_model_path=best_row.get("model_path", ""),
            insights={
                "cards": [
                    self._metric_card("Models", len(metric_rows), "brand", "Algorithms evaluated"),
                    self._metric_card("Best Model", best_row.get("model_label", best_row.get("model", "-")), "success", "Top by macro F1"),
                    self._metric_card("Best ROC AUC", best_row.get("roc_auc", "-"), "brand", "Cross-validation"),
                    self._metric_card("Best Macro F1", best_row.get("f1_macro", "-"), "success", "Cross-validation"),
                    self._metric_card("Best Accuracy", best_row.get("accuracy", "-"), "neutral", "Cross-validation"),
                ],
                "bar_chart": self._bar_chart(
                    "Model Comparison",
                    [{"label": row.get("model_label", row["model"]), "value": float(row.get("roc_auc") or row["f1_macro"]), "accent": "success"} for row in metric_rows],
                ),
                "stage_view": self._stage_view(
                    [{"step": row.get("model_label", row["model"]), "state": "done", "detail": f"AUC={row.get('roc_auc', '-')} | F1={row['f1_macro']} | Acc={row['accuracy']}"} for row in metric_rows]
                ),
            },
        )

    def _run_prediction(self, payload: dict[str, Any], progress_callback=None) -> dict[str, Any]:
        features_path = self._resolve_required_path(payload, "features")
        model_path = self._resolve_required_path(payload, "model")
        labels_path = self._resolve_required_path(payload, "labels", required=False)
        output_dir = self._ensure_output_dir(payload, "output_dir", "prediction")
        LOGGER.info(
            "Prediction request: features=%s model=%s labels=%s output_dir=%s",
            features_path,
            model_path,
            labels_path,
            output_dir,
        )
        artifacts = run_prediction_workflow(
            features_path,
            model_path,
            output_dir,
            labels_path=labels_path,
            label_column=str(payload.get("label_column", "label")),
            progress_callback=progress_callback,
        )
        prediction_rows = artifacts.predictions.head(50).to_dict(orient="records")
        metric_rows = artifacts.metrics.round(4).to_dict(orient="records") if not artifacts.metrics.empty else []
        best_metrics = metric_rows[0] if metric_rows else {}
        evaluated = not artifacts.metrics.empty
        confidence_series = pd.to_numeric(artifacts.predictions.get("confidence"), errors="coerce")
        avg_confidence = round(float(confidence_series.dropna().mean()), 4) if not confidence_series.dropna().empty else "-"
        return self._workflow_response(
            title="Prediction complete" if not evaluated else "Prediction and evaluation complete",
            summary=(
                f"generated {len(artifacts.predictions)} predictions without labels"
                if not evaluated
                else f"generated {len(artifacts.predictions)} predictions and evaluated 1 model"
            ),
            output_dir=output_dir,
            table=prediction_rows,
            insights={
                "cards": [
                    self._metric_card("Model", artifacts.model_name, "brand", "Loaded trained model"),
                    self._metric_card("Predicted Cases", len(artifacts.predictions), "success", "Rows in prediction output"),
                    self._metric_card("Avg Confidence", avg_confidence, "brand", "Mean confidence across predictions"),
                    self._metric_card("ROC AUC" if evaluated else "ROC AUC", best_metrics.get("roc_auc", "Skipped") if evaluated else "Skipped", "brand" if evaluated else "neutral", "Computed from probabilities" if evaluated else "Provide labels to evaluate"),
                    self._metric_card(
                        "Accuracy" if evaluated else "Evaluation",
                        best_metrics.get("accuracy", "Skipped") if evaluated else "Skipped",
                        "success" if evaluated else "neutral",
                        "Computed from labels" if evaluated else "Provide labels CSV to evaluate",
                    ),
                    self._metric_card("Macro F1" if evaluated else "Metrics File", best_metrics.get("f1_macro", "-") if evaluated else ("Available" if artifacts.metrics_path else "Not created"), "brand" if evaluated else "neutral", "Prediction metrics" if evaluated else "Predictions only"),
                ],
                "bar_chart": self._bar_chart(
                    "Prediction Summary",
                    [
                        {"label": "Predictions", "value": len(artifacts.predictions), "accent": "brand"},
                        {"label": "Accuracy", "value": float(best_metrics.get("accuracy", 0)) if evaluated else 0, "accent": "success"},
                    ],
                ),
                "stage_view": self._stage_view(
                    [
                        {"step": "load_model", "state": "done", "detail": f"Loaded {artifacts.model_name}"},
                        {"step": "predict", "state": "done", "detail": f"{len(artifacts.predictions)} cases predicted"},
                        {
                            "step": "evaluate" if evaluated else "evaluate",
                            "state": "done" if evaluated else "pending",
                            "detail": "Metrics computed" if evaluated else "Skipped because labels were not provided",
                        },
                    ]
                ),
            },
        )

    def _run_full_pipeline(self, payload: dict[str, Any], progress_callback=None) -> dict[str, Any]:
        manifest_path = self._resolve_required_path(payload, "manifest")
        params_path = self._resolve_required_path(payload, "params")
        labels_path = self._resolve_required_path(payload, "labels", required=False)
        output_dir = self._ensure_output_dir(payload, "output_dir", "pipeline")
        LOGGER.info(
            "Full pipeline request: manifest=%s params=%s labels=%s output_dir=%s top_k=%s folds=%s",
            manifest_path,
            params_path,
            labels_path,
            output_dir,
            payload.get("top_k", DEFAULT_TOP_K),
            payload.get("folds", DEFAULT_FOLDS),
        )
        artifacts = run_full_pipeline_workflow(
            manifest_path,
            params_path,
            output_dir,
            labels_path=labels_path,
            label_column=str(payload.get("label_column", "label")),
            label_value=int(payload.get("label_value", DEFAULT_MASK_LABEL)),
            tolerance=float(payload.get("tolerance", DEFAULT_GEOMETRY_TOLERANCE)),
            variance_threshold=float(payload.get("variance_threshold", DEFAULT_VARIANCE_THRESHOLD)),
            correlation_threshold=float(payload.get("correlation_threshold", DEFAULT_CORRELATION_THRESHOLD)),
            top_k=int(payload.get("top_k", DEFAULT_TOP_K)),
            selection_method=str(payload.get("selection_method", DEFAULT_SELECTION_METHOD)),
            folds=int(payload.get("folds", DEFAULT_FOLDS)),
            random_state=int(payload.get("random_state", DEFAULT_RANDOM_STATE)),
            model_names=self._as_string_list(payload.get("models")),
            progress_callback=progress_callback,
        )
        model_rows = artifacts.training.metrics.round(4).to_dict(orient="records")
        best_model = model_rows[0] if model_rows else {}
        selection_summary = {str(row["step"]): row["feature_count"] for row in artifacts.selection.summary.to_dict(orient="records")}
        selected_feature_names = [
            column
            for column in artifacts.selection.selected_features.columns
            if column not in {"case_id", str(payload.get("label_column", "label"))}
        ]
        steps = [
            {"step": "validation", "summary": f"{len(artifacts.validation.report)} cases checked, invalid={artifacts.validation.invalid_count}"},
            {"step": "extraction", "summary": f"success={len(artifacts.extraction.features)} failed={len(artifacts.extraction.failures)}"},
            {"step": "selection", "summary": f"selected {len(artifacts.selection.selected_features.columns) - 2} feature columns"},
            {"step": "training", "summary": f"evaluated {len(artifacts.training.metrics)} models"},
        ]
        return self._workflow_response(
            title="Full pipeline complete",
            summary="Validation, extraction, feature selection, and training all finished.",
            output_dir=output_dir,
            table=model_rows,
            steps=steps,
            selected_features=selected_feature_names,
            best_model_path=best_model.get("model_path", ""),
            insights={
                "cards": [
                    self._metric_card("Stages Done", 4, "success", "Validation to modeling"),
                    self._metric_card(
                        "Valid Cases",
                        len(artifacts.validation.report) - artifacts.validation.invalid_count,
                        "brand",
                        "Passed validation",
                    ),
                    self._metric_card("Final Features", selection_summary.get("after_selection", 0), "brand", "After selection"),
                    self._metric_card("Best ROC AUC", best_model.get("roc_auc", "-"), "brand", "Cross-validation"),
                    self._metric_card(
                        "Best Model",
                        best_model.get("model_label", best_model.get("model", "-")),
                        "success",
                        f"F1={best_model.get('f1_macro', '-')}",
                    ),
                ],
                "bar_chart": self._bar_chart(
                    "Pipeline Overview",
                    [
                        {
                            "label": "Valid",
                            "value": len(artifacts.validation.report) - artifacts.validation.invalid_count,
                            "accent": "success",
                        },
                        {"label": "Extracted", "value": len(artifacts.extraction.features), "accent": "brand"},
                        {"label": "Selected", "value": int(selection_summary.get("after_selection", 0)), "accent": "brand"},
                        {"label": "Models", "value": len(model_rows), "accent": "neutral"},
                    ],
                ),
                "stage_view": self._stage_view(
                    [
                        {"step": "validate", "state": "done", "detail": steps[0]["summary"]},
                        {"step": "extract", "state": "done", "detail": steps[1]["summary"]},
                        {"step": "select", "state": "done", "detail": steps[2]["summary"]},
                        {"step": "train", "state": "done", "detail": steps[3]["summary"]},
                    ]
                ),
            },
        )

    def _normalize_payload(self, workflow: str, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        normalized["task_name"] = self._normalize_task_name(payload.get("task_name"))
        normalized["workflow"] = workflow
        return normalized

    def _cache_key(self, workflow: str, payload: dict[str, Any]) -> str:
        material = {key: value for key, value in payload.items() if key not in {"output_dir", "run_async", "workflow"}}
        encoded = json.dumps(material, ensure_ascii=True, sort_keys=True)
        return hashlib.sha256(f"{WORKFLOW_CACHE_VERSION}:{workflow}:{encoded}".encode("utf-8")).hexdigest()

    def _resolve_required_path(self, payload: dict[str, Any], key: str, required: bool = True) -> Path | None:
        raw_value = payload.get(key)
        path = resolve_user_path(self.settings.workspace, None if raw_value is None else str(raw_value))
        if required and path is None:
            raise ValueError(f"{key} is required.")
        return path

    def _ensure_output_dir(self, payload: dict[str, Any], key: str, prefix: str) -> Path:
        task_name = self._normalize_task_name(payload.get("task_name"))
        return ensure_output_dir(
            self.settings.workspace,
            None if payload.get(key) is None else str(payload.get(key)),
            str((self._task_root(task_name) / prefix).resolve()),
        )

    def _absolute_default(self, *parts: str) -> str:
        return str((self.settings.workspace / Path(*parts)).resolve())

    def _default_run_dir(self, prefix: str, task_name: str = "default") -> str:
        return str((self._task_root(task_name) / prefix).resolve())

    def _task_root(self, task_name: str) -> Path:
        return (self.settings.workspace / "outputs" / "tasks" / self._normalize_task_name(task_name)).resolve()

    def _discover_trained_models(self, root: Path) -> list[dict[str, Any]]:
        if not root.exists():
            return []
        candidates = []
        for path in sorted(root.rglob("trained_model_*.pkl"), key=lambda item: item.stat().st_mtime, reverse=True):
            stat = path.stat()
            name = path.stem.removeprefix("trained_model_")
            candidates.append(
                {
                    "name": name,
                    "label": name.replace("_", " ").title(),
                    "path": str(path.resolve()),
                    "modified_at": stat.st_mtime,
                    "display_path": str(path.resolve().relative_to(self.settings.workspace.resolve())) if self.settings.workspace.resolve() in path.resolve().parents else str(path.resolve()),
                }
            )
        return candidates

    def _latest_trained_model_path(self, preferred_root: str | Path | None = None) -> str:
        preferred = Path(preferred_root).resolve() if preferred_root else None
        if preferred is not None:
            models = self._discover_trained_models(preferred)
            if models:
                return models[0]["path"]
        models = self._discover_trained_models(self.settings.workspace / "outputs")
        return models[0]["path"] if models else ""

    def _normalize_task_name(self, task_name: Any) -> str:
        raw = str(task_name or "default").strip()
        if not raw:
            raw = "default"
        normalized = re.sub(r"[^0-9A-Za-z._-]+", "-", raw)
        normalized = normalized.strip(".-_")
        return normalized or "default"

    def _workflow_response(
        self,
        title: str,
        summary: str,
        output_dir: Path,
        table: list[dict[str, Any]] | None = None,
        steps: list[dict[str, Any]] | None = None,
        insights: dict[str, Any] | None = None,
        selected_features: list[str] | None = None,
        best_model_path: str = "",
    ) -> dict[str, Any]:
        return {
            "ok": True,
            "title": title,
            "summary": summary,
            "output_dir": str(output_dir),
            "files": make_download_listing(output_dir, self.settings.workspace),
            "table": table or [],
            "steps": steps or [],
            "insights": insights or {},
            "selected_features": selected_features or [],
            "best_model_path": best_model_path,
        }

    def _metric_card(self, label: str, value: Any, accent: str = "brand", detail: str | None = None) -> dict[str, Any]:
        return {"label": label, "value": value, "accent": accent, "detail": detail}

    def _bar_chart(self, title: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        return {"title": title, "items": items}

    def _stage_view(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        return {"items": items}

    def _summarize_labels(self, frame: pd.DataFrame, label_column: str) -> list[dict[str, Any]]:
        if label_column not in frame.columns:
            return []
        label_series = frame[label_column].replace("", pd.NA).dropna().astype(str)
        if label_series.empty:
            return []
        counts = label_series.value_counts().reset_index()
        counts.columns = ["label", "count"]
        return counts.to_dict(orient="records")

    def _group_feature_name(self, name: str) -> str:
        parts = name.split("_")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return parts[0]

    def _handle_job_progress(self, job_id: str, workflow: str, progress: float, detail: str) -> None:
        self.job_store.update_progress(job_id, progress, detail)
        LOGGER.info("Job progress: job_id=%s workflow=%s progress=%.0f%% detail=%s", job_id, workflow, progress, detail)

    def _loggable_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        compact: dict[str, Any] = {}
        for key, value in payload.items():
            if key == "workflow":
                continue
            text = str(value)
            if key.endswith("_path") or key in {"manifest", "params", "labels", "features", "output_dir"}:
                compact[key] = text
            elif len(text) > 120:
                compact[key] = f"{text[:117]}..."
            else:
                compact[key] = value
        return compact

    def _as_string_list(self, value: Any) -> list[str] | None:
        if value is None or value == "":
            return None
        if isinstance(value, list):
            items = [str(item).strip() for item in value if str(item).strip()]
            return items or None
        if isinstance(value, tuple):
            items = [str(item).strip() for item in value if str(item).strip()]
            return items or None
        text = str(value).strip()
        if not text:
            return None
        return [part.strip() for part in text.split(",") if part.strip()]
