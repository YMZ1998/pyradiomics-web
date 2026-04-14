from __future__ import annotations

import hashlib
import json
from concurrent.futures import Executor
from pathlib import Path
from typing import Any

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
from ..domain.workflows import (
    run_extraction_workflow,
    run_feature_selection_workflow,
    run_full_pipeline_workflow,
    run_training_workflow,
    run_validation_workflow,
)
from ..infrastructure.filesystem import ensure_output_dir, make_download_listing, resolve_user_path, timestamp_token
from ..infrastructure.jobs import InMemoryJobStore, JobRecord
from ..infrastructure.settings import AppSettings

WORKFLOW_NAMES = {"validate", "extract", "select", "train", "full"}


class ClinicalRadiomicsPlatformService:
    def __init__(self, settings: AppSettings, job_store: InMemoryJobStore, executor: Executor) -> None:
        self.settings = settings
        self.job_store = job_store
        self.executor = executor

    def default_paths(self) -> dict[str, Any]:
        validation_output_dir = self._default_run_dir("validation")
        extraction_output_dir = self._default_run_dir("features")
        selection_output_dir = self._default_run_dir("selection")
        training_output_dir = self._default_run_dir("model")
        pipeline_output_dir = self._default_run_dir("pipeline")
        return {
            "manifest": self._absolute_default("manifests", "examples", "Dataset006_NPC2_cases.csv"),
            "params": self._absolute_default("configs", "ct_radiomics.yaml"),
            "labels": self._absolute_default("manifests", "labels_template.csv"),
            "label_column": "label",
            "label_value": DEFAULT_MASK_LABEL,
            "tolerance": DEFAULT_GEOMETRY_TOLERANCE,
            "variance_threshold": DEFAULT_VARIANCE_THRESHOLD,
            "correlation_threshold": DEFAULT_CORRELATION_THRESHOLD,
            "top_k": DEFAULT_TOP_K,
            "folds": DEFAULT_FOLDS,
            "random_state": DEFAULT_RANDOM_STATE,
            "validation_output_dir": validation_output_dir,
            "extraction_output_dir": extraction_output_dir,
            "selection_output_dir": selection_output_dir,
            "training_output_dir": training_output_dir,
            "pipeline_output_dir": pipeline_output_dir,
            "extraction_features_path": str((Path(extraction_output_dir) / "features.csv").resolve()),
            "selection_features_path": str((Path(selection_output_dir) / "selected_features.csv").resolve()),
        }

    def inspect_data(self, payload: dict[str, Any]) -> dict[str, Any]:
        manifest_path = self._resolve_required_path(payload, "manifest")
        labels_path = self._resolve_required_path(payload, "labels", required=False)
        label_column = str(payload.get("label_column", "label"))

        manifest = pd.read_csv(manifest_path)
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

        groups: dict[str, list[str]] = {}
        for column in feature_columns:
            group_name = self._group_feature_name(column)
            groups.setdefault(group_name, []).append(column)

        group_rows = [
            {
                "group": group_name,
                "count": len(names),
                "examples": ", ".join(names[:3]),
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

    def run_workflow_sync(self, workflow: str, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize_payload(workflow, payload)
        cache_key = self._cache_key(workflow, normalized)
        cached = self.job_store.get_cached_result(cache_key)
        if cached is not None:
            return {**cached, "cached": True}
        result = self._execute_workflow(workflow, normalized)
        self.job_store.put_cached_result(cache_key, result)
        return result

    def submit_workflow(self, workflow: str, payload: dict[str, Any]) -> JobRecord:
        normalized = self._normalize_payload(workflow, payload)
        cache_key = self._cache_key(workflow, normalized)
        cached = self.job_store.get_cached_result(cache_key)

        job = self.job_store.create_job(workflow=workflow, payload=normalized, cache_key=cache_key)
        if cached is not None:
            self.job_store.update_status(job.job_id, "completed", result={**cached, "cached": True})
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
        self.job_store.update_status(job_id, "running")
        try:
            result = self._execute_workflow(workflow, payload)
            self.job_store.put_cached_result(cache_key, result)
            self.job_store.update_status(job_id, "completed", result=result)
        except Exception as exc:
            self.job_store.update_status(job_id, "failed", error=str(exc))

    def _execute_workflow(self, workflow: str, payload: dict[str, Any]) -> dict[str, Any]:
        handlers = {
            "validate": self._run_validation,
            "extract": self._run_extraction,
            "select": self._run_selection,
            "train": self._run_training,
            "full": self._run_full_pipeline,
        }
        handler = handlers.get(workflow)
        if handler is None:
            raise ValueError(f"Unsupported workflow: {workflow}")
        return handler(payload)

    def _run_validation(self, payload: dict[str, Any]) -> dict[str, Any]:
        manifest_path = self._resolve_required_path(payload, "manifest")
        output_dir = self._ensure_output_dir(payload, "output_dir", "validation")
        artifacts = run_validation_workflow(
            manifest_path,
            output_dir,
            label_value=int(payload.get("label_value", DEFAULT_MASK_LABEL)),
            tolerance=float(payload.get("tolerance", DEFAULT_GEOMETRY_TOLERANCE)),
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

    def _run_extraction(self, payload: dict[str, Any]) -> dict[str, Any]:
        manifest_path = self._resolve_required_path(payload, "manifest")
        params_path = self._resolve_required_path(payload, "params")
        output_dir = self._ensure_output_dir(payload, "output_dir", "features")
        artifacts = run_extraction_workflow(
            manifest_path,
            params_path,
            output_dir,
            label_value=int(payload.get("label_value", DEFAULT_MASK_LABEL)),
            tolerance=float(payload.get("tolerance", DEFAULT_GEOMETRY_TOLERANCE)),
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

    def _run_selection(self, payload: dict[str, Any]) -> dict[str, Any]:
        features_path = self._resolve_required_path(payload, "features")
        labels_path = self._resolve_required_path(payload, "labels", required=False)
        output_dir = self._ensure_output_dir(payload, "output_dir", "selection")
        artifacts = run_feature_selection_workflow(
            features_path,
            output_dir,
            labels_path=labels_path,
            label_column=str(payload.get("label_column", "label")),
            variance_threshold=float(payload.get("variance_threshold", DEFAULT_VARIANCE_THRESHOLD)),
            correlation_threshold=float(payload.get("correlation_threshold", DEFAULT_CORRELATION_THRESHOLD)),
            top_k=int(payload.get("top_k", DEFAULT_TOP_K)),
        )
        summary_map = {str(row["step"]): row["feature_count"] for row in artifacts.summary.to_dict(orient="records")}
        return self._workflow_response(
            title="Feature selection complete",
            summary=f"selected {len(artifacts.selected_features.columns) - 2} feature columns",
            output_dir=output_dir,
            table=artifacts.summary.to_dict(orient="records"),
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
                        str(summary_map.get("selection_mode", "selection")),
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

    def _run_training(self, payload: dict[str, Any]) -> dict[str, Any]:
        features_path = self._resolve_required_path(payload, "features")
        labels_path = self._resolve_required_path(payload, "labels", required=False)
        output_dir = self._ensure_output_dir(payload, "output_dir", "model")
        artifacts = run_training_workflow(
            features_path,
            output_dir,
            labels_path=labels_path,
            label_column=str(payload.get("label_column", "label")),
            folds=int(payload.get("folds", DEFAULT_FOLDS)),
            random_state=int(payload.get("random_state", DEFAULT_RANDOM_STATE)),
        )
        metric_rows = artifacts.metrics.round(4).to_dict(orient="records")
        best_row = metric_rows[0] if metric_rows else {}
        return self._workflow_response(
            title="Training complete",
            summary=f"evaluated {len(artifacts.metrics)} models",
            output_dir=output_dir,
            table=metric_rows,
            insights={
                "cards": [
                    self._metric_card("Models", len(metric_rows), "brand", "Algorithms evaluated"),
                    self._metric_card("Best Model", best_row.get("model", "-"), "success", "Top by macro F1"),
                    self._metric_card("Best Macro F1", best_row.get("f1_macro", "-"), "success", "Cross-validation"),
                    self._metric_card("Best Accuracy", best_row.get("accuracy", "-"), "neutral", "Cross-validation"),
                ],
                "bar_chart": self._bar_chart(
                    "Model Comparison",
                    [{"label": row["model"], "value": float(row["f1_macro"]), "accent": "success"} for row in metric_rows],
                ),
                "stage_view": self._stage_view(
                    [{"step": row["model"], "state": "done", "detail": f"F1={row['f1_macro']} | Acc={row['accuracy']}"} for row in metric_rows]
                ),
            },
        )

    def _run_full_pipeline(self, payload: dict[str, Any]) -> dict[str, Any]:
        manifest_path = self._resolve_required_path(payload, "manifest")
        params_path = self._resolve_required_path(payload, "params")
        labels_path = self._resolve_required_path(payload, "labels", required=False)
        output_dir = self._ensure_output_dir(payload, "output_dir", "pipeline")
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
            folds=int(payload.get("folds", DEFAULT_FOLDS)),
            random_state=int(payload.get("random_state", DEFAULT_RANDOM_STATE)),
        )
        model_rows = artifacts.training.metrics.round(4).to_dict(orient="records")
        best_model = model_rows[0] if model_rows else {}
        selection_summary = {str(row["step"]): row["feature_count"] for row in artifacts.selection.summary.to_dict(orient="records")}
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
                    self._metric_card(
                        "Best Model",
                        best_model.get("model", "-"),
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
        normalized["workflow"] = workflow
        return normalized

    def _cache_key(self, workflow: str, payload: dict[str, Any]) -> str:
        material = {key: value for key, value in payload.items() if key not in {"output_dir", "run_async", "workflow"}}
        encoded = json.dumps(material, ensure_ascii=True, sort_keys=True)
        return hashlib.sha256(f"{workflow}:{encoded}".encode("utf-8")).hexdigest()

    def _resolve_required_path(self, payload: dict[str, Any], key: str, required: bool = True) -> Path | None:
        raw_value = payload.get(key)
        path = resolve_user_path(self.settings.workspace, None if raw_value is None else str(raw_value))
        if required and path is None:
            raise ValueError(f"{key} is required.")
        return path

    def _ensure_output_dir(self, payload: dict[str, Any], key: str, prefix: str) -> Path:
        return ensure_output_dir(
            self.settings.workspace,
            None if payload.get(key) is None else str(payload.get(key)),
            f"{prefix}-{timestamp_token()}",
        )

    def _absolute_default(self, *parts: str) -> str:
        return str((self.settings.workspace / Path(*parts)).resolve())

    def _default_run_dir(self, prefix: str) -> str:
        return str((self.settings.workspace / "outputs" / f"{prefix}-{timestamp_token()}").resolve())

    def _workflow_response(
        self,
        title: str,
        summary: str,
        output_dir: Path,
        table: list[dict[str, Any]] | None = None,
        steps: list[dict[str, Any]] | None = None,
        insights: dict[str, Any] | None = None,
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
