from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import Callable

import pandas as pd

from .validation import load_manifest, validate_case

LOGGER = logging.getLogger("pyrad_workflow")
ProgressCallback = Callable[[float, str], None]


def load_extractor(params_path: Path):
    try:
        from radiomics import featureextractor
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "PyRadiomics is not installed in the current environment. "
            "Install dependencies and rerun extraction."
        ) from exc

    LOGGER.info("Loading PyRadiomics extractor parameters: %s", params_path)
    return featureextractor.RadiomicsFeatureExtractor(str(params_path))


def extract_features(
    manifest_path: Path,
    params_path: Path,
    output_dir: Path,
    label_value: int = 1,
    tolerance: float = 1e-6,
    progress_callback: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_case_dir = output_dir / "per_case"
    per_case_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    extractor = load_extractor(params_path)
    total_cases = len(manifest)
    safe_total = max(total_cases, 1)
    LOGGER.info("Extraction manifest preview: %s", manifest.head(3).to_dict(orient="records"))

    LOGGER.info(
        "Starting feature extraction for %s cases: manifest=%s params=%s label_value=%s tolerance=%s",
        total_cases,
        manifest_path,
        params_path,
        label_value,
        tolerance,
    )
    if progress_callback is not None:
        progress_callback(0.0, f"Preparing extraction for {total_cases} cases")

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, (_, row) in enumerate(manifest.iterrows(), start=1):
        case_id = str(row["case_id"])
        LOGGER.info("Extracting features for case %s/%s: %s", index, total_cases, case_id)
        validation = validate_case(row, label_value=label_value, tolerance=tolerance)
        if not validation.is_valid:
            LOGGER.warning(
                "Skipping case %s because validation failed: status=%s message=%s",
                case_id,
                validation.status,
                validation.message,
            )
            failures.append(
                {
                    "case_id": case_id,
                    "status": validation.status,
                    "message": validation.message,
                }
            )
            continue

        try:
            result = extractor.execute(str(row["image_path"]), str(row["mask_path"]), label=label_value)
        except Exception as exc:
            LOGGER.warning("Feature extraction failed for case %s: %s", case_id, exc)
            failures.append(
                {
                    "case_id": case_id,
                    "status": "extraction_error",
                    "message": str(exc),
                }
            )
            continue

        normalized = {"case_id": case_id, "image_path": row["image_path"], "mask_path": row["mask_path"], "label": row.get("label")}
        for key, value in result.items():
            if isinstance(value, (list, tuple)):
                normalized[key] = str(value)
            else:
                normalized[key] = value

        case_frame = pd.DataFrame([normalized])
        case_frame.to_csv(per_case_dir / f"{case_id}.csv", index=False)
        LOGGER.info("Case %s extracted successfully with %s fields", case_id, len(case_frame.columns))
        rows.append(normalized)
        if progress_callback is not None:
            progress_callback((index / safe_total) * 100.0, f"Processed case {index}/{total_cases}: {case_id}")

    features_frame = pd.DataFrame(rows)
    failures_frame = pd.DataFrame(failures)
    LOGGER.info(
        "Feature extraction summary: total_cases=%s succeeded=%s failed=%s per_case_dir=%s",
        total_cases,
        len(features_frame),
        len(failures_frame),
        per_case_dir,
    )
    LOGGER.info(
        "Feature extraction data summary: feature_rows=%s feature_columns=%s failure_rows=%s",
        len(features_frame),
        len(features_frame.columns) if not features_frame.empty else 0,
        len(failures_frame),
    )
    if not failures_frame.empty:
        LOGGER.info("Feature extraction failure preview: %s", failures_frame.head(5).to_dict(orient="records"))
    if progress_callback is not None:
        progress_callback(100.0, "Feature extraction complete")
    return features_frame, failures_frame
