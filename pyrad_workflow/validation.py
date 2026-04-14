from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .io_utils import count_mask_voxels, geometry_matches, image_metadata, read_pair

LOGGER = logging.getLogger("pyrad_workflow")

REQUIRED_COLUMNS = ("case_id", "image_path", "mask_path", "label")


@dataclass(frozen=True)
class ValidationResult:
    case_id: str
    is_valid: bool
    status: str
    message: str
    mask_voxels: int
    image_size: str
    mask_size: str
    image_spacing: str
    mask_spacing: str

    def to_record(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "is_valid": self.is_valid,
            "status": self.status,
            "message": self.message,
            "mask_voxels": self.mask_voxels,
            "image_size": self.image_size,
            "mask_size": self.mask_size,
            "image_spacing": self.image_spacing,
            "mask_spacing": self.mask_spacing,
        }


def load_manifest(path: Path) -> pd.DataFrame:
    LOGGER.info("Loading manifest: %s", path)
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        LOGGER.error("Manifest missing required columns: %s", ", ".join(missing))
        raise ValueError(f"Manifest missing required columns: {', '.join(missing)}")
    if frame["case_id"].duplicated().any():
        duplicates = frame.loc[frame["case_id"].duplicated(), "case_id"].astype(str).tolist()
        LOGGER.error("Manifest contains duplicate case_id values: %s", ", ".join(duplicates))
        raise ValueError(f"Manifest contains duplicate case_id values: {', '.join(duplicates)}")
    LOGGER.info("Manifest loaded successfully with %s cases", len(frame))
    return frame


def validate_case(row: pd.Series, label_value: int = 1, tolerance: float = 1e-6) -> ValidationResult:
    case_id = str(row["case_id"])
    image_path = Path(str(row["image_path"]))
    mask_path = Path(str(row["mask_path"]))

    LOGGER.debug("Validating case %s: image=%s mask=%s", case_id, image_path, mask_path)

    if not image_path.exists():
        LOGGER.warning("Case %s missing image: %s", case_id, image_path)
        return ValidationResult(case_id, False, "missing_image", f"Image not found: {image_path}", 0, "", "", "", "")
    if not mask_path.exists():
        LOGGER.warning("Case %s missing mask: %s", case_id, mask_path)
        return ValidationResult(case_id, False, "missing_mask", f"Mask not found: {mask_path}", 0, "", "", "", "")

    try:
        pair = read_pair(image_path, mask_path)
    except Exception as exc:
        LOGGER.warning("Case %s failed to read image/mask pair: %s", case_id, exc)
        return ValidationResult(case_id, False, "read_error", str(exc), 0, "", "", "", "")

    image_info = image_metadata(pair.image)
    mask_info = image_metadata(pair.mask)

    if not geometry_matches(pair.image, pair.mask, tolerance=tolerance):
        LOGGER.warning("Case %s failed geometry validation", case_id)
        return ValidationResult(
            case_id,
            False,
            "geometry_mismatch",
            "Image and mask geometry do not match.",
            0,
            str(image_info["size"]),
            str(mask_info["size"]),
            str(image_info["spacing"]),
            str(mask_info["spacing"]),
        )

    mask_voxels = count_mask_voxels(pair.mask, label=label_value)
    if mask_voxels <= 0:
        LOGGER.warning("Case %s mask is empty for label=%s", case_id, label_value)
        return ValidationResult(
            case_id,
            False,
            "empty_mask",
            f"Mask contains no voxels with label {label_value}.",
            0,
            str(image_info["size"]),
            str(mask_info["size"]),
            str(image_info["spacing"]),
            str(mask_info["spacing"]),
        )

    LOGGER.debug("Case %s validation passed with %s voxels", case_id, mask_voxels)
    return ValidationResult(
        case_id,
        True,
        "ok",
        "Validation passed.",
        mask_voxels,
        str(image_info["size"]),
        str(mask_info["size"]),
        str(image_info["spacing"]),
        str(mask_info["spacing"]),
    )


def validate_manifest(manifest_path: Path, label_value: int = 1, tolerance: float = 1e-6) -> pd.DataFrame:
    manifest = load_manifest(manifest_path)
    LOGGER.info(
        "Validating %s cases with label_value=%s tolerance=%s",
        len(manifest),
        label_value,
        tolerance,
    )

    results = []
    for index, (_, row) in enumerate(manifest.iterrows(), start=1):
        case_id = str(row["case_id"])
        LOGGER.info("Validating case %s/%s: %s", index, len(manifest), case_id)
        result = validate_case(row, label_value=label_value, tolerance=tolerance)
        if result.is_valid:
            LOGGER.info("Case %s validation status=%s", case_id, result.status)
        else:
            LOGGER.warning("Case %s validation status=%s message=%s", case_id, result.status, result.message)
        results.append(result.to_record())

    result_frame = pd.DataFrame(results)
    invalid_count = int((~result_frame["is_valid"].astype(bool)).sum())
    LOGGER.info("Validation summary: total_cases=%s invalid_cases=%s", len(result_frame), invalid_count)
    return manifest.merge(result_frame, on="case_id", how="left")
