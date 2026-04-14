from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class ImagePair:
    image: sitk.Image
    mask: sitk.Image


def read_image(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def read_pair(image_path: Path, mask_path: Path) -> ImagePair:
    return ImagePair(image=read_image(image_path), mask=read_image(mask_path))


def count_mask_voxels(mask: sitk.Image, label: int = 1) -> int:
    array = sitk.GetArrayViewFromImage(mask)
    return int(np.count_nonzero(array == label))


def geometry_matches(image: sitk.Image, mask: sitk.Image, tolerance: float = 1e-6) -> bool:
    if image.GetSize() != mask.GetSize():
        return False

    checks = (
        (image.GetSpacing(), mask.GetSpacing()),
        (image.GetOrigin(), mask.GetOrigin()),
        (image.GetDirection(), mask.GetDirection()),
    )
    for left, right in checks:
        if len(left) != len(right):
            return False
        if any(abs(a - b) > tolerance for a, b in zip(left, right)):
            return False
    return True


def image_metadata(image: sitk.Image) -> dict[str, object]:
    return {
        "size": list(image.GetSize()),
        "spacing": list(image.GetSpacing()),
        "origin": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
        "pixel_id": int(image.GetPixelID()),
    }
