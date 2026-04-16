"""Utilities and interfaces for radiomics classification experiments."""

from __future__ import annotations

import importlib
import sys
import warnings

from .examples import ExampleDatasetPaths, build_test_data_examples
from .interfaces import (
    NON_FEATURE_COLUMNS,
    available_models,
    available_selection_methods,
    build_model,
    build_models,
    create_new_mask,
    draw_roc,
    load_labels,
    mean_roc_plot,
    predict_and_evaluate,
    prepare_training_data,
    select_features,
    train_and_evaluate,
)

_LEGACY_MODULE_ALIASES = {
    "Classification": "classification",
    "Classification.examples": "classification.examples",
    "Classification.interfaces": "classification.interfaces",
    "Classification.Model": "classification.model_factory",
    "Classification.Remake_mask": "classification.mask_rebuild",
    "Classification.Roc_plot": "classification.roc_plot",
}


def _register_module_aliases() -> None:
    current_module = sys.modules[__name__]
    sys.modules.setdefault("classification", current_module)
    sys.modules.setdefault("Classification", current_module)

    for legacy_name, canonical_name in _LEGACY_MODULE_ALIASES.items():
        if legacy_name in sys.modules:
            continue
        sys.modules[legacy_name] = importlib.import_module(canonical_name)


def _warn_legacy_name(legacy_name: str, replacement_name: str) -> None:
    warnings.warn(
        f"{legacy_name} is deprecated; use {replacement_name} instead.",
        DeprecationWarning,
        stacklevel=2,
    )


def Model(*args, **kwargs):
    _warn_legacy_name("Classification.Model", "classification.build_model")
    return build_model(*args, **kwargs)


def DrawROC(*args, **kwargs):
    _warn_legacy_name("Classification.DrawROC", "classification.draw_roc")
    return draw_roc(*args, **kwargs)


def Mean_roc_plot(*args, **kwargs):
    _warn_legacy_name("Classification.Mean_roc_plot", "classification.mean_roc_plot")
    return mean_roc_plot(*args, **kwargs)


def createNewMask(*args, **kwargs):
    _warn_legacy_name("Classification.createNewMask", "classification.create_new_mask")
    return create_new_mask(*args, **kwargs)


_register_module_aliases()

__all__ = [
    "ExampleDatasetPaths",
    "NON_FEATURE_COLUMNS",
    "available_models",
    "available_selection_methods",
    "build_model",
    "build_models",
    "build_test_data_examples",
    "create_new_mask",
    "draw_roc",
    "load_labels",
    "mean_roc_plot",
    "predict_and_evaluate",
    "prepare_training_data",
    "select_features",
    "train_and_evaluate",
]
