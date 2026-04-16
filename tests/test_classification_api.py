from __future__ import annotations

import importlib
import unittest
import warnings
from unittest import mock

from classification import train_and_evaluate, mean_roc_plot


class ClassificationApiTest(unittest.TestCase):
    def test_import_smoke(self) -> None:
        canonical_package = importlib.import_module("classification")
        legacy_package = importlib.import_module("Classification")

        from classification import Model, mean_roc_plot as legacy_mean_roc_plot

        self.assertIs(legacy_package, canonical_package)
        self.assertTrue(callable(train_and_evaluate))
        self.assertTrue(callable(mean_roc_plot))
        self.assertTrue(callable(Model))
        self.assertTrue(callable(legacy_mean_roc_plot))

    def test_legacy_wrappers_call_canonical_functions_with_warning(self) -> None:
        package = importlib.import_module("classification")
        cases = [
            ("Model", "build_model", ("SVM", None, None)),
            ("DrawROC", "draw_roc", tuple()),
            ("Mean_roc_plot", "mean_roc_plot", tuple()),
            ("createNewMask", "create_new_mask", ("mask.nii.gz", "output")),
        ]

        for wrapper_name, target_name, args in cases:
            with self.subTest(wrapper_name=wrapper_name):
                sentinel = object()
                with mock.patch.object(package, target_name, return_value=sentinel) as patched:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        result = getattr(package, wrapper_name)(*args)

                self.assertIs(result, sentinel)
                patched.assert_called_once()
                self.assertTrue(caught)
                self.assertTrue(any(item.category is DeprecationWarning for item in caught))

    def test_all_exports_canonical_surface(self) -> None:
        package = importlib.import_module("classification")

        expected_exports = {
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
        }

        self.assertTrue(expected_exports.issubset(set(package.__all__)))
        self.assertNotIn("Model", package.__all__)
        self.assertNotIn("DrawROC", package.__all__)
        self.assertNotIn("Mean_roc_plot", package.__all__)
        self.assertNotIn("createNewMask", package.__all__)


if __name__ == "__main__":
    unittest.main()
