from __future__ import annotations

from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor
from Classification import build_test_data_examples


def load_tabular_model(model_dir: str | Path = "agModels-predictClass") -> TabularPredictor:
    return TabularPredictor.load(str(model_dir))


def predict_from_csv(model_dir: str | Path, csv_path: str | Path, label_column: str = "label"):
    predictor = load_tabular_model(model_dir)
    frame = pd.read_csv(csv_path)
    truth = frame[label_column] if label_column in frame.columns else None
    features = frame.drop(columns=[label_column], errors="ignore")
    predictions = predictor.predict(features)
    return truth, predictions


if __name__ == "__main__":
    workspace = Path(__file__).resolve().parents[1]
    examples = build_test_data_examples(workspace)
    truth, predictions = predict_from_csv("agModels-predictClass", examples.features)
    if truth is not None:
        print("Truth:\n", truth)
    print("Predictions:\n", predictions)
