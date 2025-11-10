from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def test_scoring_outputs_are_valid() -> None:
    # Locate the trained model
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "linear_regression_pipeline.joblib"

    pipeline = joblib.load(model_path)

    # Example inputs provided in the assignment
    df = pd.DataFrame({"eruptions": [1.5, 2.0, 3.0]})
    preds = pipeline.predict(df[["eruptions"]])

    # a) Number of predictions must match number of inputs
    assert len(preds) == len(df), "Number of predictions does not match input rows."

    # b) All predicted values must be finite (not NaN or inf)
    assert np.isfinite(preds).all(), "Predictions contain non-finite values."

    # c) All predicted values must be positive (> 0)
    assert (preds > 0).all(), "Some predictions are not positive."
