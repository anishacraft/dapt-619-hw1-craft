from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    # Find the project root (same idea as in simple_linear_demo.py)
    project_root = Path(__file__).resolve().parents[1]

    # Paths to data and model
    data_path = project_root / "data" / "raw" / "geyser.tsv"
    model_path = project_root / "models" / "linear_regression_pipeline.joblib"

    # Read the original geyser dataset (tab-separated)
    df = pd.read_csv(data_path, sep="\t")

    feature_column = "eruptions"
    target_column = "waiting"

    # Load the trained pipeline (scaler + linear regression)
    pipeline = joblib.load(model_path)

    # Make predictions based on the eruptions column
    preds = pipeline.predict(df[[feature_column]])

    # Prepare scored dataframe with two original columns + predictions
    scored_df = df[[feature_column, target_column]].copy()
    scored_df["predicted_waiting"] = preds

    # Ensure output directory exists
    scored_dir = project_root / "data" / "scored"
    scored_dir.mkdir(parents=True, exist_ok=True)

    # Save scored dataset (CSV is fine for output)
    output_path = scored_dir / "geyser_scored.csv"
    scored_df.to_csv(output_path, index=False)

    print(f"Saved scored data to: {output_path}")


if __name__ == "__main__":
    main()
