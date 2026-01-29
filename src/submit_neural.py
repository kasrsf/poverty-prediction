"""Generate submission using neural network models."""

import numpy as np
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path
import glob
import argparse

from data_loader import load_all_data
from preprocessing import preprocess_data, Preprocessor
from features import create_features, get_feature_columns
from neural_models import (
    NeuralModelWrapper,
    MixtureDensityNetwork,
    ConsumptionPredictor,
    DirectRatePredictor
)
from poverty_rates import (
    calculate_survey_rates,
    create_poverty_submission,
    SurveySpecificCalibrator,
    RateMatchingCalibrator
)
from metrics import validate_rates
from config import (
    MODELS_DIR,
    SUBMISSIONS_DIR,
    POVERTY_RATE_COLS,
    POVERTY_THRESHOLDS,
    TEST_SURVEYS,
)


def load_neural_models(model_type: str = "mdn") -> list[NeuralModelWrapper]:
    """Load trained neural network models.

    Args:
        model_type: Type of models to load ('mdn', 'consumption', 'direct_rate')

    Returns:
        List of loaded model wrappers
    """
    pattern = str(MODELS_DIR / f"{model_type}_seed*.pkl")
    model_files = glob.glob(pattern)

    if not model_files:
        raise FileNotFoundError(f"No {model_type} models found in {MODELS_DIR}")

    models = []
    for path in sorted(model_files):
        try:
            model = NeuralModelWrapper.load(path)
            models.append(model)
            print(f"  Loaded: {Path(path).name}")
        except Exception as e:
            print(f"  Warning: Could not load {path}: {e}")

    return models


def ensemble_neural_predictions(
    models: list[NeuralModelWrapper],
    test_features: pd.DataFrame,
    method: str = "mean"
) -> np.ndarray:
    """Generate ensemble predictions from multiple neural models.

    Args:
        models: List of trained models
        test_features: Test features
        method: Ensemble method ('mean', 'median')

    Returns:
        Ensemble predictions
    """
    all_preds = []

    for model in models:
        preds = model.predict(test_features)
        all_preds.append(preds)

    all_preds = np.array(all_preds)

    if method == "median":
        return np.median(all_preds, axis=0)
    else:
        return np.mean(all_preds, axis=0)


def generate_neural_submission(
    models: list[NeuralModelWrapper] | None = None,
    train_features: pd.DataFrame | None = None,
    test_features: pd.DataFrame | None = None,
    train_target: pd.Series | None = None,
    train_rates: pd.DataFrame | None = None,
    model_type: str = "mdn",
    ensemble_method: str = "mean",
    use_calibration: bool = True,
    output_dir: Path | None = None,
    tag: str = ""
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate submission using neural network models.

    Args:
        models: List of trained neural models
        train_features: Training features
        test_features: Test features
        train_target: Training target
        train_rates: Ground truth poverty rates
        model_type: Type of neural models
        ensemble_method: How to combine predictions
        use_calibration: Apply survey calibration
        output_dir: Output directory
        tag: Tag for files

    Returns:
        Tuple of (household_df, rates_df)
    """
    if output_dir is None:
        output_dir = SUBMISSIONS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data if not provided
    if train_features is None or test_features is None:
        print("Loading data...")
        data = load_all_data()

        preprocessor = Preprocessor.load()
        train_processed = preprocessor.transform(data["train_features"])
        test_processed = preprocessor.transform(data["test_features"])

        train_features = create_features(train_processed)
        test_features = create_features(test_processed)
        train_target = data["train_target"]
        train_rates = data["train_rates"]

    feature_cols = get_feature_columns(train_features)

    # Load models if not provided
    if models is None:
        print(f"Loading {model_type} models...")
        try:
            models = load_neural_models(model_type)
        except FileNotFoundError:
            print(f"No trained models found. Training new {model_type} models...")
            from train_neural import train_neural_ensemble

            results = train_neural_ensemble(
                train_features,
                train_target,
                feature_cols,
                train_rates,
                model_type=model_type,
                n_seeds=5,
                n_epochs=100,
                run_validation=True,
                verbose=True
            )
            models = [r["model"] for r in results]

    print(f"Using {len(models)} {model_type} models")

    # Generate ensemble predictions
    print(f"Generating predictions (method: {ensemble_method})...")
    test_preds = ensemble_neural_predictions(models, test_features, ensemble_method)

    # Apply calibration
    if use_calibration:
        print("Applying survey calibration...")
        calibrator = SurveySpecificCalibrator()
        calibrator.fit(
            train_features,
            train_target.values,
            train_features["weight"].values,
            train_rates,
            feature_cols
        )

        test_preds = calibrator.transform_by_survey(
            test_preds,
            test_features,
            test_features["weight"].values,
            test_features["survey_id"].values
        )

    # Clip predictions
    train_min, train_max = train_target.min(), train_target.max()
    test_preds = np.clip(test_preds, train_min * 0.5, train_max * 1.5)

    print(f"Prediction range: {test_preds.min():.2f} - {test_preds.max():.2f}")
    print(f"Training range: {train_min:.2f} - {train_max:.2f}")

    # Create household consumption submission
    household_df = pd.DataFrame({
        "survey_id": test_features["survey_id"].astype(int),
        "hhid": test_features["hhid"].astype(int),
        "cons_ppp17": test_preds
    })
    household_df = household_df.sort_values(["survey_id", "hhid"]).reset_index(drop=True)

    # Calculate poverty rates
    print("Calculating poverty rates...")
    rates_df = calculate_survey_rates(
        test_preds,
        test_features["weight"].values,
        test_features["survey_id"].values
    )

    # Format and validate
    rates_df = create_poverty_submission(rates_df, ensure_monotonic=True)

    # Validate rates
    print("\nValidating poverty rates:")
    for _, row in rates_df.iterrows():
        survey_id = int(row["survey_id"])
        rates = {}
        for c in POVERTY_RATE_COLS:
            t = float(c.replace("pct_hh_below_", ""))
            rates[t] = row[c]
        is_valid = validate_rates(rates)
        print(f"  Survey {survey_id}: monotonic={is_valid}, "
              f"range=[{rates[3.17]:.4f}, {rates[27.37]:.4f}]")

    # Save submission files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_str = f"_{tag}" if tag else ""

    household_path = output_dir / f"predicted_household_consumption{tag_str}_{timestamp}.csv"
    rates_path = output_dir / f"predicted_poverty_distribution{tag_str}_{timestamp}.csv"

    household_df.to_csv(household_path, index=False)
    rates_df.to_csv(rates_path, index=False)

    print(f"\nSubmission files saved:")
    print(f"  {household_path}")
    print(f"  {rates_path}")

    # Print summary statistics
    print("\nSubmission summary:")
    print(f"  Total households: {len(household_df)}")
    for survey in TEST_SURVEYS:
        n = (household_df["survey_id"] == survey).sum()
        print(f"  Survey {survey}: {n} households")

    return household_df, rates_df


def validate_submission_format(
    household_df: pd.DataFrame,
    rates_df: pd.DataFrame
) -> bool:
    """Validate submission format."""
    valid = True

    expected_hh_cols = ["survey_id", "hhid", "cons_ppp17"]
    if list(household_df.columns) != expected_hh_cols:
        print(f"ERROR: Household columns mismatch")
        valid = False

    expected_rate_cols = ["survey_id"] + POVERTY_RATE_COLS
    if list(rates_df.columns) != expected_rate_cols:
        print(f"ERROR: Rates columns mismatch")
        valid = False

    hh_surveys = set(household_df["survey_id"].unique())
    rate_surveys = set(rates_df["survey_id"].unique())
    expected_surveys = set(TEST_SURVEYS)

    if hh_surveys != expected_surveys or rate_surveys != expected_surveys:
        print(f"ERROR: Survey mismatch")
        valid = False

    if household_df["cons_ppp17"].isna().any():
        print("ERROR: NaN values in predictions")
        valid = False

    if valid:
        print("Submission format validation: PASSED")

    return valid


def main(
    model_type: str = "mdn",
    ensemble_method: str = "mean",
    use_calibration: bool = True,
    tag: str = ""
):
    """Main neural network submission pipeline.

    Args:
        model_type: Type of neural models
        ensemble_method: How to combine predictions
        use_calibration: Apply survey calibration
        tag: Tag for submission files
    """
    print("=" * 60)
    print(f"NEURAL NETWORK SUBMISSION ({model_type})")
    print("=" * 60)

    # Generate submission
    household_df, rates_df = generate_neural_submission(
        model_type=model_type,
        ensemble_method=ensemble_method,
        use_calibration=use_calibration,
        tag=tag
    )

    # Validate format
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    validate_submission_format(household_df, rates_df)

    return household_df, rates_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate neural network submission")
    parser.add_argument(
        "--model-type",
        type=str,
        default="mdn",
        choices=["mdn", "consumption", "direct_rate"],
        help="Type of neural models"
    )
    parser.add_argument(
        "--ensemble-method",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="How to combine predictions"
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Skip survey calibration"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="v3_neural",
        help="Tag for submission files"
    )

    args = parser.parse_args()

    main(
        model_type=args.model_type,
        ensemble_method=args.ensemble_method,
        use_calibration=not args.no_calibration,
        tag=args.tag
    )
