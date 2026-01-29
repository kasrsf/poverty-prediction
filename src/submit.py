"""Generate submission files."""

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from data_loader import load_all_data
from preprocessing import preprocess_data, Preprocessor
from features import create_features, get_feature_columns
from models import (
    BaseModelWrapper, LightGBMModel, XGBoostModel, CatBoostModel,
    QuantileModelWrapper, QuantileCatBoost
)
from ensemble import (
    Ensemble, QuantileEnsemble, MedianBlendingEnsemble, HybridEnsemble,
    optimize_ensemble_weights, optimize_quantile_ensemble_weights
)
from poverty_rates import (
    calculate_survey_rates, create_poverty_submission, QuantileCalibrator,
    SurveySpecificCalibrator, RateMatchingCalibrator
)
from rate_optimizer import RateOptimizer, optimize_for_survey
from metrics import validate_rates, get_true_rates_for_survey
from config import (
    MODELS_DIR,
    SUBMISSIONS_DIR,
    POVERTY_RATE_COLS,
    POVERTY_THRESHOLDS,
    TEST_SURVEYS,
    TRAIN_SURVEYS,
)


def generate_submission(
    ensemble: Ensemble | None = None,
    train_features: pd.DataFrame | None = None,
    test_features: pd.DataFrame | None = None,
    train_target: pd.Series | None = None,
    calibrate: bool = True,
    output_dir: Path | None = None,
    tag: str = ""
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate submission files.

    Args:
        ensemble: Trained ensemble (loads from disk if None)
        train_features: Training features (loads if None)
        test_features: Test features (loads if None)
        train_target: Training target (loads if None)
        calibrate: Whether to apply quantile calibration
        output_dir: Output directory for submission files
        tag: Optional tag for submission files

    Returns:
        Tuple of (household_predictions_df, poverty_rates_df)
    """
    if output_dir is None:
        output_dir = SUBMISSIONS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data if not provided
    if train_features is None or test_features is None or train_target is None:
        print("Loading data...")
        data = load_all_data()

        preprocessor = Preprocessor.load()
        train_processed = preprocessor.transform(data["train_features"])
        test_processed = preprocessor.transform(data["test_features"])

        train_features = create_features(train_processed)
        test_features = create_features(test_processed)
        train_target = data["train_target"]

    # Load ensemble if not provided
    if ensemble is None:
        print("Loading ensemble...")
        ensemble = Ensemble.load(MODELS_DIR / "ensemble.pkl")

    # Generate predictions
    print("Generating predictions...")
    test_preds = ensemble.predict(test_features)

    # Apply calibration if requested
    if calibrate:
        print("Applying quantile calibration...")
        calibrator = QuantileCalibrator()
        calibrator.fit(train_target.values, train_features["weight"].values)
        test_preds = calibrator.transform(test_preds)

    # Ensure predictions are in reasonable range
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

    # Sort by survey_id and hhid
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


def generate_quantile_submission(
    quantile_models: list[QuantileModelWrapper] | None = None,
    train_features: pd.DataFrame | None = None,
    test_features: pd.DataFrame | None = None,
    train_target: pd.Series | None = None,
    train_rates: pd.DataFrame | None = None,
    use_rate_optimization: bool = True,
    use_survey_calibration: bool = True,
    output_dir: Path | None = None,
    tag: str = ""
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate submission using quantile regression models.

    Args:
        quantile_models: List of trained quantile models
        train_features: Training features
        test_features: Test features
        train_target: Training target
        train_rates: Ground truth poverty rates
        use_rate_optimization: Apply rate optimization
        use_survey_calibration: Apply survey-specific calibration
        output_dir: Output directory
        tag: Tag for files

    Returns:
        Tuple of (household_df, rates_df)
    """
    if output_dir is None:
        output_dir = SUBMISSIONS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data if not provided
    if train_features is None or test_features is None or train_target is None:
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

    # Load quantile models if not provided
    if quantile_models is None:
        print("Loading quantile models...")
        try:
            model = QuantileModelWrapper.load(MODELS_DIR / "quantile_catboost.pkl")
            quantile_models = [model]
        except FileNotFoundError:
            print("No quantile models found. Training new quantile model...")
            from train import train_quantile_models

            results = train_quantile_models(
                train_features,
                train_target,
                feature_cols,
                train_rates,
                n_estimators=1000,
                model_types=["catboost"],
                run_validation=True,
                verbose=True
            )
            quantile_models = [r["model"] for r in results]

    # Create quantile ensemble
    quantile_ensemble = QuantileEnsemble(quantile_models)

    # Generate predictions (use median)
    print("Generating quantile predictions...")
    test_preds = quantile_ensemble.predict_median(test_features)

    # Apply survey-specific calibration
    if use_survey_calibration:
        print("Applying survey-specific calibration...")
        survey_calibrator = SurveySpecificCalibrator()
        survey_calibrator.fit(
            train_features,
            train_target.values,
            train_features["weight"].values,
            train_rates,
            feature_cols
        )

        test_preds = survey_calibrator.transform_by_survey(
            test_preds,
            test_features,
            test_features["weight"].values,
            test_features["survey_id"].values
        )

    # Apply rate optimization per survey
    if use_rate_optimization:
        print("Applying rate optimization...")
        rate_matcher = RateMatchingCalibrator()
        rate_matcher.fit(train_rates)

        for survey_id in TEST_SURVEYS:
            mask = test_features["survey_id"] == survey_id
            survey_preds = test_preds[mask]
            survey_weights = test_features.loc[mask, "weight"].values

            # Use average training pattern as target
            test_preds[mask] = rate_matcher.transform(
                survey_preds,
                survey_weights,
                target_pattern="average"
            )

    # Ensure predictions are in reasonable range
    train_min, train_max = train_target.min(), train_target.max()
    test_preds = np.clip(test_preds, train_min * 0.5, train_max * 1.5)

    print(f"Prediction range: {test_preds.min():.2f} - {test_preds.max():.2f}")

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

    return household_df, rates_df


def generate_hybrid_submission(
    point_models: list[BaseModelWrapper] | None = None,
    quantile_models: list[QuantileModelWrapper] | None = None,
    train_features: pd.DataFrame | None = None,
    test_features: pd.DataFrame | None = None,
    train_target: pd.Series | None = None,
    train_rates: pd.DataFrame | None = None,
    point_weight: float = 0.5,
    output_dir: Path | None = None,
    tag: str = ""
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate submission using hybrid ensemble (point + quantile).

    Args:
        point_models: Point prediction models
        quantile_models: Quantile regression models
        train_features: Training features
        test_features: Test features
        train_target: Training target
        train_rates: Ground truth rates
        point_weight: Weight for point predictions (0-1)
        output_dir: Output directory
        tag: Tag for files

    Returns:
        Tuple of (household_df, rates_df)
    """
    if output_dir is None:
        output_dir = SUBMISSIONS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data if needed
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

    # Load models if needed
    if point_models is None:
        try:
            ensemble = Ensemble.load(MODELS_DIR / "ensemble.pkl")
            point_models = ensemble.models
        except FileNotFoundError:
            print("Training point models...")
            from train import train_all_models

            results = train_all_models(
                train_features, train_target, feature_cols, train_rates,
                n_estimators=1000, seeds=[42], run_validation=False, verbose=True
            )
            point_models = [r["model"] for r in results]

    if quantile_models is None:
        try:
            qmodel = QuantileModelWrapper.load(MODELS_DIR / "quantile_catboost.pkl")
            quantile_models = [qmodel]
        except FileNotFoundError:
            print("Training quantile model...")
            from train import train_quantile_models

            results = train_quantile_models(
                train_features, train_target, feature_cols, train_rates,
                n_estimators=1000, model_types=["catboost"], run_validation=False, verbose=True
            )
            quantile_models = [r["model"] for r in results]

    # Create hybrid ensemble
    print(f"Creating hybrid ensemble (point_weight={point_weight})...")
    hybrid = HybridEnsemble(point_models, quantile_models, point_weight)

    # Generate predictions
    print("Generating predictions...")
    test_preds = hybrid.predict(test_features)

    # Clip predictions
    train_min, train_max = train_target.min(), train_target.max()
    test_preds = np.clip(test_preds, train_min * 0.5, train_max * 1.5)

    print(f"Prediction range: {test_preds.min():.2f} - {test_preds.max():.2f}")

    # Create submissions
    household_df = pd.DataFrame({
        "survey_id": test_features["survey_id"].astype(int),
        "hhid": test_features["hhid"].astype(int),
        "cons_ppp17": test_preds
    })
    household_df = household_df.sort_values(["survey_id", "hhid"]).reset_index(drop=True)

    rates_df = calculate_survey_rates(
        test_preds,
        test_features["weight"].values,
        test_features["survey_id"].values
    )
    rates_df = create_poverty_submission(rates_df, ensure_monotonic=True)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_str = f"_{tag}" if tag else ""

    household_path = output_dir / f"predicted_household_consumption{tag_str}_{timestamp}.csv"
    rates_path = output_dir / f"predicted_poverty_distribution{tag_str}_{timestamp}.csv"

    household_df.to_csv(household_path, index=False)
    rates_df.to_csv(rates_path, index=False)

    print(f"\nSubmission files saved:")
    print(f"  {household_path}")
    print(f"  {rates_path}")

    return household_df, rates_df


def validate_submission_format(
    household_df: pd.DataFrame,
    rates_df: pd.DataFrame
) -> bool:
    """Validate submission format matches expected structure.

    Args:
        household_df: Household consumption predictions
        rates_df: Poverty rate predictions

    Returns:
        True if format is valid
    """
    valid = True

    # Check household columns
    expected_hh_cols = ["survey_id", "hhid", "cons_ppp17"]
    if list(household_df.columns) != expected_hh_cols:
        print(f"ERROR: Household columns mismatch. Expected {expected_hh_cols}, got {list(household_df.columns)}")
        valid = False

    # Check rates columns
    expected_rate_cols = ["survey_id"] + POVERTY_RATE_COLS
    if list(rates_df.columns) != expected_rate_cols:
        print(f"ERROR: Rates columns mismatch. Expected {expected_rate_cols}, got {list(rates_df.columns)}")
        valid = False

    # Check surveys
    hh_surveys = set(household_df["survey_id"].unique())
    rate_surveys = set(rates_df["survey_id"].unique())
    expected_surveys = set(TEST_SURVEYS)

    if hh_surveys != expected_surveys:
        print(f"ERROR: Household surveys mismatch. Expected {expected_surveys}, got {hh_surveys}")
        valid = False

    if rate_surveys != expected_surveys:
        print(f"ERROR: Rates surveys mismatch. Expected {expected_surveys}, got {rate_surveys}")
        valid = False

    # Check for NaN
    if household_df["cons_ppp17"].isna().any():
        print("ERROR: NaN values in household predictions")
        valid = False

    if rates_df[POVERTY_RATE_COLS].isna().any().any():
        print("ERROR: NaN values in poverty rates")
        valid = False

    # Check for negative values
    if (household_df["cons_ppp17"] <= 0).any():
        print("WARNING: Non-positive values in household predictions")

    if (rates_df[POVERTY_RATE_COLS] < 0).any().any():
        print("ERROR: Negative values in poverty rates")
        valid = False

    if (rates_df[POVERTY_RATE_COLS] > 1).any().any():
        print("ERROR: Values > 1 in poverty rates")
        valid = False

    if valid:
        print("Submission format validation: PASSED")

    return valid


def main(
    calibrate: bool = True,
    tag: str = "",
    mode: str = "standard",
    use_rate_optimization: bool = True,
    use_survey_calibration: bool = True,
    point_weight: float = 0.5
):
    """Main submission generation pipeline.

    Args:
        calibrate: Whether to apply quantile calibration
        tag: Optional tag for submission files
        mode: Submission mode ('standard', 'quantile', 'hybrid')
        use_rate_optimization: Apply rate optimization (quantile mode)
        use_survey_calibration: Apply survey calibration (quantile mode)
        point_weight: Weight for point predictions in hybrid mode
    """
    # Load all data and preprocess
    print("=" * 60)
    print(f"SUBMISSION GENERATION (mode: {mode})")
    print("=" * 60)

    data = load_all_data()

    train_processed, test_processed, preprocessor = preprocess_data(
        data["train_features"], data["test_features"]
    )

    train_features = create_features(train_processed)
    test_features = create_features(test_processed)
    feature_cols = get_feature_columns(train_features)

    if mode == "quantile":
        # Quantile regression submission
        household_df, rates_df = generate_quantile_submission(
            train_features=train_features,
            test_features=test_features,
            train_target=data["train_target"],
            train_rates=data["train_rates"],
            use_rate_optimization=use_rate_optimization,
            use_survey_calibration=use_survey_calibration,
            tag=tag
        )

    elif mode == "hybrid":
        # Hybrid ensemble submission
        household_df, rates_df = generate_hybrid_submission(
            train_features=train_features,
            test_features=test_features,
            train_target=data["train_target"],
            train_rates=data["train_rates"],
            point_weight=point_weight,
            tag=tag
        )

    else:
        # Standard submission
        try:
            ensemble = Ensemble.load(MODELS_DIR / "ensemble.pkl")
            print("Loaded existing ensemble.")
        except FileNotFoundError:
            print("No existing ensemble found. Training new models...")

            from train import train_all_models

            results = train_all_models(
                train_features,
                data["train_target"],
                feature_cols,
                data["train_rates"],
                n_estimators=1000,
                seeds=[42, 123, 456],
                run_validation=True,
                verbose=True
            )

            models = [r["model"] for r in results]

            # Optimize weights
            optimal_weights, _ = optimize_ensemble_weights(
                models,
                train_features,
                data["train_target"],
                feature_cols,
                data["train_rates"],
                verbose=True
            )

            ensemble = Ensemble(models, optimal_weights)
            ensemble.save()

        # Generate submission
        household_df, rates_df = generate_submission(
            ensemble=ensemble,
            train_features=train_features,
            test_features=test_features,
            train_target=data["train_target"],
            calibrate=calibrate,
            tag=tag
        )

    # Validate format
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    validate_submission_format(household_df, rates_df)

    return household_df, rates_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate submission files")
    parser.add_argument("--no-calibrate", action="store_true", help="Skip quantile calibration")
    parser.add_argument("--tag", type=str, default="", help="Tag for submission files")
    parser.add_argument(
        "--mode",
        type=str,
        default="standard",
        choices=["standard", "quantile", "hybrid"],
        help="Submission mode"
    )
    parser.add_argument(
        "--no-rate-optimization",
        action="store_true",
        help="Skip rate optimization (quantile mode)"
    )
    parser.add_argument(
        "--no-survey-calibration",
        action="store_true",
        help="Skip survey calibration (quantile mode)"
    )
    parser.add_argument(
        "--point-weight",
        type=float,
        default=0.5,
        help="Weight for point predictions in hybrid mode (0-1)"
    )

    args = parser.parse_args()

    main(
        calibrate=not args.no_calibrate,
        tag=args.tag,
        mode=args.mode,
        use_rate_optimization=not args.no_rate_optimization,
        use_survey_calibration=not args.no_survey_calibration,
        point_weight=args.point_weight
    )
