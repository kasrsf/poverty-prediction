"""Poverty rate calculation and calibration."""

import numpy as np
import pandas as pd
from scipy import stats

from config import POVERTY_THRESHOLDS, POVERTY_RATE_COLS, THRESHOLD_TO_STR
from metrics import calculate_poverty_rates, validate_rates


def calculate_survey_rates(
    predictions: np.ndarray,
    weights: np.ndarray,
    survey_ids: np.ndarray,
    thresholds: list[float] | None = None
) -> pd.DataFrame:
    """Calculate poverty rates for each survey.

    Args:
        predictions: Array of household consumption predictions
        weights: Array of household weights
        survey_ids: Array of survey IDs
        thresholds: List of poverty thresholds

    Returns:
        DataFrame with poverty rates by survey
    """
    if thresholds is None:
        thresholds = POVERTY_THRESHOLDS

    unique_surveys = np.unique(survey_ids)
    results = []

    for survey in unique_surveys:
        mask = survey_ids == survey
        survey_preds = predictions[mask]
        survey_weights = weights[mask]

        rates = calculate_poverty_rates(survey_preds, survey_weights, thresholds)

        row = {"survey_id": int(survey)}
        for t in thresholds:
            # Use string format for column names to preserve trailing zeros
            t_str = THRESHOLD_TO_STR.get(t, str(t))
            row[f"pct_hh_below_{t_str}"] = rates[t]

        results.append(row)

    return pd.DataFrame(results)


class QuantileCalibrator:
    """Calibrate predictions to match training distribution quantiles."""

    def __init__(self):
        self.train_quantiles = None
        self.n_quantiles = 100

    def fit(self, y_train: np.ndarray, weights: np.ndarray | None = None) -> "QuantileCalibrator":
        """Fit calibrator on training data.

        Args:
            y_train: Training target values
            weights: Optional sample weights
        """
        # Calculate weighted quantiles
        percentiles = np.linspace(0, 100, self.n_quantiles + 1)

        if weights is not None:
            # Weighted quantile calculation
            sorted_idx = np.argsort(y_train)
            sorted_y = y_train[sorted_idx]
            sorted_w = weights[sorted_idx]

            cum_weights = np.cumsum(sorted_w)
            cum_weights = cum_weights / cum_weights[-1]

            self.train_quantiles = np.interp(
                percentiles / 100, cum_weights, sorted_y
            )
        else:
            self.train_quantiles = np.percentile(y_train, percentiles)

        return self

    def transform(self, predictions: np.ndarray) -> np.ndarray:
        """Transform predictions to match training distribution.

        Args:
            predictions: Model predictions

        Returns:
            Calibrated predictions
        """
        if self.train_quantiles is None:
            raise ValueError("Calibrator must be fitted first")

        # Calculate prediction quantiles
        pred_quantiles = np.percentile(predictions, np.linspace(0, 100, self.n_quantiles + 1))

        # Map predictions to training distribution
        calibrated = np.interp(predictions, pred_quantiles, self.train_quantiles)

        return calibrated


class ThresholdCalibrator:
    """Calibrate predictions around specific thresholds."""

    def __init__(self, thresholds: list[float] | None = None):
        self.thresholds = thresholds or POVERTY_THRESHOLDS
        self.adjustments = {}

    def fit(
        self,
        y_train: np.ndarray,
        y_pred: np.ndarray,
        weights: np.ndarray
    ) -> "ThresholdCalibrator":
        """Fit threshold-specific adjustments.

        Args:
            y_train: True target values
            y_pred: Predicted values
            weights: Sample weights
        """
        true_rates = calculate_poverty_rates(y_train, weights, self.thresholds)
        pred_rates = calculate_poverty_rates(y_pred, weights, self.thresholds)

        for t in self.thresholds:
            # Calculate rate difference
            rate_diff = true_rates[t] - pred_rates[t]
            self.adjustments[t] = rate_diff

        return self

    def transform(
        self,
        predictions: np.ndarray,
        target_rates: dict[float, float] | None = None
    ) -> np.ndarray:
        """Apply threshold calibration.

        This is a soft adjustment that shifts predictions
        to better match target rates.

        Args:
            predictions: Model predictions
            target_rates: Optional target rates to calibrate towards

        Returns:
            Calibrated predictions
        """
        # Simple approach: apply a small multiplicative adjustment
        # based on the learned rate differences
        calibrated = predictions.copy()

        # Calculate current rates (equal weights for simplicity)
        weights = np.ones(len(predictions))
        current_rates = calculate_poverty_rates(calibrated, weights, self.thresholds)

        # Find the threshold closest to 40% rate (most important)
        key_threshold = min(self.thresholds, key=lambda t: abs(current_rates[t] - 0.4))

        # Apply adjustment
        if key_threshold in self.adjustments:
            adjustment_factor = 1 - self.adjustments[key_threshold] * 0.1
            calibrated = calibrated * adjustment_factor

        return calibrated


def ensure_monotonic_rates(rates: dict[float, float]) -> dict[float, float]:
    """Ensure poverty rates are monotonically increasing.

    Args:
        rates: Dictionary of poverty rates by threshold

    Returns:
        Adjusted rates that are monotonically increasing
    """
    sorted_thresholds = sorted(rates.keys())
    adjusted_rates = {}

    prev_rate = 0
    for t in sorted_thresholds:
        current_rate = max(rates[t], prev_rate)
        adjusted_rates[t] = current_rate
        prev_rate = current_rate

    # Ensure last rate doesn't exceed 1
    for t in sorted_thresholds:
        adjusted_rates[t] = min(adjusted_rates[t], 1.0)

    return adjusted_rates


def create_poverty_submission(
    rates_df: pd.DataFrame,
    ensure_monotonic: bool = True
) -> pd.DataFrame:
    """Format poverty rates for submission.

    Args:
        rates_df: DataFrame with survey_id and rate columns
        ensure_monotonic: Whether to ensure rates are monotonic

    Returns:
        DataFrame formatted for submission
    """
    submission = rates_df.copy()

    if ensure_monotonic:
        for idx, row in submission.iterrows():
            # Extract rates using the actual column names
            rates = {}
            for col in POVERTY_RATE_COLS:
                t = float(col.replace("pct_hh_below_", ""))
                rates[t] = row[col]
            adjusted = ensure_monotonic_rates(rates)
            for t, rate in adjusted.items():
                # Use string format for column names
                t_str = THRESHOLD_TO_STR.get(t, str(t))
                submission.loc[idx, f"pct_hh_below_{t_str}"] = rate

    # Ensure correct column order
    submission = submission[["survey_id"] + POVERTY_RATE_COLS]

    return submission


if __name__ == "__main__":
    from data_loader import load_all_data
    from preprocessing import preprocess_data
    from features import create_features, get_feature_columns
    from models import LightGBMModel

    print("Loading and preprocessing data...")
    data = load_all_data()

    train_processed, test_processed, _ = preprocess_data(
        data["train_features"], data["test_features"]
    )
    train_features = create_features(train_processed)
    feature_cols = get_feature_columns(train_features)

    # Train a quick model
    print("\nTraining model...")
    model = LightGBMModel(n_estimators=500)
    model.fit(train_features, data["train_target"], feature_cols)

    # Predict on training data
    train_preds = model.predict(train_features)

    # Calculate rates
    print("\nCalculating poverty rates...")
    rates_df = calculate_survey_rates(
        train_preds,
        train_features["weight"].values,
        train_features["survey_id"].values
    )

    print("\nPredicted rates (first survey):")
    print(rates_df.iloc[0])

    print("\nGround truth rates (first survey):")
    print(data["train_rates"].iloc[0])

    # Test calibration
    print("\n" + "=" * 50)
    print("Testing quantile calibration...")

    calibrator = QuantileCalibrator()
    calibrator.fit(data["train_target"].values, train_features["weight"].values)

    calibrated_preds = calibrator.transform(train_preds)

    print(f"Original prediction range: {train_preds.min():.2f} - {train_preds.max():.2f}")
    print(f"Calibrated prediction range: {calibrated_preds.min():.2f} - {calibrated_preds.max():.2f}")
    print(f"Training target range: {data['train_target'].min():.2f} - {data['train_target'].max():.2f}")

    # Calculate rates with calibrated predictions
    calibrated_rates = calculate_survey_rates(
        calibrated_preds,
        train_features["weight"].values,
        train_features["survey_id"].values
    )

    print("\nCalibrated rates (first survey):")
    print(calibrated_rates.iloc[0])

    # Check monotonicity
    for _, row in rates_df.iterrows():
        rates = {}
        for c in POVERTY_RATE_COLS:
            t = float(c.replace("pct_hh_below_", ""))
            rates[t] = row[c]
        is_mono = validate_rates(rates)
        print(f"Survey {int(row['survey_id'])} monotonic: {is_mono}")
