"""Poverty rate calculation and calibration."""

import numpy as np
import pandas as pd
from scipy import stats

from config import POVERTY_THRESHOLDS, POVERTY_RATE_COLS, THRESHOLD_TO_STR, TRAIN_SURVEYS
from metrics import calculate_poverty_rates, validate_rates, get_true_rates_for_survey


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


class SurveySpecificCalibrator:
    """Calibrate predictions using survey-specific distribution adjustments.

    This calibrator matches test surveys to similar training surveys and
    applies survey-specific corrections to improve poverty rate predictions.
    """

    def __init__(self, thresholds: list[float] | None = None):
        self.thresholds = thresholds or POVERTY_THRESHOLDS
        self.train_survey_stats = {}
        self.train_rates = {}
        self.feature_cols = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        weights_train: np.ndarray,
        rates_df: pd.DataFrame,
        feature_cols: list[str] | None = None
    ) -> "SurveySpecificCalibrator":
        """Fit calibrator on training data.

        Args:
            X_train: Training features (must have survey_id column)
            y_train: Training target values
            weights_train: Training sample weights
            rates_df: Ground truth poverty rates DataFrame
            feature_cols: Feature columns to use for similarity matching

        Returns:
            Self for chaining
        """
        self.feature_cols = feature_cols

        for survey_id in TRAIN_SURVEYS:
            mask = X_train["survey_id"] == survey_id
            survey_y = y_train[mask] if isinstance(y_train, np.ndarray) else y_train[mask].values
            survey_w = weights_train[mask] if isinstance(weights_train, np.ndarray) else weights_train[mask].values

            # Calculate distribution statistics
            self.train_survey_stats[survey_id] = {
                "mean": np.average(survey_y, weights=survey_w),
                "median": self._weighted_median(survey_y, survey_w),
                "std": np.sqrt(np.average((survey_y - np.average(survey_y, weights=survey_w))**2, weights=survey_w)),
                "skewness": stats.skew(survey_y),
                "p25": self._weighted_percentile(survey_y, survey_w, 25),
                "p75": self._weighted_percentile(survey_y, survey_w, 75),
                "p90": self._weighted_percentile(survey_y, survey_w, 90),
            }

            # Store true rates
            self.train_rates[survey_id] = get_true_rates_for_survey(rates_df, survey_id)

            # Store feature statistics if available
            if feature_cols is not None:
                survey_X = X_train.loc[mask, feature_cols]
                self.train_survey_stats[survey_id]["feature_means"] = survey_X.mean().values

        return self

    def _weighted_median(self, values: np.ndarray, weights: np.ndarray) -> float:
        """Calculate weighted median."""
        sorted_idx = np.argsort(values)
        sorted_values = values[sorted_idx]
        sorted_weights = weights[sorted_idx]
        cum_weights = np.cumsum(sorted_weights) / np.sum(sorted_weights)
        return sorted_values[np.searchsorted(cum_weights, 0.5)]

    def _weighted_percentile(self, values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
        """Calculate weighted percentile."""
        sorted_idx = np.argsort(values)
        sorted_values = values[sorted_idx]
        sorted_weights = weights[sorted_idx]
        cum_weights = np.cumsum(sorted_weights) / np.sum(sorted_weights)
        return sorted_values[np.searchsorted(cum_weights, percentile / 100)]

    def find_similar_survey(
        self,
        X_test: pd.DataFrame,
        predictions: np.ndarray,
        weights: np.ndarray
    ) -> int:
        """Find the training survey most similar to test data.

        Args:
            X_test: Test features for one survey
            predictions: Predictions for test survey
            weights: Sample weights

        Returns:
            Most similar training survey ID
        """
        # Calculate test survey statistics
        test_stats = {
            "mean": np.average(predictions, weights=weights),
            "median": self._weighted_median(predictions, weights),
            "std": np.sqrt(np.average((predictions - np.average(predictions, weights=weights))**2, weights=weights)),
            "skewness": stats.skew(predictions),
            "p25": self._weighted_percentile(predictions, weights, 25),
            "p75": self._weighted_percentile(predictions, weights, 75),
            "p90": self._weighted_percentile(predictions, weights, 90),
        }

        # Find most similar training survey
        best_survey = None
        best_distance = float("inf")

        for survey_id, train_stats in self.train_survey_stats.items():
            # Calculate similarity based on distribution statistics
            distance = 0
            for stat in ["mean", "median", "std", "p25", "p75", "p90"]:
                # Normalized difference
                if train_stats[stat] > 0:
                    distance += abs(test_stats[stat] - train_stats[stat]) / train_stats[stat]

            # Add feature-based similarity if available
            if self.feature_cols is not None and "feature_means" in train_stats:
                test_feature_means = X_test[self.feature_cols].mean().values
                feature_distance = np.mean(np.abs(test_feature_means - train_stats["feature_means"]))
                distance += feature_distance * 0.5

            if distance < best_distance:
                best_distance = distance
                best_survey = survey_id

        return best_survey

    def transform(
        self,
        predictions: np.ndarray,
        X_test: pd.DataFrame,
        weights: np.ndarray
    ) -> np.ndarray:
        """Apply survey-specific calibration.

        Args:
            predictions: Model predictions for one survey
            X_test: Test features for the survey
            weights: Sample weights

        Returns:
            Calibrated predictions
        """
        # Find similar training survey
        similar_survey = self.find_similar_survey(X_test, predictions, weights)

        # Get calibration parameters from similar survey
        similar_stats = self.train_survey_stats[similar_survey]
        similar_rates = self.train_rates[similar_survey]

        # Calculate current statistics
        pred_mean = np.average(predictions, weights=weights)
        pred_median = self._weighted_median(predictions, weights)

        # Apply mean/median-matching calibration
        target_mean = similar_stats["mean"]
        scale_factor = target_mean / pred_mean if pred_mean > 0 else 1.0

        # Constrain scale factor
        scale_factor = np.clip(scale_factor, 0.8, 1.2)

        calibrated = predictions * scale_factor

        # Additional adjustment for skewness (upper tail)
        if similar_stats["skewness"] > stats.skew(calibrated):
            # Predictions are too compressed, expand upper tail
            median_pred = np.median(calibrated)
            upper_mask = calibrated > median_pred
            expansion_factor = 1.05  # Mild expansion
            calibrated[upper_mask] = median_pred + (calibrated[upper_mask] - median_pred) * expansion_factor

        return np.clip(calibrated, 0.1, None)

    def transform_by_survey(
        self,
        predictions: np.ndarray,
        X_test: pd.DataFrame,
        weights: np.ndarray,
        survey_ids: np.ndarray
    ) -> np.ndarray:
        """Apply survey-specific calibration to multiple test surveys.

        Args:
            predictions: All predictions
            X_test: All test features
            weights: All sample weights
            survey_ids: Survey IDs for each prediction

        Returns:
            Calibrated predictions
        """
        calibrated = predictions.copy()

        for survey_id in np.unique(survey_ids):
            mask = survey_ids == survey_id
            survey_preds = predictions[mask]
            survey_X = X_test[mask]
            survey_weights = weights[mask]

            calibrated[mask] = self.transform(survey_preds, survey_X, survey_weights)

        return calibrated


class RateMatchingCalibrator:
    """Calibrate predictions to match expected poverty rate patterns.

    This calibrator directly optimizes predictions to match the distribution
    of poverty rates observed in training surveys.
    """

    def __init__(self, thresholds: list[float] | None = None):
        self.thresholds = thresholds or POVERTY_THRESHOLDS
        self.train_rate_patterns = {}

    def fit(
        self,
        rates_df: pd.DataFrame
    ) -> "RateMatchingCalibrator":
        """Fit calibrator on training poverty rates.

        Args:
            rates_df: Ground truth poverty rates DataFrame

        Returns:
            Self for chaining
        """
        for survey_id in TRAIN_SURVEYS:
            rates = get_true_rates_for_survey(rates_df, survey_id)
            self.train_rate_patterns[survey_id] = rates

        # Calculate average rate pattern
        avg_rates = {}
        for t in self.thresholds:
            avg_rates[t] = np.mean([
                self.train_rate_patterns[s][t] for s in TRAIN_SURVEYS
            ])
        self.train_rate_patterns["average"] = avg_rates

        return self

    def transform(
        self,
        predictions: np.ndarray,
        weights: np.ndarray,
        target_pattern: str | int = "average"
    ) -> np.ndarray:
        """Adjust predictions to match target rate pattern.

        Args:
            predictions: Model predictions
            weights: Sample weights
            target_pattern: 'average' or specific survey ID

        Returns:
            Calibrated predictions
        """
        target_rates = self.train_rate_patterns.get(target_pattern)
        if target_rates is None:
            target_rates = self.train_rate_patterns["average"]

        # Calculate current rates
        current_rates = calculate_poverty_rates(predictions, weights, self.thresholds)

        # Find key threshold (closest to 40%)
        key_threshold = min(self.thresholds, key=lambda t: abs(current_rates[t] - 0.4))
        rate_diff = target_rates[key_threshold] - current_rates[key_threshold]

        # Apply adjustment
        # If we're overpredicting poverty (rate too high), scale up predictions
        # If underpredicting poverty (rate too low), scale down
        if rate_diff > 0.05:  # Underpredicting poverty
            scale_factor = 0.95
        elif rate_diff < -0.05:  # Overpredicting poverty
            scale_factor = 1.05
        else:
            scale_factor = 1.0

        calibrated = predictions * scale_factor
        return np.clip(calibrated, 0.1, None)


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
