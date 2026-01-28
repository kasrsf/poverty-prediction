"""Competition metrics implementation."""

import numpy as np
import pandas as pd

from config import POVERTY_THRESHOLDS, POVERTY_RATE_COLS, THRESHOLD_TO_STR


def calculate_poverty_rates(
    predictions: np.ndarray,
    weights: np.ndarray,
    thresholds: list[float] | None = None
) -> dict[float, float]:
    """Calculate weighted poverty rates for each threshold.

    Args:
        predictions: Array of household consumption predictions
        weights: Array of household weights
        thresholds: List of poverty thresholds (defaults to POVERTY_THRESHOLDS)

    Returns:
        Dictionary mapping threshold to poverty rate
    """
    if thresholds is None:
        thresholds = POVERTY_THRESHOLDS

    total_weight = weights.sum()
    rates = {}

    for threshold in thresholds:
        below_threshold = predictions < threshold
        weight_below = weights[below_threshold].sum()
        rates[threshold] = weight_below / total_weight

    return rates


def poverty_rate_mape(
    true_rates: dict[float, float],
    pred_rates: dict[float, float],
    thresholds: list[float] | None = None
) -> float:
    """Calculate MAPE between true and predicted poverty rates.

    Args:
        true_rates: Dictionary of true poverty rates
        pred_rates: Dictionary of predicted poverty rates
        thresholds: List of thresholds to evaluate

    Returns:
        Mean absolute percentage error
    """
    if thresholds is None:
        thresholds = POVERTY_THRESHOLDS

    errors = []
    for t in thresholds:
        if true_rates[t] > 0:
            error = abs(pred_rates[t] - true_rates[t]) / true_rates[t]
        else:
            error = abs(pred_rates[t])  # If true is 0, error is just the prediction
        errors.append(error)

    return np.mean(errors) * 100  # As percentage


def weighted_poverty_rate_mape(
    true_rates: dict[float, float],
    pred_rates: dict[float, float],
    thresholds: list[float] | None = None
) -> float:
    """Calculate weighted MAPE giving more weight to rates near 40%.

    Weights: w_t = 1 - |p_t - 0.4|

    Args:
        true_rates: Dictionary of true poverty rates
        pred_rates: Dictionary of predicted poverty rates
        thresholds: List of thresholds to evaluate

    Returns:
        Weighted mean absolute percentage error
    """
    if thresholds is None:
        thresholds = POVERTY_THRESHOLDS

    errors = []
    threshold_weights = []

    for t in thresholds:
        # Weight based on proximity to 40%
        w = 1 - abs(true_rates[t] - 0.4)
        threshold_weights.append(w)

        if true_rates[t] > 0:
            error = abs(pred_rates[t] - true_rates[t]) / true_rates[t]
        else:
            error = abs(pred_rates[t])
        errors.append(error)

    total_weight = sum(threshold_weights)
    weighted_error = sum(e * w for e, w in zip(errors, threshold_weights)) / total_weight

    return weighted_error * 100  # As percentage


def consumption_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MAPE for household consumption.

    Args:
        y_true: True consumption values
        y_pred: Predicted consumption values

    Returns:
        Mean absolute percentage error
    """
    # Avoid division by zero
    mask = y_true > 0
    return np.mean(np.abs(y_pred[mask] - y_true[mask]) / y_true[mask]) * 100


def competition_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    true_rates: dict[float, float] | None = None,
    thresholds: list[float] | None = None
) -> dict[str, float]:
    """Calculate the full competition metric.

    Competition metric = 0.9 * weighted_poverty_rate_mape + 0.1 * consumption_mape

    Args:
        y_true: True consumption values
        y_pred: Predicted consumption values
        weights: Household weights
        true_rates: Pre-computed true poverty rates (optional)
        thresholds: List of poverty thresholds

    Returns:
        Dictionary with component scores and total
    """
    if thresholds is None:
        thresholds = POVERTY_THRESHOLDS

    # Calculate true rates if not provided
    if true_rates is None:
        true_rates = calculate_poverty_rates(y_true, weights, thresholds)

    # Calculate predicted rates
    pred_rates = calculate_poverty_rates(y_pred, weights, thresholds)

    # Component metrics
    rate_error = weighted_poverty_rate_mape(true_rates, pred_rates, thresholds)
    cons_error = consumption_mape(y_true, y_pred)

    # Combined metric
    total = 0.9 * rate_error + 0.1 * cons_error

    return {
        "poverty_rate_mape": rate_error,
        "consumption_mape": cons_error,
        "total": total,
        "pred_rates": pred_rates,
        "true_rates": true_rates,
    }


def validate_rates(rates: dict[float, float]) -> bool:
    """Validate that poverty rates are monotonically increasing.

    Args:
        rates: Dictionary of poverty rates by threshold

    Returns:
        True if valid (monotonically increasing)
    """
    sorted_thresholds = sorted(rates.keys())
    prev_rate = 0

    for t in sorted_thresholds:
        if rates[t] < prev_rate:
            return False
        prev_rate = rates[t]

    return True


def get_true_rates_for_survey(rates_df: pd.DataFrame, survey_id: int) -> dict[float, float]:
    """Extract true poverty rates for a specific survey from the rates dataframe.

    Args:
        rates_df: DataFrame with ground truth rates
        survey_id: Survey ID to look up

    Returns:
        Dictionary mapping threshold (float) to rate (float)
    """
    row = rates_df[rates_df["survey_id"] == survey_id].iloc[0]
    return {
        float(col.replace("pct_hh_below_", "")): row[col]
        for col in POVERTY_RATE_COLS
    }


if __name__ == "__main__":
    from data_loader import load_train_data

    print("Loading training data...")
    features, target, rates_df = load_train_data()

    # Test with training data (perfect predictions)
    print("\nTesting metrics with perfect predictions on survey 100000...")

    survey_mask = features["survey_id"] == 100000
    survey_target = target[survey_mask].values
    survey_weights = features.loc[survey_mask, "weight"].values

    # Calculate rates from data
    calculated_rates = calculate_poverty_rates(survey_target, survey_weights)

    # Get true rates from file using helper
    true_rates = get_true_rates_for_survey(rates_df, 100000)

    print("\nComparing calculated vs. ground truth rates:")
    for t in POVERTY_THRESHOLDS[:5]:
        print(f"  Threshold {t}: Calculated={calculated_rates[t]:.4f}, GT={true_rates[t]:.4f}")

    # Calculate metric with perfect predictions
    metric = competition_metric(survey_target, survey_target, survey_weights, true_rates)
    print(f"\nPerfect prediction metric (should be ~0): {metric['total']:.4f}")

    # Test with noisy predictions
    print("\nTesting with noisy predictions...")
    noisy_pred = survey_target * (1 + np.random.normal(0, 0.1, len(survey_target)))
    noisy_pred = np.clip(noisy_pred, 0.1, None)  # Ensure positive

    metric = competition_metric(survey_target, noisy_pred, survey_weights, true_rates)
    print(f"Noisy prediction metric: {metric['total']:.2f}")
    print(f"  - Poverty rate MAPE: {metric['poverty_rate_mape']:.2f}")
    print(f"  - Consumption MAPE: {metric['consumption_mape']:.2f}")

    # Validate monotonicity
    print(f"\nRates monotonically increasing: {validate_rates(metric['pred_rates'])}")
