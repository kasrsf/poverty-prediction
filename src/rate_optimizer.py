"""Direct poverty rate optimization.

This module optimizes prediction transformations to minimize poverty rate MAPE
directly, rather than consumption prediction error.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from typing import Callable

from config import POVERTY_THRESHOLDS, TRAIN_SURVEYS
from metrics import (
    calculate_poverty_rates,
    weighted_poverty_rate_mape,
    competition_metric,
    get_true_rates_for_survey,
)


def multiplicative_transform(predictions: np.ndarray, alpha: float) -> np.ndarray:
    """Apply multiplicative scaling: pred * alpha."""
    return predictions * alpha


def power_transform(predictions: np.ndarray, beta: float) -> np.ndarray:
    """Apply power transformation: pred ** beta."""
    return np.power(predictions, beta)


def combined_transform(
    predictions: np.ndarray,
    alpha: float,
    beta: float
) -> np.ndarray:
    """Apply combined transformation: (pred ** beta) * alpha."""
    return np.power(predictions, beta) * alpha


def shift_transform(predictions: np.ndarray, shift: float) -> np.ndarray:
    """Apply additive shift: pred + shift."""
    return predictions + shift


class RateOptimizer:
    """Optimize predictions to minimize poverty rate MAPE."""

    def __init__(
        self,
        transform_type: str = "combined",
        thresholds: list[float] | None = None
    ):
        """Initialize rate optimizer.

        Args:
            transform_type: Type of transformation to apply
                - 'multiplicative': pred * alpha
                - 'power': pred ** beta
                - 'combined': (pred ** beta) * alpha
                - 'shift': pred + shift
            thresholds: Poverty thresholds to use
        """
        self.transform_type = transform_type
        self.thresholds = thresholds or POVERTY_THRESHOLDS
        self.optimal_params = None
        self.optimization_result = None

    def _apply_transform(
        self,
        predictions: np.ndarray,
        params: np.ndarray
    ) -> np.ndarray:
        """Apply transformation based on transform_type."""
        if self.transform_type == "multiplicative":
            return multiplicative_transform(predictions, params[0])
        elif self.transform_type == "power":
            return power_transform(predictions, params[0])
        elif self.transform_type == "combined":
            return combined_transform(predictions, params[0], params[1])
        elif self.transform_type == "shift":
            return shift_transform(predictions, params[0])
        else:
            raise ValueError(f"Unknown transform type: {self.transform_type}")

    def _get_bounds(self) -> list[tuple[float, float]]:
        """Get parameter bounds based on transform type."""
        if self.transform_type == "multiplicative":
            return [(0.5, 2.0)]  # alpha
        elif self.transform_type == "power":
            return [(0.5, 1.5)]  # beta
        elif self.transform_type == "combined":
            return [(0.5, 2.0), (0.5, 1.5)]  # alpha, beta
        elif self.transform_type == "shift":
            return [(-5.0, 5.0)]  # shift
        else:
            raise ValueError(f"Unknown transform type: {self.transform_type}")

    def _get_initial_params(self) -> np.ndarray:
        """Get initial parameter values (identity transform)."""
        if self.transform_type == "multiplicative":
            return np.array([1.0])
        elif self.transform_type == "power":
            return np.array([1.0])
        elif self.transform_type == "combined":
            return np.array([1.0, 1.0])
        elif self.transform_type == "shift":
            return np.array([0.0])
        else:
            raise ValueError(f"Unknown transform type: {self.transform_type}")

    def fit(
        self,
        predictions: np.ndarray,
        weights: np.ndarray,
        true_rates: dict[float, float],
        method: str = "SLSQP",
        use_global: bool = False
    ) -> "RateOptimizer":
        """Fit optimal transformation parameters.

        Args:
            predictions: Array of household consumption predictions
            weights: Array of household weights
            true_rates: Dictionary of true poverty rates
            method: Optimization method for scipy.optimize.minimize
            use_global: Use global optimization (differential_evolution)

        Returns:
            Self for chaining
        """
        def objective(params):
            transformed = self._apply_transform(predictions, params)
            transformed = np.clip(transformed, 0.1, None)  # Ensure positive
            pred_rates = calculate_poverty_rates(
                transformed, weights, self.thresholds
            )
            return weighted_poverty_rate_mape(true_rates, pred_rates, self.thresholds)

        bounds = self._get_bounds()
        x0 = self._get_initial_params()

        if use_global:
            result = differential_evolution(
                objective,
                bounds=bounds,
                seed=42,
                maxiter=100,
                tol=1e-6
            )
        else:
            result = minimize(
                objective,
                x0=x0,
                method=method,
                bounds=bounds,
                options={"maxiter": 1000}
            )

        self.optimal_params = result.x
        self.optimization_result = result

        return self

    def transform(self, predictions: np.ndarray) -> np.ndarray:
        """Apply optimal transformation to predictions.

        Args:
            predictions: Array of consumption predictions

        Returns:
            Transformed predictions
        """
        if self.optimal_params is None:
            raise ValueError("Must call fit() before transform()")

        transformed = self._apply_transform(predictions, self.optimal_params)
        return np.clip(transformed, 0.1, None)

    def fit_transform(
        self,
        predictions: np.ndarray,
        weights: np.ndarray,
        true_rates: dict[float, float],
        **fit_kwargs
    ) -> np.ndarray:
        """Fit and transform in one call."""
        self.fit(predictions, weights, true_rates, **fit_kwargs)
        return self.transform(predictions)


class ThresholdSpecificOptimizer:
    """Optimize predictions with threshold-specific adjustments.

    This optimizer learns a separate multiplicative adjustment for predictions
    near each poverty threshold, then interpolates between them.
    """

    def __init__(self, thresholds: list[float] | None = None):
        self.thresholds = thresholds or POVERTY_THRESHOLDS
        self.threshold_factors = None

    def fit(
        self,
        predictions: np.ndarray,
        weights: np.ndarray,
        true_rates: dict[float, float]
    ) -> "ThresholdSpecificOptimizer":
        """Fit threshold-specific adjustment factors.

        Args:
            predictions: Consumption predictions
            weights: Household weights
            true_rates: True poverty rates

        Returns:
            Self for chaining
        """
        n_thresholds = len(self.thresholds)

        def objective(factors):
            # Apply smooth interpolated adjustment
            adjusted = self._apply_threshold_adjustments(predictions, factors)
            adjusted = np.clip(adjusted, 0.1, None)
            pred_rates = calculate_poverty_rates(adjusted, weights, self.thresholds)
            return weighted_poverty_rate_mape(true_rates, pred_rates, self.thresholds)

        # Bounds: adjustment factors between 0.7 and 1.3
        bounds = [(0.7, 1.3) for _ in range(n_thresholds)]

        result = minimize(
            objective,
            x0=np.ones(n_thresholds),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 500}
        )

        self.threshold_factors = result.x
        return self

    def _apply_threshold_adjustments(
        self,
        predictions: np.ndarray,
        factors: np.ndarray
    ) -> np.ndarray:
        """Apply smooth threshold-specific adjustments."""
        adjusted = predictions.copy()

        # For each prediction, find the closest threshold and apply blended adjustment
        for i, pred in enumerate(predictions):
            # Find which thresholds this prediction is between
            if pred <= self.thresholds[0]:
                adjusted[i] = pred * factors[0]
            elif pred >= self.thresholds[-1]:
                adjusted[i] = pred * factors[-1]
            else:
                # Linear interpolation between adjacent thresholds
                for j in range(len(self.thresholds) - 1):
                    if self.thresholds[j] <= pred < self.thresholds[j + 1]:
                        t = (pred - self.thresholds[j]) / (
                            self.thresholds[j + 1] - self.thresholds[j]
                        )
                        factor = (1 - t) * factors[j] + t * factors[j + 1]
                        adjusted[i] = pred * factor
                        break

        return adjusted

    def transform(self, predictions: np.ndarray) -> np.ndarray:
        """Apply fitted threshold adjustments."""
        if self.threshold_factors is None:
            raise ValueError("Must call fit() before transform()")

        adjusted = self._apply_threshold_adjustments(predictions, self.threshold_factors)
        return np.clip(adjusted, 0.1, None)


class DistributionShapeOptimizer:
    """Optimize the shape of the prediction distribution.

    This optimizer adjusts predictions to better match the expected
    distribution shape (e.g., skewness, kurtosis) of consumption.
    """

    def __init__(self, target_skewness: float = 3.72):
        """Initialize with target distribution parameters.

        Args:
            target_skewness: Target skewness for predictions
        """
        self.target_skewness = target_skewness
        self.optimal_params = None

    def fit(
        self,
        predictions: np.ndarray,
        weights: np.ndarray,
        true_rates: dict[float, float],
        thresholds: list[float] | None = None
    ) -> "DistributionShapeOptimizer":
        """Fit optimal distribution adjustment parameters.

        Uses Box-Cox style power transform with shift to match target skewness
        while optimizing poverty rates.

        Args:
            predictions: Consumption predictions
            weights: Household weights
            true_rates: True poverty rates
            thresholds: Poverty thresholds

        Returns:
            Self for chaining
        """
        if thresholds is None:
            thresholds = POVERTY_THRESHOLDS

        def objective(params):
            # params: [power, scale, upper_tail_boost]
            power, scale, tail_boost = params

            # Apply power transform
            adjusted = np.power(predictions / scale, power) * scale

            # Boost upper tail to reduce compression
            median_pred = np.median(predictions)
            upper_mask = predictions > median_pred
            adjusted[upper_mask] *= (1 + tail_boost * (
                predictions[upper_mask] - median_pred
            ) / median_pred)

            adjusted = np.clip(adjusted, 0.1, None)

            # Calculate rate error
            pred_rates = calculate_poverty_rates(adjusted, weights, thresholds)
            rate_error = weighted_poverty_rate_mape(true_rates, pred_rates, thresholds)

            # Penalize skewness deviation
            from scipy.stats import skew
            current_skewness = skew(adjusted)
            skewness_penalty = abs(current_skewness - self.target_skewness) * 0.5

            return rate_error + skewness_penalty

        # Bounds: power, scale, tail_boost
        bounds = [(0.7, 1.3), (0.8, 1.2), (0.0, 0.3)]

        result = differential_evolution(
            objective,
            bounds=bounds,
            seed=42,
            maxiter=100
        )

        self.optimal_params = result.x
        return self

    def transform(self, predictions: np.ndarray) -> np.ndarray:
        """Apply fitted distribution adjustment."""
        if self.optimal_params is None:
            raise ValueError("Must call fit() before transform()")

        power, scale, tail_boost = self.optimal_params

        adjusted = np.power(predictions / scale, power) * scale

        median_pred = np.median(predictions)
        upper_mask = predictions > median_pred
        adjusted[upper_mask] *= (1 + tail_boost * (
            predictions[upper_mask] - median_pred
        ) / median_pred)

        return np.clip(adjusted, 0.1, None)


def optimize_for_survey(
    predictions: np.ndarray,
    weights: np.ndarray,
    true_rates: dict[float, float],
    optimizer_type: str = "combined"
) -> tuple[np.ndarray, dict]:
    """Optimize predictions for a single survey.

    Args:
        predictions: Consumption predictions for one survey
        weights: Household weights
        true_rates: True poverty rates for the survey
        optimizer_type: Type of optimizer to use

    Returns:
        Tuple of (optimized_predictions, optimization_info)
    """
    if optimizer_type == "combined":
        optimizer = RateOptimizer(transform_type="combined")
    elif optimizer_type == "threshold":
        optimizer = ThresholdSpecificOptimizer()
    elif optimizer_type == "distribution":
        optimizer = DistributionShapeOptimizer()
    else:
        optimizer = RateOptimizer(transform_type=optimizer_type)

    optimized = optimizer.fit_transform(predictions, weights, true_rates)

    # Calculate improvement
    original_rates = calculate_poverty_rates(predictions, weights)
    optimized_rates = calculate_poverty_rates(optimized, weights)

    original_mape = weighted_poverty_rate_mape(true_rates, original_rates)
    optimized_mape = weighted_poverty_rate_mape(true_rates, optimized_rates)

    info = {
        "original_mape": original_mape,
        "optimized_mape": optimized_mape,
        "improvement": original_mape - optimized_mape,
        "improvement_pct": (original_mape - optimized_mape) / original_mape * 100,
    }

    if hasattr(optimizer, "optimal_params"):
        info["optimal_params"] = optimizer.optimal_params

    return optimized, info


def cross_validate_optimizer(
    X: pd.DataFrame,
    y: pd.Series,
    predictions: np.ndarray,
    rates_df: pd.DataFrame,
    optimizer_type: str = "combined"
) -> dict:
    """Cross-validate rate optimizer using LOSO.

    Args:
        X: Features DataFrame (must have survey_id and weight)
        y: True consumption values
        predictions: Out-of-fold predictions from model
        rates_df: Ground truth poverty rates
        optimizer_type: Type of optimizer

    Returns:
        Dictionary with CV results
    """
    fold_results = []

    for test_survey in TRAIN_SURVEYS:
        test_mask = X["survey_id"] == test_survey

        test_preds = predictions[test_mask]
        test_weights = X.loc[test_mask, "weight"].values
        true_rates = get_true_rates_for_survey(rates_df, test_survey)

        optimized, info = optimize_for_survey(
            test_preds, test_weights, true_rates, optimizer_type
        )

        fold_results.append({
            "survey": test_survey,
            **info
        })

    # Aggregate results
    original_scores = [r["original_mape"] for r in fold_results]
    optimized_scores = [r["optimized_mape"] for r in fold_results]

    return {
        "fold_results": fold_results,
        "mean_original": np.mean(original_scores),
        "mean_optimized": np.mean(optimized_scores),
        "mean_improvement": np.mean([r["improvement"] for r in fold_results]),
        "mean_improvement_pct": np.mean([r["improvement_pct"] for r in fold_results]),
    }


if __name__ == "__main__":
    from data_loader import load_all_data
    from preprocessing import preprocess_data
    from features import create_features, get_feature_columns
    from models import CatBoostModel

    print("Loading data...")
    data = load_all_data()

    print("Preprocessing...")
    train_processed, _, _ = preprocess_data(
        data["train_features"], data["test_features"]
    )

    print("Creating features...")
    train_features = create_features(train_processed)
    feature_cols = get_feature_columns(train_features)

    # Train a simple model to get predictions
    print("\nTraining baseline model...")
    model = CatBoostModel(n_estimators=500)
    model.fit(train_features, data["train_target"], feature_cols)

    # Get OOF predictions (using training data for demo)
    predictions = model.predict(train_features)

    # Test rate optimization on survey 100000
    print("\nTesting rate optimization on survey 100000...")
    survey_mask = train_features["survey_id"] == 100000

    survey_preds = predictions[survey_mask]
    survey_weights = train_features.loc[survey_mask, "weight"].values
    true_rates = get_true_rates_for_survey(data["train_rates"], 100000)

    for opt_type in ["multiplicative", "power", "combined", "threshold"]:
        optimized, info = optimize_for_survey(
            survey_preds, survey_weights, true_rates, opt_type
        )
        print(f"\n{opt_type.upper()}:")
        print(f"  Original MAPE: {info['original_mape']:.2f}%")
        print(f"  Optimized MAPE: {info['optimized_mape']:.2f}%")
        print(f"  Improvement: {info['improvement_pct']:.1f}%")
