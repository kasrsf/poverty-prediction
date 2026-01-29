"""Model ensembling strategies."""

import numpy as np
import pandas as pd
from typing import Callable
from scipy.optimize import minimize
import pickle

from config import MODELS_DIR, RANDOM_SEED, POVERTY_THRESHOLDS
from models import BaseModelWrapper, QuantileModelWrapper, QUANTILES
from metrics import competition_metric, calculate_poverty_rates, get_true_rates_for_survey
from validation import LeaveOneSurveyOut


class Ensemble:
    """Weighted ensemble of models."""

    def __init__(self, models: list[BaseModelWrapper], weights: np.ndarray | None = None):
        """Initialize ensemble.

        Args:
            models: List of trained model wrappers
            weights: Optional weights for each model (uniform if None)
        """
        self.models = models
        self.n_models = len(models)

        if weights is None:
            self.weights = np.ones(self.n_models) / self.n_models
        else:
            self.weights = np.array(weights) / np.sum(weights)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions.

        Args:
            X: Features DataFrame

        Returns:
            Weighted average of model predictions
        """
        predictions = np.zeros((len(X), self.n_models))

        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)

        return np.average(predictions, axis=1, weights=self.weights)

    def set_weights(self, weights: np.ndarray) -> None:
        """Set ensemble weights.

        Args:
            weights: New weights for each model
        """
        self.weights = np.array(weights) / np.sum(weights)

    def save(self, path: str | None = None) -> None:
        """Save ensemble to disk."""
        if path is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path = MODELS_DIR / "ensemble.pkl"

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "Ensemble":
        """Load ensemble from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)


def optimize_ensemble_weights(
    models: list[BaseModelWrapper],
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    rates_df: pd.DataFrame,
    metric_fn: Callable | None = None,
    verbose: bool = True
) -> tuple[np.ndarray, float]:
    """Optimize ensemble weights using LOSO CV.

    Args:
        models: List of trained model wrappers
        X: Features DataFrame
        y: Target series
        feature_cols: List of feature column names
        rates_df: Ground truth poverty rates
        metric_fn: Custom metric function (lower is better)
        verbose: Whether to print progress

    Returns:
        Tuple of (optimal weights, best score)
    """
    n_models = len(models)

    # Get predictions from all models using LOSO
    all_predictions = []
    for model in models:
        # Get out-of-fold predictions
        oof_preds = np.zeros(len(y))
        loso = LeaveOneSurveyOut()

        for train_idx, test_idx in loso.split(X):
            X_train = X.iloc[train_idx][feature_cols]
            X_test = X.iloc[test_idx][feature_cols]
            y_train = y.iloc[train_idx]

            # Clone and train
            model_clone = model.model_class(**model.params)
            y_train_fit = np.log1p(y_train) if model.use_log_target else y_train
            model_clone.fit(X_train, y_train_fit)

            # Predict
            preds = model_clone.predict(X_test)
            if model.use_log_target:
                preds = np.expm1(preds)
            preds = np.clip(preds, 0.1, None)

            oof_preds[test_idx] = preds

        all_predictions.append(oof_preds)

    predictions = np.column_stack(all_predictions)

    def objective(weights):
        """Objective function: weighted competition metric."""
        weights = np.abs(weights)
        weights = weights / weights.sum()

        ensemble_preds = np.average(predictions, axis=1, weights=weights)

        # Calculate metric per survey
        total_score = 0
        total_samples = 0

        for survey_id in X["survey_id"].unique():
            mask = X["survey_id"] == survey_id
            survey_preds = ensemble_preds[mask]
            survey_y = y[mask].values
            survey_weights = X.loc[mask, "weight"].values

            # Get true rates
            true_rates = get_true_rates_for_survey(rates_df, survey_id)

            metric = competition_metric(survey_y, survey_preds, survey_weights, true_rates)
            total_score += metric["total"] * len(survey_y)
            total_samples += len(survey_y)

        return total_score / total_samples

    # Initial weights (uniform)
    x0 = np.ones(n_models) / n_models

    # Constraints: weights sum to 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    # Bounds: weights between 0 and 1
    bounds = [(0, 1) for _ in range(n_models)]

    if verbose:
        print("Optimizing ensemble weights...")
        print(f"Initial score: {objective(x0):.4f}")

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200}
    )

    optimal_weights = np.abs(result.x)
    optimal_weights = optimal_weights / optimal_weights.sum()
    best_score = result.fun

    if verbose:
        print(f"Optimized score: {best_score:.4f}")
        print("\nOptimal weights:")
        for model, weight in zip(models, optimal_weights):
            print(f"  {model.name}: {weight:.4f}")

    return optimal_weights, best_score


def create_stacking_ensemble(
    base_models: list[BaseModelWrapper],
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    verbose: bool = True
) -> tuple[Ensemble, BaseModelWrapper]:
    """Create a stacking ensemble with a linear meta-learner.

    Args:
        base_models: List of trained base models
        X: Features DataFrame
        y: Target series
        feature_cols: List of feature column names
        verbose: Whether to print progress

    Returns:
        Tuple of (base ensemble, meta model)
    """
    from sklearn.linear_model import Ridge

    # Generate out-of-fold predictions for stacking
    loso = LeaveOneSurveyOut()
    n_models = len(base_models)
    oof_predictions = np.zeros((len(y), n_models))

    for i, model in enumerate(base_models):
        if verbose:
            print(f"Generating OOF predictions for {model.name}...")

        for train_idx, test_idx in loso.split(X):
            X_train = X.iloc[train_idx][feature_cols]
            X_test = X.iloc[test_idx][feature_cols]
            y_train = y.iloc[train_idx]

            model_clone = model.model_class(**model.params)
            y_train_fit = np.log1p(y_train) if model.use_log_target else y_train
            model_clone.fit(X_train, y_train_fit)

            preds = model_clone.predict(X_test)
            if model.use_log_target:
                preds = np.expm1(preds)

            oof_predictions[test_idx, i] = np.clip(preds, 0.1, None)

    # Train meta-learner on OOF predictions
    if verbose:
        print("\nTraining meta-learner...")

    meta_model = Ridge(alpha=1.0)
    meta_model.fit(np.log1p(oof_predictions), np.log1p(y))

    # Get meta-model weights
    meta_weights = np.abs(meta_model.coef_)
    meta_weights = meta_weights / meta_weights.sum()

    if verbose:
        print("Meta-learner weights:")
        for model, weight in zip(base_models, meta_weights):
            print(f"  {model.name}: {weight:.4f}")

    ensemble = Ensemble(base_models, meta_weights)

    return ensemble, meta_model


def load_trained_models() -> list[BaseModelWrapper]:
    """Load all trained models from disk.

    Returns:
        List of loaded model wrappers
    """
    import glob

    model_files = glob.glob(str(MODELS_DIR / "*.pkl"))
    model_files = [f for f in model_files if "preprocessor" not in f and "ensemble" not in f]

    models = []
    for path in model_files:
        try:
            model = BaseModelWrapper.load(path)
            models.append(model)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")

    return models


# =============================================================================
# Quantile Ensemble Methods
# =============================================================================

class QuantileEnsemble:
    """Ensemble of quantile regression models.

    Combines predictions from multiple quantile models to estimate
    the full conditional distribution of consumption.
    """

    def __init__(
        self,
        models: list[QuantileModelWrapper],
        weights: np.ndarray | None = None
    ):
        """Initialize quantile ensemble.

        Args:
            models: List of trained quantile model wrappers
            weights: Optional weights for each model (uniform if None)
        """
        self.models = models
        self.n_models = len(models)

        if weights is None:
            self.weights = np.ones(self.n_models) / self.n_models
        else:
            self.weights = np.array(weights) / np.sum(weights)

        # Get quantiles from first model
        self.quantiles = models[0].quantiles if models else QUANTILES

    def predict_quantiles(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        """Predict all quantiles using weighted ensemble.

        Args:
            X: Features DataFrame

        Returns:
            Dict mapping quantile -> predictions array
        """
        # Collect predictions from all models
        all_predictions = {q: [] for q in self.quantiles}

        for model in self.models:
            model_preds = model.predict(X)
            for q in self.quantiles:
                if q in model_preds:
                    all_predictions[q].append(model_preds[q])

        # Weighted average across models
        ensemble_preds = {}
        for q in self.quantiles:
            if all_predictions[q]:
                stacked = np.column_stack(all_predictions[q])
                ensemble_preds[q] = np.average(stacked, axis=1, weights=self.weights[:len(all_predictions[q])])

        return ensemble_preds

    def predict_mean(self, X: pd.DataFrame) -> np.ndarray:
        """Predict mean consumption (average of all quantile predictions)."""
        quantile_preds = self.predict_quantiles(X)
        return np.mean(list(quantile_preds.values()), axis=0)

    def predict_median(self, X: pd.DataFrame) -> np.ndarray:
        """Predict median consumption (0.5 quantile or nearest)."""
        quantile_preds = self.predict_quantiles(X)
        if 0.5 in quantile_preds:
            return quantile_preds[0.5]
        else:
            # Find closest quantile to 0.5
            closest_q = min(self.quantiles, key=lambda q: abs(q - 0.5))
            return quantile_preds[closest_q]

    def predict_poverty_rates(
        self,
        X: pd.DataFrame,
        weights: np.ndarray,
        thresholds: list[float] | None = None
    ) -> dict[float, float]:
        """Estimate poverty rates from quantile predictions.

        For each threshold, estimate the proportion below by finding
        which quantiles the threshold falls between.

        Args:
            X: Features DataFrame
            weights: Household weights
            thresholds: Poverty thresholds

        Returns:
            Dict mapping threshold -> estimated poverty rate
        """
        if thresholds is None:
            thresholds = POVERTY_THRESHOLDS

        quantile_preds = self.predict_quantiles(X)
        sorted_quantiles = sorted(quantile_preds.keys())

        rates = {}
        for threshold in thresholds:
            # For each household, estimate P(consumption < threshold)
            household_probs = np.zeros(len(X))

            for i in range(len(X)):
                # Get predicted quantile values for this household
                pred_values = [quantile_preds[q][i] for q in sorted_quantiles]

                # Find where threshold falls in the predicted distribution
                if threshold <= pred_values[0]:
                    # Below all quantiles
                    prob = sorted_quantiles[0] / 2
                elif threshold >= pred_values[-1]:
                    # Above all quantiles
                    prob = (1 + sorted_quantiles[-1]) / 2
                else:
                    # Interpolate between quantiles
                    for j in range(len(pred_values) - 1):
                        if pred_values[j] <= threshold < pred_values[j + 1]:
                            # Linear interpolation
                            t = (threshold - pred_values[j]) / (pred_values[j + 1] - pred_values[j])
                            prob = sorted_quantiles[j] + t * (sorted_quantiles[j + 1] - sorted_quantiles[j])
                            break

                household_probs[i] = prob

            # Weighted average probability = poverty rate
            rates[threshold] = np.average(household_probs, weights=weights)

        return rates

    def set_weights(self, weights: np.ndarray) -> None:
        """Set ensemble weights."""
        self.weights = np.array(weights) / np.sum(weights)

    def save(self, path: str | None = None) -> None:
        """Save quantile ensemble to disk."""
        if path is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path = MODELS_DIR / "quantile_ensemble.pkl"

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "QuantileEnsemble":
        """Load quantile ensemble from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)


class MedianBlendingEnsemble:
    """Ensemble that uses median blending instead of mean.

    More robust to outliers in model predictions.
    """

    def __init__(self, models: list[BaseModelWrapper]):
        """Initialize median blending ensemble.

        Args:
            models: List of trained model wrappers
        """
        self.models = models
        self.n_models = len(models)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using median of all model outputs.

        Args:
            X: Features DataFrame

        Returns:
            Median of model predictions
        """
        predictions = np.zeros((len(X), self.n_models))

        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)

        return np.median(predictions, axis=1)

    def predict_with_uncertainty(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty estimate.

        Args:
            X: Features DataFrame

        Returns:
            Tuple of (median predictions, IQR uncertainty)
        """
        predictions = np.zeros((len(X), self.n_models))

        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)

        median_pred = np.median(predictions, axis=1)
        p25 = np.percentile(predictions, 25, axis=1)
        p75 = np.percentile(predictions, 75, axis=1)
        iqr = p75 - p25

        return median_pred, iqr


class HybridEnsemble:
    """Hybrid ensemble combining point prediction and quantile models.

    Uses point predictions for mean estimation and quantile predictions
    for distribution estimation, optimized for poverty rate accuracy.
    """

    def __init__(
        self,
        point_models: list[BaseModelWrapper],
        quantile_models: list[QuantileModelWrapper] | None = None,
        point_weight: float = 0.5
    ):
        """Initialize hybrid ensemble.

        Args:
            point_models: List of point prediction models
            quantile_models: List of quantile regression models
            point_weight: Weight for point predictions vs quantile (0-1)
        """
        self.point_ensemble = Ensemble(point_models)
        self.quantile_ensemble = QuantileEnsemble(quantile_models) if quantile_models else None
        self.point_weight = point_weight

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make hybrid predictions.

        Args:
            X: Features DataFrame

        Returns:
            Blended predictions
        """
        point_preds = self.point_ensemble.predict(X)

        if self.quantile_ensemble is not None:
            quantile_preds = self.quantile_ensemble.predict_mean(X)
            return self.point_weight * point_preds + (1 - self.point_weight) * quantile_preds
        else:
            return point_preds

    def predict_poverty_rates(
        self,
        X: pd.DataFrame,
        weights: np.ndarray,
        thresholds: list[float] | None = None
    ) -> dict[float, float]:
        """Predict poverty rates using best available method.

        Uses quantile-based estimation if available, otherwise
        falls back to standard rate calculation.
        """
        if self.quantile_ensemble is not None:
            return self.quantile_ensemble.predict_poverty_rates(X, weights, thresholds)
        else:
            predictions = self.point_ensemble.predict(X)
            return calculate_poverty_rates(predictions, weights, thresholds)


def optimize_quantile_ensemble_weights(
    quantile_models: list[QuantileModelWrapper],
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    rates_df: pd.DataFrame,
    verbose: bool = True
) -> tuple[np.ndarray, float]:
    """Optimize quantile ensemble weights for poverty rate accuracy.

    Args:
        quantile_models: List of trained quantile models
        X: Features DataFrame
        y: Target series
        feature_cols: Feature columns
        rates_df: Ground truth rates
        verbose: Print progress

    Returns:
        Tuple of (optimal weights, best score)
    """
    n_models = len(quantile_models)
    n_samples = len(X)
    surveys = X["survey_id"].unique()

    # Collect all quantile predictions
    all_model_preds = []
    for model in quantile_models:
        model_preds = model.predict(X)
        # Use median prediction for optimization
        if 0.5 in model_preds:
            all_model_preds.append(model_preds[0.5])
        else:
            # Average of all quantiles
            all_model_preds.append(np.mean(list(model_preds.values()), axis=0))

    predictions = np.column_stack(all_model_preds)

    def objective(weights):
        weights = np.abs(weights)
        weights = weights / weights.sum()

        ensemble_preds = np.average(predictions, axis=1, weights=weights)

        total_score = 0
        total_samples = 0

        for survey_id in surveys:
            mask = X["survey_id"] == survey_id
            survey_preds = ensemble_preds[mask]
            survey_y = y[mask].values
            survey_weights = X.loc[mask, "weight"].values

            true_rates = get_true_rates_for_survey(rates_df, survey_id)
            metric = competition_metric(survey_y, survey_preds, survey_weights, true_rates)
            total_score += metric["total"] * len(survey_y)
            total_samples += len(survey_y)

        return total_score / total_samples

    x0 = np.ones(n_models) / n_models
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n_models)]

    if verbose:
        print("Optimizing quantile ensemble weights...")
        print(f"Initial score: {objective(x0):.4f}")

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = np.abs(result.x)
    optimal_weights = optimal_weights / optimal_weights.sum()

    if verbose:
        print(f"Optimized score: {result.fun:.4f}")
        print("Optimal weights:")
        for model, weight in zip(quantile_models, optimal_weights):
            print(f"  {model.name}: {weight:.4f}")

    return optimal_weights, result.fun


if __name__ == "__main__":
    from data_loader import load_all_data
    from preprocessing import preprocess_data
    from features import create_features, get_feature_columns
    from train import train_all_models

    print("Loading and preprocessing data...")
    data = load_all_data()

    train_processed, test_processed, preprocessor = preprocess_data(
        data["train_features"], data["test_features"]
    )
    train_features = create_features(train_processed)
    test_features = create_features(test_processed)
    feature_cols = get_feature_columns(train_features)

    # Train models (or load if already trained)
    print("\nTraining models...")
    results = train_all_models(
        train_features,
        data["train_target"],
        feature_cols,
        data["train_rates"],
        n_estimators=500,
        seeds=[42],
        run_validation=False,
        verbose=True
    )

    models = [r["model"] for r in results]

    # Optimize ensemble weights
    print("\n" + "=" * 60)
    print("ENSEMBLE OPTIMIZATION")
    print("=" * 60)

    optimal_weights, best_score = optimize_ensemble_weights(
        models,
        train_features,
        data["train_target"],
        feature_cols,
        data["train_rates"],
        verbose=True
    )

    # Create ensemble
    ensemble = Ensemble(models, optimal_weights)

    # Test ensemble predictions
    print("\nTesting ensemble predictions...")
    train_preds = ensemble.predict(train_features)
    test_preds = ensemble.predict(test_features)

    print(f"Train prediction range: {train_preds.min():.2f} - {train_preds.max():.2f}")
    print(f"Test prediction range: {test_preds.min():.2f} - {test_preds.max():.2f}")

    # Save ensemble
    ensemble.save()
    print("\nEnsemble saved.")
