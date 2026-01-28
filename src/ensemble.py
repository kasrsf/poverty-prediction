"""Model ensembling strategies."""

import numpy as np
import pandas as pd
from typing import Callable
from scipy.optimize import minimize
import pickle

from config import MODELS_DIR, RANDOM_SEED
from models import BaseModelWrapper
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
