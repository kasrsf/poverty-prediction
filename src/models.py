"""Model definitions and wrappers."""

import numpy as np
import pandas as pd
from typing import Any
import pickle

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from config import (
    LIGHTGBM_PARAMS,
    XGBOOST_PARAMS,
    CATBOOST_PARAMS,
    MODELS_DIR,
    RANDOM_SEED,
)

# Quantiles for distribution modeling
QUANTILES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
             0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
             0.85, 0.90, 0.95]


class BaseModelWrapper:
    """Base wrapper for gradient boosting models."""

    def __init__(
        self,
        model_class: type,
        params: dict,
        use_log_target: bool = True,
        name: str = "model"
    ):
        self.model_class = model_class
        self.params = params.copy()
        self.use_log_target = use_log_target
        self.name = name
        self.model = None
        self.feature_cols = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: list[str] | None = None,
        **fit_params
    ) -> "BaseModelWrapper":
        """Fit the model.

        Args:
            X: Features DataFrame
            y: Target series
            feature_cols: List of feature columns to use
            **fit_params: Additional parameters for fit()
        """
        if feature_cols is None:
            feature_cols = X.columns.tolist()

        self.feature_cols = feature_cols
        X_fit = X[feature_cols]

        if self.use_log_target:
            y_fit = np.log1p(y)
        else:
            y_fit = y

        self.model = self.model_class(**self.params)
        self.model.fit(X_fit, y_fit, **fit_params)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.

        Args:
            X: Features DataFrame

        Returns:
            Array of predictions
        """
        X_pred = X[self.feature_cols]
        y_pred = self.model.predict(X_pred)

        if self.use_log_target:
            y_pred = np.expm1(y_pred)

        # Ensure positive predictions
        return np.clip(y_pred, 0.1, None)

    def get_feature_importance(self) -> pd.Series:
        """Get feature importances."""
        if self.model is None:
            raise ValueError("Model must be fitted first")

        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.feature_cols).sort_values(ascending=False)

    def save(self, path: str | None = None) -> None:
        """Save model to disk."""
        if path is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path = MODELS_DIR / f"{self.name}.pkl"

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BaseModelWrapper":
        """Load model from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)


class LightGBMModel(BaseModelWrapper):
    """LightGBM model wrapper."""

    def __init__(
        self,
        n_estimators: int = 1000,
        use_log_target: bool = True,
        params: dict | None = None,
        random_state: int | None = None
    ):
        base_params = LIGHTGBM_PARAMS.copy()
        if params:
            base_params.update(params)
        base_params["n_estimators"] = n_estimators
        if random_state is not None:
            base_params["random_state"] = random_state

        super().__init__(
            model_class=LGBMRegressor,
            params=base_params,
            use_log_target=use_log_target,
            name="lightgbm"
        )


class XGBoostModel(BaseModelWrapper):
    """XGBoost model wrapper."""

    def __init__(
        self,
        n_estimators: int = 1000,
        use_log_target: bool = True,
        params: dict | None = None,
        random_state: int | None = None
    ):
        base_params = XGBOOST_PARAMS.copy()
        if params:
            base_params.update(params)
        base_params["n_estimators"] = n_estimators
        if random_state is not None:
            base_params["random_state"] = random_state

        super().__init__(
            model_class=XGBRegressor,
            params=base_params,
            use_log_target=use_log_target,
            name="xgboost"
        )


class CatBoostModel(BaseModelWrapper):
    """CatBoost model wrapper."""

    def __init__(
        self,
        n_estimators: int = 1000,
        use_log_target: bool = True,
        params: dict | None = None,
        random_state: int | None = None
    ):
        base_params = CATBOOST_PARAMS.copy()
        if params:
            base_params.update(params)
        base_params["iterations"] = n_estimators
        if random_state is not None:
            base_params["random_seed"] = random_state

        super().__init__(
            model_class=CatBoostRegressor,
            params=base_params,
            use_log_target=use_log_target,
            name="catboost"
        )


def create_models(
    n_estimators: int = 1000,
    use_log_target: bool = True,
    seeds: list[int] | None = None
) -> list[BaseModelWrapper]:
    """Create a list of models for ensembling.

    Args:
        n_estimators: Number of boosting rounds
        use_log_target: Whether to use log-transformed target
        seeds: List of random seeds for multiple models

    Returns:
        List of model wrappers
    """
    if seeds is None:
        seeds = [RANDOM_SEED]

    models = []

    for seed in seeds:
        models.append(LightGBMModel(n_estimators, use_log_target, random_state=seed))
        models.append(XGBoostModel(n_estimators, use_log_target, random_state=seed))
        models.append(CatBoostModel(n_estimators, use_log_target, random_state=seed))

    # Update names to be unique
    for i, model in enumerate(models):
        model.name = f"{model.name}_{i}"

    return models


# =============================================================================
# Quantile Regression Models
# =============================================================================

class QuantileModelWrapper:
    """Base wrapper for quantile regression models.

    Trains multiple models, one for each quantile, to estimate the full
    conditional distribution of consumption.
    """

    def __init__(
        self,
        quantiles: list[float] | None = None,
        use_log_target: bool = True,
        name: str = "quantile_model"
    ):
        self.quantiles = quantiles if quantiles is not None else QUANTILES
        self.use_log_target = use_log_target
        self.name = name
        self.models = {}  # Dict mapping quantile -> model
        self.feature_cols = None

    def _create_single_model(self, quantile: float) -> Any:
        """Create a single quantile model. Override in subclasses."""
        raise NotImplementedError

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: list[str] | None = None,
        verbose: bool = True,
        **fit_params
    ) -> "QuantileModelWrapper":
        """Fit quantile models for all quantiles.

        Args:
            X: Features DataFrame
            y: Target series
            feature_cols: List of feature columns to use
            verbose: Whether to print progress
            **fit_params: Additional parameters for fit()
        """
        if feature_cols is None:
            feature_cols = X.columns.tolist()

        self.feature_cols = feature_cols
        X_fit = X[feature_cols]

        if self.use_log_target:
            y_fit = np.log1p(y)
        else:
            y_fit = y

        for i, q in enumerate(self.quantiles):
            if verbose:
                print(f"  Training quantile {q:.2f} ({i+1}/{len(self.quantiles)})...")

            model = self._create_single_model(q)
            model.fit(X_fit, y_fit, **fit_params)
            self.models[q] = model

        return self

    def predict(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        """Predict quantiles for all data points.

        Args:
            X: Features DataFrame

        Returns:
            Dict mapping quantile -> predictions array
        """
        X_pred = X[self.feature_cols]
        predictions = {}

        for q, model in self.models.items():
            y_pred = model.predict(X_pred)
            if self.use_log_target:
                y_pred = np.expm1(y_pred)
            predictions[q] = np.clip(y_pred, 0.1, None)

        return predictions

    def predict_mean(self, X: pd.DataFrame) -> np.ndarray:
        """Predict mean consumption (average of quantile predictions)."""
        quantile_preds = self.predict(X)
        return np.mean(list(quantile_preds.values()), axis=0)

    def predict_median(self, X: pd.DataFrame) -> np.ndarray:
        """Predict median consumption (0.5 quantile)."""
        X_pred = X[self.feature_cols]
        if 0.5 not in self.models:
            # Use closest quantile to 0.5
            closest_q = min(self.quantiles, key=lambda q: abs(q - 0.5))
            model = self.models[closest_q]
        else:
            model = self.models[0.5]

        y_pred = model.predict(X_pred)
        if self.use_log_target:
            y_pred = np.expm1(y_pred)
        return np.clip(y_pred, 0.1, None)

    def save(self, path: str | None = None) -> None:
        """Save all quantile models to disk."""
        if path is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path = MODELS_DIR / f"{self.name}.pkl"

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "QuantileModelWrapper":
        """Load quantile models from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)


class QuantileLightGBM(QuantileModelWrapper):
    """LightGBM quantile regression model."""

    def __init__(
        self,
        quantiles: list[float] | None = None,
        n_estimators: int = 1000,
        use_log_target: bool = True,
        params: dict | None = None,
        random_state: int | None = None
    ):
        super().__init__(quantiles, use_log_target, name="quantile_lightgbm")
        self.n_estimators = n_estimators
        self.base_params = LIGHTGBM_PARAMS.copy()
        if params:
            self.base_params.update(params)
        self.base_params["n_estimators"] = n_estimators
        if random_state is not None:
            self.base_params["random_state"] = random_state
        # Remove regression-specific params
        self.base_params.pop("objective", None)
        self.base_params.pop("metric", None)

    def _create_single_model(self, quantile: float) -> LGBMRegressor:
        """Create LightGBM model with quantile loss."""
        params = self.base_params.copy()
        params["objective"] = "quantile"
        params["alpha"] = quantile
        return LGBMRegressor(**params)


class QuantileXGBoost(QuantileModelWrapper):
    """XGBoost quantile regression model."""

    def __init__(
        self,
        quantiles: list[float] | None = None,
        n_estimators: int = 1000,
        use_log_target: bool = True,
        params: dict | None = None,
        random_state: int | None = None
    ):
        super().__init__(quantiles, use_log_target, name="quantile_xgboost")
        self.n_estimators = n_estimators
        self.base_params = XGBOOST_PARAMS.copy()
        if params:
            self.base_params.update(params)
        self.base_params["n_estimators"] = n_estimators
        if random_state is not None:
            self.base_params["random_state"] = random_state
        # Remove regression-specific params
        self.base_params.pop("objective", None)
        self.base_params.pop("eval_metric", None)

    def _create_single_model(self, quantile: float) -> XGBRegressor:
        """Create XGBoost model with quantile loss."""
        params = self.base_params.copy()
        params["objective"] = "reg:quantileerror"
        params["quantile_alpha"] = quantile
        return XGBRegressor(**params)


class QuantileCatBoost(QuantileModelWrapper):
    """CatBoost quantile regression model."""

    def __init__(
        self,
        quantiles: list[float] | None = None,
        n_estimators: int = 1000,
        use_log_target: bool = True,
        params: dict | None = None,
        random_state: int | None = None
    ):
        super().__init__(quantiles, use_log_target, name="quantile_catboost")
        self.n_estimators = n_estimators
        self.base_params = CATBOOST_PARAMS.copy()
        if params:
            self.base_params.update(params)
        self.base_params["iterations"] = n_estimators
        if random_state is not None:
            self.base_params["random_seed"] = random_state
        # Remove regression-specific params
        self.base_params.pop("loss_function", None)

    def _create_single_model(self, quantile: float) -> CatBoostRegressor:
        """Create CatBoost model with quantile loss."""
        params = self.base_params.copy()
        params["loss_function"] = f"Quantile:alpha={quantile}"
        return CatBoostRegressor(**params)


def create_quantile_models(
    n_estimators: int = 1000,
    use_log_target: bool = True,
    quantiles: list[float] | None = None,
    model_types: list[str] | None = None,
    random_state: int | None = None
) -> list[QuantileModelWrapper]:
    """Create quantile regression models for ensembling.

    Args:
        n_estimators: Number of boosting rounds
        use_log_target: Whether to use log-transformed target
        quantiles: List of quantiles to predict (default: 0.05 to 0.95)
        model_types: List of model types to create (default: all)
        random_state: Random seed

    Returns:
        List of quantile model wrappers
    """
    if model_types is None:
        model_types = ["lightgbm", "xgboost", "catboost"]

    if random_state is None:
        random_state = RANDOM_SEED

    models = []

    for model_type in model_types:
        if model_type == "lightgbm":
            models.append(QuantileLightGBM(
                quantiles=quantiles,
                n_estimators=n_estimators,
                use_log_target=use_log_target,
                random_state=random_state
            ))
        elif model_type == "xgboost":
            models.append(QuantileXGBoost(
                quantiles=quantiles,
                n_estimators=n_estimators,
                use_log_target=use_log_target,
                random_state=random_state
            ))
        elif model_type == "catboost":
            models.append(QuantileCatBoost(
                quantiles=quantiles,
                n_estimators=n_estimators,
                use_log_target=use_log_target,
                random_state=random_state
            ))

    return models


if __name__ == "__main__":
    from data_loader import load_all_data
    from preprocessing import preprocess_data
    from features import create_features, get_feature_columns

    print("Loading and preprocessing data...")
    data = load_all_data()

    train_processed, test_processed, _ = preprocess_data(
        data["train_features"], data["test_features"]
    )
    train_features = create_features(train_processed)
    feature_cols = get_feature_columns(train_features)

    print(f"Number of features: {len(feature_cols)}")

    # Test each model type
    print("\nTesting model types...")

    # Subset for quick testing
    sample_idx = np.random.choice(len(train_features), 5000, replace=False)
    X_sample = train_features.iloc[sample_idx]
    y_sample = data["train_target"].iloc[sample_idx]

    for model in [
        LightGBMModel(n_estimators=100),
        XGBoostModel(n_estimators=100),
        CatBoostModel(n_estimators=100),
    ]:
        print(f"\nTraining {model.name}...")
        model.fit(X_sample, y_sample, feature_cols)

        predictions = model.predict(X_sample)
        mape = np.mean(np.abs(predictions - y_sample) / y_sample) * 100
        print(f"  Training MAPE: {mape:.2f}%")

        # Top features
        importance = model.get_feature_importance()
        print(f"  Top 5 features: {importance.head().index.tolist()}")
