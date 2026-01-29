"""Optuna hyperparameter tuning for poverty prediction models."""

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
import pickle
from pathlib import Path
from typing import Callable
import argparse

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from config import (
    RANDOM_SEED,
    MODELS_DIR,
    TRAIN_SURVEYS,
    POVERTY_THRESHOLDS,
)
from data_loader import load_all_data
from preprocessing import preprocess_data
from features import create_features, get_feature_columns
from metrics import competition_metric, get_true_rates_for_survey


def create_lgbm_objective(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    rates_df: pd.DataFrame,
    use_quantile: bool = False,
    quantile: float = 0.5
) -> Callable:
    """Create Optuna objective function for LightGBM.

    Args:
        X: Features DataFrame
        y: Target series
        feature_cols: Feature columns to use
        rates_df: Ground truth poverty rates
        use_quantile: Whether to use quantile regression
        quantile: Quantile to optimize (if use_quantile=True)

    Returns:
        Objective function for Optuna
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "n_estimators": 500,  # Fixed for faster tuning
            "random_state": RANDOM_SEED,
            "verbose": -1,
            "n_jobs": -1,
        }

        if use_quantile:
            params["objective"] = "quantile"
            params["alpha"] = quantile
        else:
            params["objective"] = "regression"
            params["metric"] = "mape"

        # LOSO cross-validation
        scores = []
        for test_survey in TRAIN_SURVEYS:
            train_mask = X["survey_id"] != test_survey
            test_mask = X["survey_id"] == test_survey

            X_train = X.loc[train_mask, feature_cols]
            X_test = X.loc[test_mask, feature_cols]
            y_train = y[train_mask]
            y_test = y[test_mask]

            # Log transform target
            y_train_log = np.log1p(y_train)

            model = LGBMRegressor(**params)
            model.fit(X_train, y_train_log)

            # Predict and transform back
            y_pred = np.expm1(model.predict(X_test))
            y_pred = np.clip(y_pred, 0.1, None)

            # Calculate competition metric
            weights = X.loc[test_mask, "weight"].values
            true_rates = get_true_rates_for_survey(rates_df, test_survey)

            metric = competition_metric(y_test.values, y_pred, weights, true_rates)
            scores.append(metric["total"])

        return np.mean(scores)

    return objective


def create_xgb_objective(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    rates_df: pd.DataFrame,
    use_quantile: bool = False,
    quantile: float = 0.5
) -> Callable:
    """Create Optuna objective function for XGBoost."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "n_estimators": 500,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
        }

        if use_quantile:
            params["objective"] = "reg:quantileerror"
            params["quantile_alpha"] = quantile
        else:
            params["objective"] = "reg:squarederror"

        # LOSO cross-validation
        scores = []
        for test_survey in TRAIN_SURVEYS:
            train_mask = X["survey_id"] != test_survey
            test_mask = X["survey_id"] == test_survey

            X_train = X.loc[train_mask, feature_cols]
            X_test = X.loc[test_mask, feature_cols]
            y_train = y[train_mask]
            y_test = y[test_mask]

            y_train_log = np.log1p(y_train)

            model = XGBRegressor(**params)
            model.fit(X_train, y_train_log)

            y_pred = np.expm1(model.predict(X_test))
            y_pred = np.clip(y_pred, 0.1, None)

            weights = X.loc[test_mask, "weight"].values
            true_rates = get_true_rates_for_survey(rates_df, test_survey)

            metric = competition_metric(y_test.values, y_pred, weights, true_rates)
            scores.append(metric["total"])

        return np.mean(scores)

    return objective


def create_catboost_objective(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    rates_df: pd.DataFrame,
    use_quantile: bool = False,
    quantile: float = 0.5
) -> Callable:
    """Create Optuna objective function for CatBoost."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "iterations": 500,
            "random_seed": RANDOM_SEED,
            "verbose": False,
        }

        if use_quantile:
            params["loss_function"] = f"Quantile:alpha={quantile}"
        else:
            params["loss_function"] = "MAPE"

        # LOSO cross-validation
        scores = []
        for test_survey in TRAIN_SURVEYS:
            train_mask = X["survey_id"] != test_survey
            test_mask = X["survey_id"] == test_survey

            X_train = X.loc[train_mask, feature_cols]
            X_test = X.loc[test_mask, feature_cols]
            y_train = y[train_mask]
            y_test = y[test_mask]

            y_train_log = np.log1p(y_train)

            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train_log)

            y_pred = np.expm1(model.predict(X_test))
            y_pred = np.clip(y_pred, 0.1, None)

            weights = X.loc[test_mask, "weight"].values
            true_rates = get_true_rates_for_survey(rates_df, test_survey)

            metric = competition_metric(y_test.values, y_pred, weights, true_rates)
            scores.append(metric["total"])

        return np.mean(scores)

    return objective


def run_optuna_study(
    model_type: str,
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    rates_df: pd.DataFrame,
    n_trials: int = 100,
    use_quantile: bool = False,
    quantile: float = 0.5,
    study_name: str | None = None,
    storage: str | None = None
) -> optuna.Study:
    """Run Optuna hyperparameter optimization study.

    Args:
        model_type: One of 'lightgbm', 'xgboost', 'catboost'
        X: Features DataFrame
        y: Target series
        feature_cols: Feature columns
        rates_df: Ground truth rates
        n_trials: Number of optimization trials
        use_quantile: Whether to use quantile regression
        quantile: Quantile value if using quantile regression
        study_name: Name for the study
        storage: Optional database URL for persistence

    Returns:
        Completed Optuna study
    """
    # Create objective function
    if model_type == "lightgbm":
        objective = create_lgbm_objective(
            X, y, feature_cols, rates_df, use_quantile, quantile
        )
    elif model_type == "xgboost":
        objective = create_xgb_objective(
            X, y, feature_cols, rates_df, use_quantile, quantile
        )
    elif model_type == "catboost":
        objective = create_catboost_objective(
            X, y, feature_cols, rates_df, use_quantile, quantile
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create or load study
    if study_name is None:
        quantile_str = f"_q{int(quantile*100)}" if use_quantile else ""
        study_name = f"{model_type}{quantile_str}_tuning"

    sampler = TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study


def save_best_params(studies: dict[str, optuna.Study], output_path: Path | None = None):
    """Save best parameters from all studies.

    Args:
        studies: Dict mapping model type to Optuna study
        output_path: Path to save parameters (defaults to MODELS_DIR/best_params.pkl)
    """
    if output_path is None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = MODELS_DIR / "best_params.pkl"

    best_params = {}
    for model_type, study in studies.items():
        best_params[model_type] = {
            "params": study.best_params,
            "score": study.best_value,
            "n_trials": len(study.trials)
        }

    with open(output_path, "wb") as f:
        pickle.dump(best_params, f)

    print(f"\nBest parameters saved to {output_path}")
    return best_params


def load_best_params(path: Path | None = None) -> dict:
    """Load previously saved best parameters.

    Args:
        path: Path to saved parameters

    Returns:
        Dictionary of best parameters per model type
    """
    if path is None:
        path = MODELS_DIR / "best_params.pkl"

    with open(path, "rb") as f:
        return pickle.load(f)


def main(
    n_trials: int = 50,
    model_types: list[str] | None = None,
    use_quantile: bool = False,
    quantiles: list[float] | None = None
):
    """Run full hyperparameter optimization.

    Args:
        n_trials: Number of trials per model type
        model_types: List of model types to tune
        use_quantile: Whether to tune quantile models
        quantiles: List of quantiles to tune (if use_quantile=True)
    """
    if model_types is None:
        model_types = ["lightgbm", "xgboost", "catboost"]

    if quantiles is None:
        quantiles = [0.5]  # Default: median only for faster tuning

    # Load and preprocess data
    print("Loading data...")
    data = load_all_data()

    print("Preprocessing...")
    train_processed, _, _ = preprocess_data(
        data["train_features"], data["test_features"]
    )

    print("Creating features...")
    train_features = create_features(train_processed)
    feature_cols = get_feature_columns(train_features)

    print(f"Number of features: {len(feature_cols)}")
    print(f"Training samples: {len(train_features)}")

    # Run optimization for each model type
    studies = {}

    for model_type in model_types:
        if use_quantile:
            for q in quantiles:
                print(f"\n{'='*60}")
                print(f"Tuning {model_type} (quantile={q:.2f})...")
                print(f"{'='*60}")

                study = run_optuna_study(
                    model_type=model_type,
                    X=train_features,
                    y=data["train_target"],
                    feature_cols=feature_cols,
                    rates_df=data["train_rates"],
                    n_trials=n_trials,
                    use_quantile=True,
                    quantile=q
                )

                key = f"{model_type}_q{int(q*100)}"
                studies[key] = study

                print(f"\nBest score: {study.best_value:.4f}")
                print(f"Best params: {study.best_params}")
        else:
            print(f"\n{'='*60}")
            print(f"Tuning {model_type}...")
            print(f"{'='*60}")

            study = run_optuna_study(
                model_type=model_type,
                X=train_features,
                y=data["train_target"],
                feature_cols=feature_cols,
                rates_df=data["train_rates"],
                n_trials=n_trials,
                use_quantile=False
            )

            studies[model_type] = study

            print(f"\nBest score: {study.best_value:.4f}")
            print(f"Best params: {study.best_params}")

    # Save all best parameters
    best_params = save_best_params(studies)

    # Print summary
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    for model_type, info in best_params.items():
        print(f"\n{model_type}:")
        print(f"  Best Score: {info['score']:.4f}")
        print(f"  Trials: {info['n_trials']}")

    return studies, best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of optimization trials per model"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lightgbm", "xgboost", "catboost"],
        help="Model types to tune"
    )
    parser.add_argument(
        "--quantile",
        action="store_true",
        help="Tune quantile regression models"
    )
    parser.add_argument(
        "--quantiles",
        nargs="+",
        type=float,
        default=[0.5],
        help="Quantiles to tune (if --quantile is set)"
    )

    args = parser.parse_args()

    # Suppress optuna logging except for warnings
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    studies, best_params = main(
        n_trials=args.n_trials,
        model_types=args.models,
        use_quantile=args.quantile,
        quantiles=args.quantiles
    )
