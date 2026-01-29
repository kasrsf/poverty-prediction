"""Training pipeline with LOSO validation."""

import numpy as np
import pandas as pd
import json
from datetime import datetime

from data_loader import load_all_data
from preprocessing import preprocess_data, Preprocessor
from features import create_features, get_feature_columns
from models import (
    LightGBMModel, XGBoostModel, CatBoostModel, create_models,
    QuantileLightGBM, QuantileXGBoost, QuantileCatBoost,
    QuantileModelWrapper, create_quantile_models, QUANTILES
)
from validation import evaluate_model_loso
from config import MODELS_DIR, OUTPUT_DIR, RANDOM_SEED


def train_single_model(
    model,
    train_features: pd.DataFrame,
    train_target: pd.Series,
    feature_cols: list[str],
    train_rates: pd.DataFrame,
    run_validation: bool = True,
    verbose: bool = True
) -> dict:
    """Train a single model and optionally validate.

    Args:
        model: Model wrapper instance
        train_features: Preprocessed training features
        train_target: Training target
        feature_cols: List of feature columns to use
        train_rates: Ground truth poverty rates
        run_validation: Whether to run LOSO validation
        verbose: Whether to print progress

    Returns:
        Dictionary with model and validation results
    """
    result = {"model": model, "name": model.name}

    if run_validation:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Validating {model.name}...")

        val_results = evaluate_model_loso(
            model.model_class(**model.params),
            train_features,
            train_target,
            feature_cols,
            train_rates,
            use_log_target=model.use_log_target,
            verbose=verbose
        )

        result["validation"] = {
            "mean_total": val_results["mean_total"],
            "std_total": val_results["std_total"],
            "weighted_total": val_results["weighted_total"],
            "fold_results": val_results["fold_results"],
        }

    # Train on full data
    if verbose:
        print(f"\nTraining {model.name} on full data...")

    model.fit(train_features, train_target, feature_cols)
    result["trained"] = True

    # Get feature importance
    importance = model.get_feature_importance()
    result["top_features"] = importance.head(20).to_dict()

    return result


def train_all_models(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    feature_cols: list[str],
    train_rates: pd.DataFrame,
    n_estimators: int = 1000,
    seeds: list[int] | None = None,
    run_validation: bool = True,
    verbose: bool = True
) -> list[dict]:
    """Train multiple models for ensembling.

    Args:
        train_features: Preprocessed training features
        train_target: Training target
        feature_cols: List of feature columns to use
        train_rates: Ground truth poverty rates
        n_estimators: Number of boosting rounds
        seeds: Random seeds for multiple runs
        run_validation: Whether to run LOSO validation
        verbose: Whether to print progress

    Returns:
        List of result dictionaries
    """
    if seeds is None:
        seeds = [RANDOM_SEED]

    all_results = []

    for seed in seeds:
        models = [
            LightGBMModel(n_estimators, random_state=seed),
            XGBoostModel(n_estimators, random_state=seed),
            CatBoostModel(n_estimators, random_state=seed),
        ]

        for model in models:
            model.name = f"{model.name}_seed{seed}"
            result = train_single_model(
                model,
                train_features,
                train_target,
                feature_cols,
                train_rates,
                run_validation=run_validation,
                verbose=verbose
            )
            all_results.append(result)

    return all_results


def train_quantile_model(
    model: QuantileModelWrapper,
    train_features: pd.DataFrame,
    train_target: pd.Series,
    feature_cols: list[str],
    train_rates: pd.DataFrame,
    run_validation: bool = True,
    verbose: bool = True
) -> dict:
    """Train a quantile regression model.

    Args:
        model: Quantile model wrapper instance
        train_features: Preprocessed training features
        train_target: Training target
        feature_cols: List of feature columns to use
        train_rates: Ground truth poverty rates
        run_validation: Whether to run LOSO validation
        verbose: Whether to print progress

    Returns:
        Dictionary with model and validation results
    """
    from metrics import competition_metric, get_true_rates_for_survey
    from config import TRAIN_SURVEYS

    result = {"model": model, "name": model.name}

    if run_validation:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Validating {model.name} with LOSO...")

        fold_results = []
        all_predictions = np.zeros(len(train_target))

        for test_survey in TRAIN_SURVEYS:
            train_mask = train_features["survey_id"] != test_survey
            test_mask = train_features["survey_id"] == test_survey

            X_train = train_features[train_mask]
            X_test = train_features[test_mask]
            y_train = train_target[train_mask]
            y_test = train_target[test_mask]

            if verbose:
                print(f"\n  Fold: Testing on survey {test_survey}")
                print(f"    Train: {len(X_train)}, Test: {len(X_test)}")

            # Create fresh model for this fold
            model_clone = model.__class__(
                quantiles=model.quantiles,
                n_estimators=model.n_estimators,
                use_log_target=model.use_log_target,
                random_state=model.base_params.get("random_state", RANDOM_SEED)
            )

            # Train
            model_clone.fit(X_train, y_train, feature_cols, verbose=False)

            # Predict (using median or mean)
            y_pred = model_clone.predict_median(X_test)
            y_pred = np.clip(y_pred, 0.1, None)

            all_predictions[test_mask] = y_pred

            # Calculate metric
            weights = X_test["weight"].values
            true_rates = get_true_rates_for_survey(train_rates, test_survey)
            metric = competition_metric(y_test.values, y_pred, weights, true_rates)

            fold_result = {
                "survey": test_survey,
                "poverty_rate_mape": metric["poverty_rate_mape"],
                "consumption_mape": metric["consumption_mape"],
                "total": metric["total"],
                "n_samples": len(X_test),
            }
            fold_results.append(fold_result)

            if verbose:
                print(f"    Poverty Rate MAPE: {metric['poverty_rate_mape']:.2f}%")
                print(f"    Consumption MAPE: {metric['consumption_mape']:.2f}%")
                print(f"    Total Score: {metric['total']:.2f}")

        # Aggregate
        totals = [f["total"] for f in fold_results]
        weights_arr = [f["n_samples"] for f in fold_results]

        result["validation"] = {
            "mean_total": np.mean(totals),
            "std_total": np.std(totals),
            "weighted_total": np.average(totals, weights=weights_arr),
            "fold_results": fold_results,
        }

        if verbose:
            print(f"\n  Mean Total Score: {result['validation']['mean_total']:.2f}")

    # Train on full data
    if verbose:
        print(f"\nTraining {model.name} on full data...")

    model.fit(train_features, train_target, feature_cols, verbose=verbose)
    result["trained"] = True

    return result


def train_quantile_models(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    feature_cols: list[str],
    train_rates: pd.DataFrame,
    n_estimators: int = 1000,
    quantiles: list[float] | None = None,
    model_types: list[str] | None = None,
    run_validation: bool = True,
    verbose: bool = True
) -> list[dict]:
    """Train quantile regression models.

    Args:
        train_features: Preprocessed training features
        train_target: Training target
        feature_cols: List of feature columns to use
        train_rates: Ground truth poverty rates
        n_estimators: Number of boosting rounds
        quantiles: List of quantiles to predict
        model_types: List of model types to train
        run_validation: Whether to run LOSO validation
        verbose: Whether to print progress

    Returns:
        List of result dictionaries
    """
    if quantiles is None:
        quantiles = QUANTILES

    if model_types is None:
        model_types = ["catboost"]  # CatBoost is fastest for quantile regression

    all_results = []

    for model_type in model_types:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training {model_type} quantile model")
            print(f"Quantiles: {len(quantiles)} ({quantiles[0]:.2f} to {quantiles[-1]:.2f})")
            print(f"{'='*60}")

        if model_type == "lightgbm":
            model = QuantileLightGBM(
                quantiles=quantiles,
                n_estimators=n_estimators,
                use_log_target=True,
                random_state=RANDOM_SEED
            )
        elif model_type == "xgboost":
            model = QuantileXGBoost(
                quantiles=quantiles,
                n_estimators=n_estimators,
                use_log_target=True,
                random_state=RANDOM_SEED
            )
        elif model_type == "catboost":
            model = QuantileCatBoost(
                quantiles=quantiles,
                n_estimators=n_estimators,
                use_log_target=True,
                random_state=RANDOM_SEED
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        result = train_quantile_model(
            model,
            train_features,
            train_target,
            feature_cols,
            train_rates,
            run_validation=run_validation,
            verbose=verbose
        )
        all_results.append(result)

    return all_results


def save_training_results(results: list[dict], output_dir=None) -> None:
    """Save training results to disk.

    Args:
        results: List of training result dictionaries
        output_dir: Output directory (defaults to OUTPUT_DIR)
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save models
    for result in results:
        model = result["model"]
        model.save(MODELS_DIR / f"{model.name}.pkl")

    # Save summary
    summary = []
    for result in results:
        entry = {
            "name": result["name"],
            "trained": result.get("trained", False),
        }
        if "validation" in result:
            entry["validation"] = {
                "mean_total": result["validation"]["mean_total"],
                "std_total": result["validation"]["std_total"],
                "weighted_total": result["validation"]["weighted_total"],
            }
        if "top_features" in result:
            entry["top_features"] = list(result["top_features"].keys())[:10]

        summary.append(entry)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = output_dir / f"training_summary_{timestamp}.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining summary saved to {summary_path}")


def main(
    n_estimators: int = 1000,
    seeds: list[int] | None = None,
    run_validation: bool = True,
    verbose: bool = True,
    use_quantile: bool = False,
    quantile_model_types: list[str] | None = None,
    quantiles: list[float] | None = None
) -> tuple[list[dict], pd.DataFrame, pd.DataFrame, Preprocessor]:
    """Main training pipeline.

    Args:
        n_estimators: Number of boosting rounds
        seeds: Random seeds for multiple runs
        run_validation: Whether to run LOSO validation
        verbose: Whether to print progress
        use_quantile: Whether to train quantile regression models
        quantile_model_types: Model types for quantile regression
        quantiles: List of quantiles to predict

    Returns:
        Tuple of (results, train_features, test_features, preprocessor)
    """
    if seeds is None:
        seeds = [42, 123, 456]

    # Load data
    print("Loading data...")
    data = load_all_data()

    # Preprocess
    print("Preprocessing...")
    train_processed, test_processed, preprocessor = preprocess_data(
        data["train_features"], data["test_features"]
    )

    # Feature engineering
    print("Creating features...")
    train_features = create_features(train_processed)
    test_features = create_features(test_processed)
    feature_cols = get_feature_columns(train_features)

    print(f"Number of features: {len(feature_cols)}")
    print(f"Training samples: {len(train_features)}")
    print(f"Test samples: {len(test_features)}")

    all_results = []

    # Train standard point prediction models
    if not use_quantile:
        results = train_all_models(
            train_features,
            data["train_target"],
            feature_cols,
            data["train_rates"],
            n_estimators=n_estimators,
            seeds=seeds,
            run_validation=run_validation,
            verbose=verbose
        )
        all_results.extend(results)
    else:
        # Train quantile regression models
        quantile_results = train_quantile_models(
            train_features,
            data["train_target"],
            feature_cols,
            data["train_rates"],
            n_estimators=n_estimators,
            quantiles=quantiles,
            model_types=quantile_model_types,
            run_validation=run_validation,
            verbose=verbose
        )
        all_results.extend(quantile_results)

    # Save results
    save_training_results(all_results)
    preprocessor.save()

    # Print summary
    if run_validation:
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        for result in all_results:
            if "validation" in result:
                val = result["validation"]
                print(f"{result['name']:30s} | Total: {val['mean_total']:6.2f} (+/- {val['std_total']:.2f})")

        # Find best model
        best = min(all_results, key=lambda r: r.get("validation", {}).get("mean_total", float("inf")))
        if "validation" in best:
            print(f"\nBest model: {best['name']} (score: {best['validation']['mean_total']:.2f})")

    return all_results, train_features, test_features, preprocessor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train poverty prediction models")
    parser.add_argument("--n-estimators", type=int, default=1000, help="Number of boosting rounds")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456], help="Random seeds")
    parser.add_argument("--no-validation", action="store_true", help="Skip LOSO validation")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--quantile", action="store_true", help="Train quantile regression models")
    parser.add_argument(
        "--quantile-models",
        nargs="+",
        default=["catboost"],
        choices=["lightgbm", "xgboost", "catboost"],
        help="Model types for quantile regression"
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=None,
        help="Specific quantiles to predict (default: 0.05 to 0.95 by 0.05)"
    )

    args = parser.parse_args()

    results, train_features, test_features, preprocessor = main(
        n_estimators=args.n_estimators,
        seeds=args.seeds,
        run_validation=not args.no_validation,
        verbose=not args.quiet,
        use_quantile=args.quantile,
        quantile_model_types=args.quantile_models,
        quantiles=args.quantiles
    )
