"""Training pipeline with LOSO validation."""

import numpy as np
import pandas as pd
import json
from datetime import datetime

from data_loader import load_all_data
from preprocessing import preprocess_data, Preprocessor
from features import create_features, get_feature_columns
from models import LightGBMModel, XGBoostModel, CatBoostModel, create_models
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
    verbose: bool = True
) -> tuple[list[dict], pd.DataFrame, pd.DataFrame, Preprocessor]:
    """Main training pipeline.

    Args:
        n_estimators: Number of boosting rounds
        seeds: Random seeds for multiple runs
        run_validation: Whether to run LOSO validation
        verbose: Whether to print progress

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

    # Train models
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

    # Save results
    save_training_results(results)
    preprocessor.save()

    # Print summary
    if run_validation:
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        for result in results:
            if "validation" in result:
                val = result["validation"]
                print(f"{result['name']:30s} | Total: {val['mean_total']:6.2f} (+/- {val['std_total']:.2f})")

        # Find best model
        best = min(results, key=lambda r: r.get("validation", {}).get("mean_total", float("inf")))
        if "validation" in best:
            print(f"\nBest model: {best['name']} (score: {best['validation']['mean_total']:.2f})")

    return results, train_features, test_features, preprocessor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train poverty prediction models")
    parser.add_argument("--n-estimators", type=int, default=1000, help="Number of boosting rounds")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456], help="Random seeds")
    parser.add_argument("--no-validation", action="store_true", help="Skip LOSO validation")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    results, train_features, test_features, preprocessor = main(
        n_estimators=args.n_estimators,
        seeds=args.seeds,
        run_validation=not args.no_validation,
        verbose=not args.quiet
    )
