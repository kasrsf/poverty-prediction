"""Training pipeline for neural network models.

This module handles training of neural network models with:
- LOSO cross-validation
- Early stopping
- Learning rate scheduling
- Ensemble of multiple seeds
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import json
import argparse

from data_loader import load_all_data
from preprocessing import preprocess_data
from features import create_features, get_feature_columns
from neural_models import (
    PovertyDataset,
    MixtureDensityNetwork,
    ConsumptionPredictor,
    DirectRatePredictor,
    MDNLoss,
    NeuralModelWrapper,
    create_neural_models
)
from metrics import competition_metric, get_true_rates_for_survey
from config import (
    MODELS_DIR, OUTPUT_DIR, RANDOM_SEED, TRAIN_SURVEYS, POVERTY_THRESHOLDS
)


def train_neural_model_loso(
    model_wrapper: NeuralModelWrapper,
    train_features: pd.DataFrame,
    train_target: pd.Series,
    feature_cols: list[str],
    train_rates: pd.DataFrame,
    verbose: bool = True
) -> dict:
    """Train and validate neural model using LOSO CV.

    Args:
        model_wrapper: Neural model wrapper
        train_features: Training features
        train_target: Training target
        feature_cols: Feature columns
        train_rates: Ground truth rates
        verbose: Print progress

    Returns:
        Dictionary with validation results
    """
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
        fold_model = NeuralModelWrapper(
            model_class=model_wrapper.model_class,
            input_dim=len(feature_cols),
            hidden_dim=model_wrapper.hidden_dim,
            learning_rate=model_wrapper.learning_rate,
            n_epochs=model_wrapper.n_epochs,
            patience=model_wrapper.patience,
            name=f"{model_wrapper.name}_fold{test_survey}"
        )

        # Train
        fold_model.fit(X_train, y_train, feature_cols, verbose=False)

        # Predict
        y_pred = fold_model.predict(X_test)
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

    return {
        "fold_results": fold_results,
        "mean_total": np.mean(totals),
        "std_total": np.std(totals),
        "weighted_total": np.average(totals, weights=weights_arr),
        "all_predictions": all_predictions,
    }


def train_neural_ensemble(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    feature_cols: list[str],
    train_rates: pd.DataFrame,
    model_type: str = "mdn",
    n_seeds: int = 5,
    n_epochs: int = 100,
    hidden_dim: int = 256,
    learning_rate: float = 1e-3,
    run_validation: bool = True,
    verbose: bool = True
) -> list[dict]:
    """Train ensemble of neural network models.

    Args:
        train_features: Training features
        train_target: Training target
        feature_cols: Feature columns
        train_rates: Ground truth rates
        model_type: Type of neural network ('mdn', 'consumption', 'direct_rate')
        n_seeds: Number of models to train
        n_epochs: Training epochs
        hidden_dim: Hidden layer dimension
        learning_rate: Learning rate
        run_validation: Run LOSO validation
        verbose: Print progress

    Returns:
        List of training results
    """
    input_dim = len(feature_cols)
    all_results = []

    for seed in range(n_seeds):
        torch.manual_seed(RANDOM_SEED + seed)
        np.random.seed(RANDOM_SEED + seed)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Training {model_type} model (seed {seed})")
            print(f"{'='*60}")

        # Select model class
        if model_type == "mdn":
            model_class = MixtureDensityNetwork
        elif model_type == "consumption":
            model_class = ConsumptionPredictor
        elif model_type == "direct_rate":
            model_class = DirectRatePredictor
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model_wrapper = NeuralModelWrapper(
            model_class=model_class,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            patience=15,
            name=f"{model_type}_seed{seed}"
        )

        result = {"model": model_wrapper, "name": model_wrapper.name}

        # Validation
        if run_validation:
            val_results = train_neural_model_loso(
                model_wrapper,
                train_features,
                train_target,
                feature_cols,
                train_rates,
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
            print(f"\nTraining {model_wrapper.name} on full data...")

        model_wrapper.fit(
            train_features,
            train_target,
            feature_cols,
            verbose=verbose
        )
        result["trained"] = True

        all_results.append(result)

    return all_results


def save_neural_results(results: list[dict], output_dir=None):
    """Save neural network training results."""
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
        summary.append(entry)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = output_dir / f"neural_training_summary_{timestamp}.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nNeural training summary saved to {summary_path}")


def main(
    model_type: str = "mdn",
    n_seeds: int = 5,
    n_epochs: int = 100,
    hidden_dim: int = 256,
    learning_rate: float = 1e-3,
    run_validation: bool = True,
    verbose: bool = True
):
    """Main neural network training pipeline.

    Args:
        model_type: Type of neural network
        n_seeds: Number of models to train
        n_epochs: Training epochs
        hidden_dim: Hidden layer dimension
        learning_rate: Learning rate
        run_validation: Run LOSO validation
        verbose: Print progress
    """
    print("=" * 60)
    print(f"NEURAL NETWORK TRAINING ({model_type})")
    print("=" * 60)

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("\nLoading data...")
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

    # Train neural networks
    results = train_neural_ensemble(
        train_features,
        data["train_target"],
        feature_cols,
        data["train_rates"],
        model_type=model_type,
        n_seeds=n_seeds,
        n_epochs=n_epochs,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        run_validation=run_validation,
        verbose=verbose
    )

    # Save results
    save_neural_results(results)

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

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural network models")
    parser.add_argument(
        "--model-type",
        type=str,
        default="mdn",
        choices=["mdn", "consumption", "direct_rate"],
        help="Type of neural network"
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Number of models to train"
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=100,
        help="Training epochs"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden layer dimension"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip LOSO validation"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    main(
        model_type=args.model_type,
        n_seeds=args.n_seeds,
        n_epochs=args.n_epochs,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        run_validation=not args.no_validation,
        verbose=not args.quiet
    )
