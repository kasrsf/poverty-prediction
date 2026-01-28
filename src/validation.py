"""Cross-validation strategies for model evaluation."""

import numpy as np
import pandas as pd
from typing import Iterator

from config import TRAIN_SURVEYS
from metrics import competition_metric, calculate_poverty_rates, get_true_rates_for_survey


class LeaveOneSurveyOut:
    """Leave-One-Survey-Out cross-validation.

    This CV strategy simulates the test scenario where we train on
    some surveys and predict on unseen surveys.
    """

    def __init__(self, surveys: list[int] | None = None):
        """Initialize LOSO CV.

        Args:
            surveys: List of survey IDs. Defaults to TRAIN_SURVEYS.
        """
        self.surveys = surveys if surveys is not None else TRAIN_SURVEYS

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices for each fold.

        Args:
            X: Features DataFrame (must contain 'survey_id' column)
            y: Target series (unused, for sklearn compatibility)

        Yields:
            Tuple of (train_indices, test_indices) for each fold
        """
        survey_ids = X["survey_id"].values

        for test_survey in self.surveys:
            test_mask = survey_ids == test_survey
            train_mask = ~test_mask

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        """Return number of folds."""
        return len(self.surveys)


def evaluate_model_loso(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    rates_df: pd.DataFrame,
    use_log_target: bool = True,
    verbose: bool = True
) -> dict:
    """Evaluate model using Leave-One-Survey-Out CV.

    Args:
        model: Model with fit() and predict() methods
        X: Features DataFrame
        y: Target series
        feature_cols: List of feature column names to use
        rates_df: DataFrame with ground truth poverty rates
        use_log_target: Whether to use log-transformed target
        verbose: Whether to print progress

    Returns:
        Dictionary with fold results and overall metrics
    """
    loso = LeaveOneSurveyOut()
    fold_results = []
    all_predictions = np.zeros(len(y))

    for fold_idx, (train_idx, test_idx) in enumerate(loso.split(X)):
        test_survey = X.iloc[test_idx]["survey_id"].iloc[0]

        if verbose:
            print(f"\nFold {fold_idx + 1}: Testing on survey {test_survey}")
            print(f"  Train size: {len(train_idx)}, Test size: {len(test_idx)}")

        # Prepare data
        X_train = X.iloc[train_idx][feature_cols]
        X_test = X.iloc[test_idx][feature_cols]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Transform target if using log
        if use_log_target:
            y_train_fit = np.log1p(y_train)
        else:
            y_train_fit = y_train

        # Clone model for fresh training
        model_clone = model.__class__(**model.get_params())

        # Fit model
        model_clone.fit(X_train, y_train_fit)

        # Predict
        if use_log_target:
            y_pred = np.expm1(model_clone.predict(X_test))
        else:
            y_pred = model_clone.predict(X_test)

        # Ensure positive predictions
        y_pred = np.clip(y_pred, 0.1, None)

        # Store predictions
        all_predictions[test_idx] = y_pred

        # Get weights
        weights = X.iloc[test_idx]["weight"].values

        # Get true rates for this survey
        true_rates = get_true_rates_for_survey(rates_df, test_survey)

        # Calculate metric
        metric = competition_metric(y_test.values, y_pred, weights, true_rates)

        fold_result = {
            "fold": fold_idx,
            "survey": test_survey,
            "poverty_rate_mape": metric["poverty_rate_mape"],
            "consumption_mape": metric["consumption_mape"],
            "total": metric["total"],
            "n_samples": len(test_idx),
        }
        fold_results.append(fold_result)

        if verbose:
            print(f"  Poverty Rate MAPE: {metric['poverty_rate_mape']:.2f}%")
            print(f"  Consumption MAPE: {metric['consumption_mape']:.2f}%")
            print(f"  Total Score: {metric['total']:.2f}")

    # Aggregate results
    results_df = pd.DataFrame(fold_results)

    # Weighted average by number of samples
    total_samples = results_df["n_samples"].sum()
    weighted_avg = lambda col: (results_df[col] * results_df["n_samples"]).sum() / total_samples

    summary = {
        "fold_results": fold_results,
        "mean_poverty_rate_mape": results_df["poverty_rate_mape"].mean(),
        "mean_consumption_mape": results_df["consumption_mape"].mean(),
        "mean_total": results_df["total"].mean(),
        "weighted_poverty_rate_mape": weighted_avg("poverty_rate_mape"),
        "weighted_consumption_mape": weighted_avg("consumption_mape"),
        "weighted_total": weighted_avg("total"),
        "std_total": results_df["total"].std(),
        "all_predictions": all_predictions,
    }

    if verbose:
        print("\n" + "=" * 50)
        print("LOSO CV Summary:")
        print(f"  Mean Total Score: {summary['mean_total']:.2f} (+/- {summary['std_total']:.2f})")
        print(f"  Weighted Total Score: {summary['weighted_total']:.2f}")

    return summary


def cross_validate_quick(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    n_splits: int = 5,
    use_log_target: bool = True
) -> dict:
    """Quick cross-validation using stratified K-fold by survey.

    This is faster than LOSO but less representative of test scenario.

    Args:
        model: Model with fit() and predict() methods
        X: Features DataFrame
        y: Target series
        feature_cols: List of feature column names
        n_splits: Number of CV folds
        use_log_target: Whether to use log-transformed target

    Returns:
        Dictionary with CV results
    """
    from sklearn.model_selection import StratifiedKFold

    # Stratify by survey_id
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    mape_scores = []
    for train_idx, test_idx in skf.split(X, X["survey_id"]):
        X_train = X.iloc[train_idx][feature_cols]
        X_test = X.iloc[test_idx][feature_cols]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        if use_log_target:
            y_train_fit = np.log1p(y_train)
        else:
            y_train_fit = y_train

        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_train, y_train_fit)

        if use_log_target:
            y_pred = np.expm1(model_clone.predict(X_test))
        else:
            y_pred = model_clone.predict(X_test)

        y_pred = np.clip(y_pred, 0.1, None)

        # Simple MAPE for consumption
        mape = np.mean(np.abs(y_pred - y_test) / y_test) * 100
        mape_scores.append(mape)

    return {
        "mean_mape": np.mean(mape_scores),
        "std_mape": np.std(mape_scores),
        "scores": mape_scores,
    }


if __name__ == "__main__":
    from data_loader import load_all_data
    from preprocessing import preprocess_data
    from features import create_features, get_feature_columns
    from lightgbm import LGBMRegressor
    from config import LIGHTGBM_PARAMS

    print("Loading and preprocessing data...")
    data = load_all_data()

    train_processed, _, _ = preprocess_data(
        data["train_features"], data["test_features"]
    )
    train_features = create_features(train_processed)
    feature_cols = get_feature_columns(train_features)

    print(f"Number of features: {len(feature_cols)}")

    # Quick validation
    print("\n" + "=" * 50)
    print("Quick 5-fold CV (stratified by survey):")
    model = LGBMRegressor(**LIGHTGBM_PARAMS, n_estimators=100)
    quick_results = cross_validate_quick(
        model, train_features, data["train_target"], feature_cols
    )
    print(f"Consumption MAPE: {quick_results['mean_mape']:.2f}% (+/- {quick_results['std_mape']:.2f}%)")

    # Full LOSO validation
    print("\n" + "=" * 50)
    print("Leave-One-Survey-Out CV (simulates test scenario):")
    loso_results = evaluate_model_loso(
        model, train_features, data["train_target"],
        feature_cols, data["train_rates"]
    )
