"""Data loading utilities."""

import pandas as pd
from config import (
    TRAIN_FEATURES_PATH,
    TRAIN_GT_PATH,
    TRAIN_RATES_PATH,
    TEST_FEATURES_PATH,
    ID_COLS,
    TARGET_COL,
)


def load_train_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load training features, target, and poverty rates.

    Returns:
        Tuple of (features_df, target_series, rates_df)
    """
    features = pd.read_csv(TRAIN_FEATURES_PATH)
    gt = pd.read_csv(TRAIN_GT_PATH)
    rates = pd.read_csv(TRAIN_RATES_PATH)

    # Merge features with ground truth
    df = features.merge(gt, on=ID_COLS, how="left")

    # Separate features and target
    target = df[TARGET_COL]
    features_only = df.drop(columns=[TARGET_COL])

    return features_only, target, rates


def load_test_data() -> pd.DataFrame:
    """Load test features.

    Returns:
        Test features DataFrame
    """
    return pd.read_csv(TEST_FEATURES_PATH)


def load_all_data() -> dict:
    """Load all data files.

    Returns:
        Dictionary with train_features, train_target, train_rates, test_features
    """
    train_features, train_target, train_rates = load_train_data()
    test_features = load_test_data()

    return {
        "train_features": train_features,
        "train_target": train_target,
        "train_rates": train_rates,
        "test_features": test_features,
    }


if __name__ == "__main__":
    # Test data loading
    data = load_all_data()

    print("Training Features Shape:", data["train_features"].shape)
    print("Training Target Shape:", data["train_target"].shape)
    print("Training Rates Shape:", data["train_rates"].shape)
    print("Test Features Shape:", data["test_features"].shape)

    print("\nSurveys in train:", data["train_features"]["survey_id"].unique())
    print("Surveys in test:", data["test_features"]["survey_id"].unique())

    print("\nTarget statistics:")
    print(data["train_target"].describe())
