"""Feature engineering module."""

import pandas as pd
import numpy as np

from config import CONSUMED_COLS, REGION_COLS


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features.

    Args:
        df: Preprocessed DataFrame

    Returns:
        DataFrame with additional engineered features
    """
    df = df.copy()

    # 1. Household composition ratios
    df["children5_ratio"] = df["num_children5"] / df["hsize"].clip(lower=1)
    df["children10_ratio"] = df["num_children10"] / df["hsize"].clip(lower=1)
    df["children18_ratio"] = df["num_children18"] / df["hsize"].clip(lower=1)
    df["elderly_ratio"] = df["num_elderly"] / df["hsize"].clip(lower=1)

    # Adults = hsize - children under 18
    df["num_adults"] = df["hsize"] - df["num_children18"]
    df["adults_ratio"] = df["num_adults"] / df["hsize"].clip(lower=1)

    # Workers per adult
    total_workers = (df["num_adult_female"] + df["num_adult_male"]) * df["sworkershh"]
    df["workers_per_adult"] = total_workers / df["num_adults"].clip(lower=1)

    # 2. Food diversity score (sum of consumed indicators)
    consumed_cols_present = [c for c in CONSUMED_COLS if c in df.columns]
    df["food_diversity"] = df[consumed_cols_present].sum(axis=1)
    df["food_diversity_ratio"] = df["food_diversity"] / len(consumed_cols_present)

    # 3. Infrastructure index
    infra_cols = ["water", "toilet", "sewer", "elect"]
    infra_cols_present = [c for c in infra_cols if c in df.columns]
    df["infrastructure_index"] = df[infra_cols_present].sum(axis=1)
    df["infrastructure_ratio"] = df["infrastructure_index"] / len(infra_cols_present)

    # 4. Region interactions with urban
    if "urban" in df.columns:
        for region_col in REGION_COLS:
            if region_col in df.columns:
                df[f"{region_col}_x_urban"] = df[region_col] * df["urban"]

    # 5. Employment quality features
    df["formal_worker_ratio"] = df["sfworkershh"] * df["employed"]
    df["any_formal_worker"] = (df["sfworkershh"] > 0).astype(int)

    # 6. Log transforms for skewed features
    if "utl_exp_ppp17" in df.columns:
        df["log_utl_exp"] = np.log1p(df["utl_exp_ppp17"])

    if "weight" in df.columns:
        df["log_weight"] = np.log1p(df["weight"])

    # 7. Household size features
    df["hsize_squared"] = df["hsize"] ** 2
    df["large_household"] = (df["hsize"] >= 5).astype(int)
    df["single_person"] = (df["hsize"] == 1).astype(int)

    # 8. Age-related features
    df["age_squared"] = df["age"] ** 2
    df["elderly_head"] = (df["age"] >= 65).astype(int)
    df["young_head"] = (df["age"] <= 30).astype(int)

    # 9. Education and employment interaction
    if "educ_max" in df.columns and "employed" in df.columns:
        df["educ_x_employed"] = df["educ_max"] * df["employed"]

    # 10. Dependency ratio (non-workers / workers)
    df["potential_workers"] = df["num_adult_female"] + df["num_adult_male"]
    df["dependents"] = df["num_children18"] + df["num_elderly"]
    df["dependency_ratio"] = df["dependents"] / df["potential_workers"].clip(lower=1)

    # 11. Gender balance in household
    df["female_adult_ratio"] = df["num_adult_female"] / df["potential_workers"].clip(lower=1)

    # 12. Strata-based features (strata correlates strongly with target)
    df["high_strata"] = (df["strata"] >= 6).astype(int)
    df["low_strata"] = (df["strata"] <= 2).astype(int)

    # 13. Utility expense per person
    if "utl_exp_ppp17" in df.columns:
        df["utl_exp_per_person"] = df["utl_exp_ppp17"] / df["hsize"].clip(lower=1)
        df["log_utl_exp_per_person"] = np.log1p(df["utl_exp_per_person"])

    # 14. Survey-level features (survey ID as categorical)
    # Already included as survey_id column

    return df


def get_feature_columns(df: pd.DataFrame, exclude_cols: list[str] | None = None) -> list[str]:
    """Get list of feature columns for modeling.

    Args:
        df: DataFrame with features
        exclude_cols: Columns to exclude

    Returns:
        List of feature column names
    """
    if exclude_cols is None:
        exclude_cols = ["survey_id", "hhid", "com"]

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    return feature_cols


if __name__ == "__main__":
    from data_loader import load_all_data
    from preprocessing import preprocess_data

    print("Loading data...")
    data = load_all_data()

    print("Preprocessing...")
    train_processed, test_processed, preprocessor = preprocess_data(
        data["train_features"], data["test_features"]
    )

    print("Creating features...")
    train_features = create_features(train_processed)
    test_features = create_features(test_processed)

    print(f"\nTrain shape after feature engineering: {train_features.shape}")
    print(f"Test shape after feature engineering: {test_features.shape}")

    # Show new features
    original_cols = set(train_processed.columns)
    new_cols = [c for c in train_features.columns if c not in original_cols]
    print(f"\nNew features created ({len(new_cols)}):")
    for col in sorted(new_cols):
        print(f"  - {col}")

    # Feature statistics
    feature_cols = get_feature_columns(train_features)
    print(f"\nTotal feature columns for modeling: {len(feature_cols)}")

    # Check for any issues
    print("\nChecking for infinite values...")
    inf_counts = train_features[feature_cols].apply(lambda x: np.isinf(x).sum())
    inf_cols = inf_counts[inf_counts > 0]
    if len(inf_cols) > 0:
        print(f"Columns with inf values: {inf_cols.to_dict()}")
    else:
        print("No infinite values found.")
