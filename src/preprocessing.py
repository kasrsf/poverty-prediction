"""Data preprocessing: encoding, missing value handling."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

from config import (
    BINARY_CATEGORICALS,
    MULTI_CLASS_CATEGORICALS,
    CONSUMED_COLS,
    REGION_COLS,
    NUMERICAL_FEATURES,
    ID_COLS,
    MODELS_DIR,
)


class Preprocessor:
    """Handles all preprocessing transformations."""

    def __init__(self):
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.binary_mappings: dict[str, dict] = {}
        self.feature_columns: list[str] = []
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """Fit preprocessor on training data."""
        df = df.copy()

        # Fit binary encodings
        for col in BINARY_CATEGORICALS:
            if col in df.columns:
                unique_vals = df[col].dropna().unique()
                # Create mapping: first alphabetically = 0, second = 1
                sorted_vals = sorted(unique_vals)
                self.binary_mappings[col] = {v: i for i, v in enumerate(sorted_vals)}

        # Fit label encoders for multi-class categoricals
        for col in MULTI_CLASS_CATEGORICALS:
            if col in df.columns:
                le = LabelEncoder()
                # Handle missing values by adding a placeholder
                values = df[col].fillna("__MISSING__").astype(str)
                le.fit(values)
                self.label_encoders[col] = le

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features."""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        df = df.copy()

        # Encode binary categoricals
        for col, mapping in self.binary_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(-1).astype(int)

        # Encode multi-class categoricals
        for col, le in self.label_encoders.items():
            if col in df.columns:
                values = df[col].fillna("__MISSING__").astype(str)
                # Handle unseen categories
                df[col] = values.apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        # Encode consumed columns (Yes/No -> 1/0)
        for col in CONSUMED_COLS:
            if col in df.columns:
                df[col] = (df[col] == "Yes").astype(int)

        # Region columns are already 0/1

        # Handle missing values in numerical features
        for col in NUMERICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)

        # Store feature columns (excluding IDs)
        self.feature_columns = [c for c in df.columns if c not in ID_COLS]

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def save(self, path: str | None = None) -> None:
        """Save preprocessor to disk."""
        if path is None:
            path = MODELS_DIR / "preprocessor.pkl"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | None = None) -> "Preprocessor":
        """Load preprocessor from disk."""
        if path is None:
            path = MODELS_DIR / "preprocessor.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)


def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Preprocessor]:
    """Preprocess train and test data.

    Args:
        train_df: Training features DataFrame
        test_df: Test features DataFrame

    Returns:
        Tuple of (processed_train, processed_test, preprocessor)
    """
    preprocessor = Preprocessor()

    # Fit on training data
    train_processed = preprocessor.fit_transform(train_df)

    # Transform test data
    test_processed = preprocessor.transform(test_df)

    return train_processed, test_processed, preprocessor


if __name__ == "__main__":
    from data_loader import load_all_data

    print("Loading data...")
    data = load_all_data()

    print("\nPreprocessing...")
    train_processed, test_processed, preprocessor = preprocess_data(
        data["train_features"], data["test_features"]
    )

    print("\nProcessed train shape:", train_processed.shape)
    print("Processed test shape:", test_processed.shape)

    # Check for any remaining object columns
    train_obj_cols = train_processed.select_dtypes(include=["object"]).columns.tolist()
    print(f"\nRemaining object columns in train: {train_obj_cols}")

    # Check for missing values
    train_missing = train_processed.isnull().sum()
    cols_with_missing = train_missing[train_missing > 0]
    print(f"\nColumns with missing values: {cols_with_missing.to_dict()}")

    # Save preprocessor
    preprocessor.save()
    print("\nPreprocessor saved.")

    # Sample of processed data
    print("\nSample of processed numerical features:")
    print(train_processed[["strata", "utl_exp_ppp17", "hsize", "male", "urban"]].head())
