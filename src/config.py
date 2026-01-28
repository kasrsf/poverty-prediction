"""Configuration constants for the poverty prediction project."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
SUBMISSIONS_DIR = OUTPUT_DIR / "submissions"

# Data files
TRAIN_FEATURES_PATH = DATA_DIR / "train_hh_features.csv"
TRAIN_GT_PATH = DATA_DIR / "train_hh_gt.csv"
TRAIN_RATES_PATH = DATA_DIR / "train_rates_gt.csv"
TEST_FEATURES_PATH = DATA_DIR / "test_hh_features.csv"

# Poverty thresholds (ventiles from survey 300000)
POVERTY_THRESHOLDS = [
    3.17, 3.94, 4.60, 5.26, 5.88, 6.47, 7.06, 7.70,
    8.40, 9.13, 9.87, 10.70, 11.62, 12.69, 14.03,
    15.64, 17.76, 20.99, 27.37
]

# Column names for poverty rates (must match exact format in data files)
POVERTY_THRESHOLD_STRS = [
    "3.17", "3.94", "4.60", "5.26", "5.88", "6.47", "7.06", "7.70",
    "8.40", "9.13", "9.87", "10.70", "11.62", "12.69", "14.03",
    "15.64", "17.76", "20.99", "27.37"
]
POVERTY_RATE_COLS = [f"pct_hh_below_{t}" for t in POVERTY_THRESHOLD_STRS]

# Map from threshold value to string for column lookup
THRESHOLD_TO_STR = {float(s): s for s in POVERTY_THRESHOLD_STRS}

# Survey IDs
TRAIN_SURVEYS = [100000, 200000, 300000]
TEST_SURVEYS = [400000, 500000, 600000]

# Feature groups
BINARY_CATEGORICALS = [
    "male", "owner", "water", "toilet", "sewer", "elect",
    "urban", "employed", "any_nonagric"
]

MULTI_CLASS_CATEGORICALS = [
    "water_source", "sanitation_source", "dweltyp", "sector1d", "educ_max"
]

# Regions (one-hot encoded in original data)
REGION_COLS = ["region1", "region2", "region3", "region4", "region5", "region6", "region7"]

# Food consumption indicators
CONSUMED_COLS = [f"consumed{i}00" for i in range(1, 51)]

# Key numerical features
NUMERICAL_FEATURES = [
    "weight", "strata", "utl_exp_ppp17", "hsize",
    "num_children5", "num_children10", "num_children18",
    "age", "num_adult_female", "num_adult_male", "num_elderly",
    "sworkershh", "share_secondary", "sfworkershh"
]

# Target column
TARGET_COL = "cons_ppp17"

# ID columns
ID_COLS = ["survey_id", "hhid"]

# Random seed for reproducibility
RANDOM_SEED = 42

# Cross-validation folds for LOSO
N_FOLDS = 3  # One per survey

# Model hyperparameters (defaults)
LIGHTGBM_PARAMS = {
    "objective": "regression",
    "metric": "mape",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}

XGBOOST_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "mape",
    "learning_rate": 0.05,
    "max_depth": 7,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}

CATBOOST_PARAMS = {
    "loss_function": "MAPE",
    "learning_rate": 0.05,
    "depth": 7,
    "l2_leaf_reg": 3,
    "random_seed": RANDOM_SEED,
    "verbose": False,
}
