# Poverty Prediction Competition

A machine learning pipeline for household consumption imputation and poverty rate prediction from survey data.

## Project Overview

This project tackles a poverty prediction competition that simulates a real-world challenge in poverty monitoring: imputing household consumption from survey features when detailed consumption modules are unavailable.

### Competition Background

- **Task**: Predict household-level consumption and aggregate poverty rates for test surveys
- **Data**: 6 household surveys (~35,000 households each)
  - Training: Surveys 100000, 200000, 300000 (with consumption labels)
  - Test: Surveys 400000, 500000, 600000 (features only)
- **Metric**: 90% weighted poverty rate MAPE + 10% consumption MAPE
- **Challenge**: Models must generalize to unseen survey populations

### Results Summary

| Submission | CV Score | Test Score | Approach |
|------------|----------|------------|----------|
| CatBoost Ensemble | 9.14 | 9.353 | Point prediction with GBDT |
| Quantile Regression | ~9.44 | 10.024 | Distribution modeling |
| Neural Network (MDN) | 7.72 | 12.038 | Mixture density network |
| **Competition Leader** | - | **3.207** | Unknown |

## Installation & Setup

### Prerequisites

- Python 3.10+
- pip or conda

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd poverty-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Packages

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
xgboost>=2.0.0
catboost>=1.2.0
torch>=2.0.0
scipy>=1.11.0
```

## Project Structure

```
poverty-prediction/
├── data/
│   ├── train_hh_features.csv    # Training survey features
│   ├── train_hh_gt.csv          # Training consumption labels
│   ├── train_rates_gt.csv       # Ground truth poverty rates
│   ├── test_hh_features.csv     # Test survey features
│   ├── feature_descriptions.csv # Data dictionary
│   └── feature_value_descriptions.csv
├── src/
│   ├── config.py                # Constants and hyperparameters
│   ├── data_loader.py           # Data loading utilities
│   ├── preprocessing.py         # Feature preprocessing
│   ├── features.py              # Feature engineering
│   ├── models.py                # Model wrappers (point & quantile)
│   ├── neural_models.py         # PyTorch models (MDN, etc.)
│   ├── metrics.py               # Competition metric implementation
│   ├── validation.py            # LOSO cross-validation
│   ├── ensemble.py              # Ensemble methods
│   ├── poverty_rates.py         # Rate calculation utilities
│   ├── rate_optimizer.py        # Rate optimization
│   ├── train.py                 # Training pipeline
│   ├── train_neural.py          # Neural network training
│   ├── submit.py                # Submission generation
│   └── submit_neural.py         # Neural submission generation
├── outputs/
│   ├── models/                  # Saved model files
│   └── submissions/             # Generated submission files
├── docs/
│   ├── paper.tex               # LaTeX research paper
│   ├── README.md               # This documentation
│   └── index.html              # Interactive dashboard
└── problem_description.md      # Competition problem statement
```

## Quick Start Guide

### 1. Train Models

```bash
cd src

# Train gradient boosting models with LOSO validation
python train.py --n-estimators 1000 --seeds 42 123

# Train quantile regression models
python train.py --quantile --quantile-models catboost

# Train neural network models
python train_neural.py --model-type mdn --n-seeds 5
```

### 2. Generate Submissions

```bash
# Standard submission (point prediction ensemble)
python submit.py --mode standard

# Quantile regression submission
python submit.py --mode quantile

# Neural network submission
python submit_neural.py
```

### 3. View Results

Open `docs/index.html` in a browser to see interactive visualizations of model performance and feature analysis.

## Module Documentation

### config.py

Central configuration file containing:

- **Paths**: Project directories, data file locations
- **Constants**: Poverty thresholds, survey IDs, feature groups
- **Hyperparameters**: Default parameters for LightGBM, XGBoost, CatBoost

```python
from config import POVERTY_THRESHOLDS, TRAIN_SURVEYS, LIGHTGBM_PARAMS
```

### data_loader.py

Data loading utilities for training and test data:

```python
from data_loader import load_all_data

data = load_all_data()
# Returns: {train_features, train_target, train_rates, test_features}
```

### preprocessing.py

Feature preprocessing pipeline:

- Binary categorical encoding
- Multi-class label encoding
- Missing value imputation
- Consumed indicator encoding

```python
from preprocessing import Preprocessor, preprocess_data

train_processed, test_processed, preprocessor = preprocess_data(train_df, test_df)
preprocessor.save()  # Save for later use
```

### features.py

Feature engineering module creating 30+ derived features:

```python
from features import create_features, get_feature_columns

train_features = create_features(train_processed)
feature_cols = get_feature_columns(train_features)
```

**Engineered Features:**
- `children5_ratio`, `elderly_ratio`, `adults_ratio` - Household composition
- `food_diversity`, `food_diversity_ratio` - Dietary variety
- `infrastructure_index` - Utilities availability
- `workers_per_adult`, `formal_worker_ratio` - Employment quality
- `log_utl_exp`, `log_utl_exp_per_person` - Log-transformed expenditure
- `hsize_squared`, `large_household` - Household size features
- `high_strata`, `low_strata` - Stratum indicators
- `dependency_ratio` - Economic dependency

### models.py

Model wrappers for gradient boosting and quantile regression:

**Point Prediction Models:**
```python
from models import LightGBMModel, XGBoostModel, CatBoostModel

model = CatBoostModel(n_estimators=1000, use_log_target=True)
model.fit(X_train, y_train, feature_cols)
predictions = model.predict(X_test)
importance = model.get_feature_importance()
```

**Quantile Regression Models:**
```python
from models import QuantileCatBoost, QUANTILES

qmodel = QuantileCatBoost(quantiles=QUANTILES, n_estimators=1000)
qmodel.fit(X_train, y_train, feature_cols)
quantile_preds = qmodel.predict(X_test)  # Dict[quantile, predictions]
median_pred = qmodel.predict_median(X_test)
```

### neural_models.py

PyTorch neural network implementations:

**Mixture Density Network (MDN):**
```python
from neural_models import MixtureDensityNetwork, NeuralModelWrapper

wrapper = NeuralModelWrapper(
    model_class=MixtureDensityNetwork,
    hidden_dim=256,
    n_epochs=100
)
wrapper.fit(X_train, y_train, feature_cols)
predictions = wrapper.predict(X_test)
```

**Model Architectures:**
- `MixtureDensityNetwork`: 5-component Gaussian mixture for distribution prediction
- `ConsumptionPredictor`: Standard feed-forward network for point prediction
- `DirectRatePredictor`: Directly predicts poverty rates (experimental)

### metrics.py

Competition metric implementation:

```python
from metrics import (
    competition_metric,
    calculate_poverty_rates,
    weighted_poverty_rate_mape,
    consumption_mape
)

# Calculate full competition metric
result = competition_metric(y_true, y_pred, weights, true_rates)
# Returns: {total, poverty_rate_mape, consumption_mape, pred_rates, true_rates}

# Calculate poverty rates from predictions
rates = calculate_poverty_rates(predictions, weights, thresholds)
```

**Metric Formula:**
```
Score = 0.9 * Rate_Error + 0.1 * Consumption_Error
w_t = 1 - |p_t - 0.4|  # Weight for threshold t
```

### validation.py

Leave-One-Survey-Out cross-validation:

```python
from validation import LeaveOneSurveyOut, evaluate_model_loso

# Manual CV split
loso = LeaveOneSurveyOut()
for train_idx, test_idx in loso.split(X):
    # train_idx: indices for 2 surveys
    # test_idx: indices for held-out survey

# Full evaluation
results = evaluate_model_loso(model, X, y, feature_cols, rates_df)
# Returns: fold_results, mean_total, std_total, weighted_total
```

### ensemble.py

Ensemble methods for combining models:

```python
from ensemble import Ensemble, QuantileEnsemble, optimize_ensemble_weights

# Create weighted ensemble
ensemble = Ensemble(models, weights)
predictions = ensemble.predict(X_test)

# Optimize weights via LOSO
optimal_weights, score = optimize_ensemble_weights(
    models, X, y, feature_cols, rates_df
)

# Quantile ensemble for distribution modeling
q_ensemble = QuantileEnsemble(quantile_models)
rates = q_ensemble.predict_poverty_rates(X, weights, thresholds)
```

### train.py

Main training pipeline:

```bash
# Command line usage
python train.py --n-estimators 1000 --seeds 42 123 456
python train.py --quantile --quantile-models catboost lightgbm
python train.py --no-validation  # Skip LOSO CV
```

```python
# Programmatic usage
from train import main, train_all_models

results, train_features, test_features, preprocessor = main(
    n_estimators=1000,
    seeds=[42, 123],
    run_validation=True
)
```

### submit.py

Submission file generation:

```bash
# Generate submissions
python submit.py --mode standard --tag v1
python submit.py --mode quantile --no-rate-optimization
python submit.py --mode hybrid --point-weight 0.6
```

**Output Files:**
- `predicted_household_consumption_{tag}_{timestamp}.csv`
- `predicted_poverty_distribution_{tag}_{timestamp}.csv`

## Usage Examples

### Example 1: Basic Training and Submission

```python
from data_loader import load_all_data
from preprocessing import preprocess_data
from features import create_features, get_feature_columns
from models import CatBoostModel
from submit import generate_submission

# Load and preprocess
data = load_all_data()
train_proc, test_proc, preprocessor = preprocess_data(
    data["train_features"], data["test_features"]
)

# Feature engineering
train_features = create_features(train_proc)
test_features = create_features(test_proc)
feature_cols = get_feature_columns(train_features)

# Train model
model = CatBoostModel(n_estimators=1000)
model.fit(train_features, data["train_target"], feature_cols)

# Generate predictions
predictions = model.predict(test_features)
```

### Example 2: LOSO Cross-Validation

```python
from validation import evaluate_model_loso
from models import CatBoostModel

model = CatBoostModel(n_estimators=500)
results = evaluate_model_loso(
    model.model_class(**model.params),
    train_features,
    data["train_target"],
    feature_cols,
    data["train_rates"],
    use_log_target=True,
    verbose=True
)

print(f"CV Score: {results['mean_total']:.2f} (+/- {results['std_total']:.2f})")
```

### Example 3: Quantile Regression

```python
from models import QuantileCatBoost, QUANTILES
from ensemble import QuantileEnsemble

# Train quantile model
qmodel = QuantileCatBoost(quantiles=QUANTILES, n_estimators=1000)
qmodel.fit(train_features, data["train_target"], feature_cols)

# Predict poverty rates
ensemble = QuantileEnsemble([qmodel])
rates = ensemble.predict_poverty_rates(
    test_features,
    test_features["weight"].values,
    POVERTY_THRESHOLDS
)
```

## Results Summary

### Model Performance (LOSO CV)

| Model | Mean Score | Std | Top Features |
|-------|------------|-----|--------------|
| LightGBM | 10.18 | 0.91 | weight, utl_exp_ppp17, age |
| XGBoost | 10.15 | 0.97 | high_strata, urban, utl_exp_per_person |
| **CatBoost** | **9.14** | 1.03 | log_utl_exp_per_person, sfworkershh |
| MDN (best) | **7.72** | 0.82 | (neural network) |

### Competition Submissions

| Approach | CV | Test | Notes |
|----------|-----|------|-------|
| CatBoost Ensemble | 9.14 | 9.353 | Best generalization |
| Quantile Regression | ~9.44 | 10.024 | Distribution modeling |
| MDN | 7.72 | 12.038 | Overfit to training |

### Key Insights

1. **CatBoost** consistently outperformed LightGBM and XGBoost on this task
2. **Utility expenditure per person** was the strongest predictor across models
3. **Food diversity** and **formal employment** were strong poverty indicators
4. **MDN** achieved best CV but worst test score - significant overfitting
5. **Distribution modeling** didn't improve over point prediction for this metric

## License

This project was developed for research purposes as part of a poverty prediction competition.
