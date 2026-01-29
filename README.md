# Poverty Prediction Competition

A machine learning pipeline for household consumption imputation and poverty rate prediction from survey data.

## Project Overview

This project tackles a poverty prediction competition that simulates a real-world challenge in poverty monitoring: imputing household consumption from survey features when detailed consumption modules are unavailable.

### Competition Background

**Competition Link:** [DrivenData - World Bank Poverty Prediction](https://www.drivendata.org/competitions/305/competition-worldbank-poverty/)

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

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train models
cd src
python train.py --n-estimators 1000 --seeds 42 123

# Generate submission
python submit.py --mode standard
```

## Documentation

- **[Full Documentation](docs/README.md)** - Comprehensive project documentation
- **[Research Paper](docs/paper.tex)** - LaTeX research paper (compile with pdflatex)
- **[Interactive Dashboard](docs/index.html)** - Open in browser for visualizations

## Project Structure

```
poverty-prediction/
├── data/                    # Survey data files
├── src/                     # Source code modules
│   ├── config.py           # Constants and hyperparameters
│   ├── data_loader.py      # Data loading utilities
│   ├── preprocessing.py    # Feature preprocessing
│   ├── features.py         # Feature engineering
│   ├── models.py           # Model wrappers (GBDT & quantile)
│   ├── neural_models.py    # PyTorch models (MDN)
│   ├── metrics.py          # Competition metric
│   ├── validation.py       # LOSO cross-validation
│   ├── ensemble.py         # Ensemble methods
│   ├── train.py            # Training pipeline
│   └── submit.py           # Submission generation
├── outputs/                 # Models and submissions
└── docs/                    # Documentation and analysis
```

## Key Findings

1. **CatBoost** consistently outperformed LightGBM and XGBoost on this task
2. **Utility expenditure per person** was the strongest predictor across models
3. **MDN** achieved best CV but worst test score - significant overfitting
4. **Distribution modeling** didn't improve over point prediction for this metric

## License

This project was developed for research purposes as part of a poverty prediction competition.
