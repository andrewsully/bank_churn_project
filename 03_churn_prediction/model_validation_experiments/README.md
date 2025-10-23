# Model Validation Experiments

This folder contains complementary modeling experiments designed to validate the Random Forest baseline from `03_churn_prediction/`.

## Purpose

After building our production model, we conducted three validation experiments to ensure our choices were optimal:
- **Model Comparison**: Test if XGBoost or LightGBM outperform Random Forest
- **SMOTE Experiment**: Test SMOTE oversampling as alternative to class_weight
- **Feature Engineering**: Test engineered features (interactions, polynomials, bins)

## Validation Results

All experiments validated our original model configuration as optimal:

### Experiment 1: Model Comparison
**Question:** Can XGBoost or LightGBM beat Random Forest?

**Results:**
- 🥇 Random Forest: 62.52% F1-Score, 68.01% Precision, 57.84% Recall
- 🥈 LightGBM: 60.97% F1-Score, 61.65% Precision, 60.29% Recall
- 🥉 XGBoost: 60.47% F1-Score, 61.15% Precision, 59.80% Recall

**Conclusion:** Random Forest outperforms both alternatives by 1.55-2.05% F1-Score.

### Experiment 2: SMOTE vs Class Weight
**Question:** Does SMOTE oversampling improve recall?

**Results:**
- Class Weight: 62.52% F1-Score, 68.01% Precision, 57.84% Recall
- SMOTE: 60.57% F1-Score, 56.18% Precision, 65.69% Recall

**Trade-off:** SMOTE catches +32 churners but creates +64 false alarms (2:1 ratio unfavorable).

**Conclusion:** Class weight approach more efficient and better balanced.

### Experiment 3: Feature Engineering
**Question:** Do engineered features improve performance?

**Results:**
- Baseline Features: 62.52% F1-Score (13 features)
- Engineered Features: 62.09% F1-Score (27 features, +14 added)

**Conclusion:** Feature engineering hurts performance despite doubling feature count.

## Final Validated Configuration

Our baseline model from `03_churn_prediction` is **VALIDATED AS OPTIMAL**:

- **Algorithm:** Random Forest Classifier ✓
- **Class Handling:** class_weight={0:1, 1:2} ✓
- **Features:** Original 13 features ✓
- **Performance:** 62.52% F1-Score, 57.84% Recall, 68.01% Precision

**No changes needed** - production model ready for deployment.

## Structure

```
model_validation_experiments/
├── advanced_modeling_utils.py    # Shared utilities for consistent preprocessing
├── 01_model_comparison.py        # RF vs XGBoost vs LightGBM comparison
├── 02_smote_experiment.py        # SMOTE vs class_weight experiment
├── 03_feature_engineering.py    # Feature engineering impact analysis
└── results/                       # Saved experiment results
```

## Key Features

- **Consistent Preprocessing**: All experiments use identical preprocessing from `03_churn_prediction`
- **Same Train/Test Split**: `random_state=42` ensures fair comparison
- **Independent Experiments**: Each experiment can run standalone
- **Results Caching**: Pickle files save expensive computations

## Insights

**Why Random Forest?**
- Handles non-linear patterns (U-shape in products) without feature engineering
- After careful hyperparameter tuning, matches/exceeds gradient boosting performance
- Provides interpretable feature importance for stakeholders

**Why Class Weight?**
- Simpler and more transparent than SMOTE
- Better precision-recall balance for moderate imbalance (20% churn rate)
- SMOTE works best when minority class <5%, our rate is 20%

**Why Baseline Features?**
- Random Forest captures interactions automatically through tree splits
- Adding features dilutes signal with noise
- Simpler model (13 features) easier to interpret and maintain

## Usage

```python
# Load consistent preprocessed data
from advanced_modeling_utils import load_preprocessed_data
X_train, X_test, y_train, y_test, feature_names = load_preprocessed_data()

# Run individual experiments
# See each .py file for detailed execution instructions
```

## Status

- ✅ All experiments complete
- ✅ Model configuration validated
- ✅ Production-ready deployment confirmed

