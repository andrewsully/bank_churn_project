# %% [markdown]
# # Advanced Modeling: Feature Engineering Experiment
# ## Testing Engineered Features for Performance Improvement
#
# This notebook tests whether engineered features can improve model performance 
# beyond our current feature set.
#
# **Objective:** Determine if feature engineering (interactions, polynomials, 
# binning) can improve F1-Score beyond our baseline 62.52%.
#
# **Features to Engineer:**
# 1. **Interaction Terms:** Age × Balance, Geography × Products, Age × Activity
# 2. **Polynomial Features:** Age², Balance²
# 3. **Binning:** CreditScore bins, Balance bins, Age lifecycle groups
# 4. **Ratios:** Balance/EstimatedSalary (if we had it), Engagement velocity
#
# **Key Question:** Do engineered features capture synergies that improve 
# predictions, or do they just add noise?
#
# ---
#
# ## Key Findings & Insights
#
# **Results Summary:**
# - **Baseline Features**: 62.52% F1-Score (13 features)
# - **Engineered Features**: 62.09% F1-Score (27 features, +14 added)
#
# **Validation Conclusion:**
# Feature engineering slightly hurts performance (-0.43% F1-Score) despite doubling 
# the feature count (+108% complexity). All primary metrics decrease: accuracy 
# (-0.20%), precision (-0.67%), recall (-0.24%). Only ROC-AUC improves marginally 
# (+0.31%), insufficient to justify added complexity.
#
# **Technical Insights:**
# Random Forest handles interactions and non-linearities automatically through tree 
# splits. Explicit interaction terms are redundant - trees naturally create interactions 
# like "if Age > 50 AND Products > 2". Binning continuous features loses information 
# compared to letting trees split on raw values. Adding features dilutes signal with 
# noise and increases overfitting risk without performance benefit.
#
# **Lesson Learned:**
# More features ≠ better performance. Quality > quantity. Random Forest is powerful 
# enough to capture complexity without explicit engineering. Our baseline features 
# already capture core patterns (age groups, geography, products) - the essential 
# predictors for churn.
#
# **Recommendation:**
# Stick with baseline features. Simpler model (13 vs 27 features) is easier to 
# interpret, faster to train, and performs better. No feature engineering needed 
# for tree-based models.

# %% [markdown]
# ## 1. Import Libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
from pathlib import Path

# Sklearn models and utilities
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, auc, confusion_matrix, classification_report
)

# Custom utilities
from advanced_modeling_utils import (
    load_preprocessed_data,
    compare_models_results,
    save_experiment_results
)

# Settings
sns.set_style('darkgrid')
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

print("✓ Libraries loaded successfully")

# %% [markdown]
# ## 2. Load Preprocessed Data
#
# Load data and create copies for feature engineering experiments.

# %%
# Load baseline data
X_train_base, X_test_base, y_train, y_test, feature_names = load_preprocessed_data()

print(f"✓ Baseline data loaded")
print(f"  Train shape: {X_train_base.shape}")
print(f"  Test shape: {X_test_base.shape}")
print(f"  Baseline features: {len(feature_names)}")

# Create copies for engineering
X_train = X_train_base.copy()
X_test = X_test_base.copy()

# %% [markdown]
# ## 3. Baseline Model (No Feature Engineering)
#
# Train Random Forest on original features to establish baseline.

# %%
print("="*80)
print("BASELINE: NO FEATURE ENGINEERING")
print("="*80)

rf_baseline = RandomForestClassifier(
    n_estimators=900,
    max_depth=11,
    criterion='gini',
    max_features=None,
    min_samples_split=4,
    min_samples_leaf=1,
    class_weight={0:1, 1:2},
    random_state=42,
    n_jobs=-1
)

print("\n🔨 Training baseline model...")
rf_baseline.fit(X_train, y_train)
print("✓ Training complete")

# Evaluate
baseline_pred = rf_baseline.predict(X_test)
baseline_proba = rf_baseline.predict_proba(X_test)[:, 1]

baseline_accuracy = accuracy_score(y_test, baseline_pred)
baseline_precision = precision_score(y_test, baseline_pred)
baseline_recall = recall_score(y_test, baseline_pred)
baseline_f1 = f1_score(y_test, baseline_pred)
baseline_roc = roc_auc_score(y_test, baseline_proba)

print(f"\n📊 Baseline Performance:")
print(f"  Accuracy:  {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
print(f"  Precision: {baseline_precision:.4f} ({baseline_precision*100:.2f}%)")
print(f"  Recall:    {baseline_recall:.4f} ({baseline_recall*100:.2f}%)")
print(f"  F1-Score:  {baseline_f1:.4f} ({baseline_f1*100:.2f}%)")
print(f"  ROC-AUC:   {baseline_roc:.4f} ({baseline_roc*100:.2f}%)")

# %% [markdown]
# ## 4. Feature Engineering
#
# Create engineered features based on domain knowledge and EDA insights.

# %% [markdown]
# ### 4.1 Interaction Terms
#
# Create interactions between features that showed relationships in EDA.

# %%
print("\n" + "="*80)
print("CREATING INTERACTION TERMS")
print("="*80)

# Age × Balance (high-balance older customers = retirement planning)
X_train['age_x_balance'] = X_train['age'] * X_train['balance']
X_test['age_x_balance'] = X_test['age'] * X_test['balance']
print("✓ Created: age_x_balance")

# Age × IsActiveMember (age-driven engagement patterns)
X_train['age_x_active'] = X_train['age'] * X_train['isactivemember']
X_test['age_x_active'] = X_test['age'] * X_test['isactivemember']
print("✓ Created: age_x_active")

# NumOfProducts × Geography (Germany + 3+ products = catastrophic)
# Check if geography columns exist
geography_cols = [col for col in X_train.columns if 'geography' in col.lower()]
if len(geography_cols) > 0:
    for geo_col in geography_cols:
        X_train[f'products_x_{geo_col}'] = X_train['numofproducts'] * X_train[geo_col]
        X_test[f'products_x_{geo_col}'] = X_test['numofproducts'] * X_test[geo_col]
    print(f"✓ Created: products_x_geography interaction terms")

# Tenure × IsActiveMember (engagement duration)
X_train['tenure_x_active'] = X_train['tenure'] * X_train['isactivemember']
X_test['tenure_x_active'] = X_test['tenure'] * X_test['isactivemember']
print("✓ Created: tenure_x_active")

print(f"\nTotal features after interactions: {X_train.shape[1]}")

# %% [markdown]
# ### 4.2 Polynomial Features
#
# Create polynomial terms to capture non-linear relationships.

# %%
print("\n" + "="*80)
print("CREATING POLYNOMIAL FEATURES")
print("="*80)

# Age² (capture lifecycle non-linearity more explicitly)
X_train['age_squared'] = X_train['age'] ** 2
X_test['age_squared'] = X_test['age'] ** 2
print("✓ Created: age_squared")

# Balance² (high-balance effect on retention)
X_train['balance_squared'] = X_train['balance'] ** 2
X_test['balance_squared'] = X_test['balance'] ** 2
print("✓ Created: balance_squared")

# NumOfProducts² (capture the U-shape effect)
X_train['products_squared'] = X_train['numofproducts'] ** 2
X_test['products_squared'] = X_test['numofproducts'] ** 2
print("✓ Created: products_squared")

print(f"\nTotal features after polynomials: {X_train.shape[1]}")

# %% [markdown]
# ### 4.3 Binning Features
#
# Create binned versions of continuous features.

# %%
print("\n" + "="*80)
print("CREATING BINNED FEATURES")
print("="*80)

# Age lifecycle bins (already have age_group, but creating more granular bins)
X_train['age_lifecycle'] = pd.cut(X_train['age'], 
                                   bins=[0, 35, 50, 65, 100],
                                   labels=['Young', 'Mid-Career', 'Pre-Retirement', 'Retirement'])
X_test['age_lifecycle'] = pd.cut(X_test['age'],
                                  bins=[0, 35, 50, 65, 100],
                                  labels=['Young', 'Mid-Career', 'Pre-Retirement', 'Retirement'])

# One-hot encode age_lifecycle
X_train = pd.get_dummies(X_train, columns=['age_lifecycle'], drop_first=True, dtype=int)
X_test = pd.get_dummies(X_test, columns=['age_lifecycle'], drop_first=True, dtype=int)
print("✓ Created: age_lifecycle bins")

# Balance bins (zero, low, medium, high)
# Note: Need to reload original balance since we created balance_squared
balance_train = X_train_base['balance'].copy()
balance_test = X_test_base['balance'].copy()

X_train['balance_category'] = pd.cut(balance_train,
                                     bins=[-np.inf, 0, 50000, 150000, np.inf],
                                     labels=['Zero', 'Low', 'Medium', 'High'])
X_test['balance_category'] = pd.cut(balance_test,
                                    bins=[-np.inf, 0, 50000, 150000, np.inf],
                                    labels=['Zero', 'Low', 'Medium', 'High'])

# One-hot encode balance_category
X_train = pd.get_dummies(X_train, columns=['balance_category'], drop_first=True, dtype=int)
X_test = pd.get_dummies(X_test, columns=['balance_category'], drop_first=True, dtype=int)
print("✓ Created: balance_category bins")

print(f"\nTotal features after binning: {X_train.shape[1]}")

# %% [markdown]
# ### 4.4 Summary of Engineered Features
#
# Review all engineered features created.

# %%
print("\n" + "="*80)
print("FEATURE ENGINEERING SUMMARY")
print("="*80)

baseline_feature_count = X_train_base.shape[1]
engineered_feature_count = X_train.shape[1]
new_features = engineered_feature_count - baseline_feature_count

print(f"\nBaseline features: {baseline_feature_count}")
print(f"Engineered features: {engineered_feature_count}")
print(f"New features added: {new_features}")
print(f"\nEngineered feature names:")
new_cols = [col for col in X_train.columns if col not in X_train_base.columns]
for i, col in enumerate(new_cols, 1):
    print(f"  {i}. {col}")

# %% [markdown]
# ## 5. Train Model with Engineered Features
#
# Train Random Forest on expanded feature set.

# %%
print("\n" + "="*80)
print("TRAINING MODEL WITH ENGINEERED FEATURES")
print("="*80)

rf_engineered = RandomForestClassifier(
    n_estimators=900,
    max_depth=11,
    criterion='gini',
    max_features=None,
    min_samples_split=4,
    min_samples_leaf=1,
    class_weight={0:1, 1:2},
    random_state=42,
    n_jobs=-1
)

print("\n🔨 Training model with engineered features...")
rf_engineered.fit(X_train, y_train)
print("✓ Training complete")

# Evaluate
engineered_pred = rf_engineered.predict(X_test)
engineered_proba = rf_engineered.predict_proba(X_test)[:, 1]

engineered_accuracy = accuracy_score(y_test, engineered_pred)
engineered_precision = precision_score(y_test, engineered_pred)
engineered_recall = recall_score(y_test, engineered_pred)
engineered_f1 = f1_score(y_test, engineered_pred)
engineered_roc = roc_auc_score(y_test, engineered_proba)

print(f"\n📊 Engineered Features Performance:")
print(f"  Accuracy:  {engineered_accuracy:.4f} ({engineered_accuracy*100:.2f}%)")
print(f"  Precision: {engineered_precision:.4f} ({engineered_precision*100:.2f}%)")
print(f"  Recall:    {engineered_recall:.4f} ({engineered_recall*100:.2f}%)")
print(f"  F1-Score:  {engineered_f1:.4f} ({engineered_f1*100:.2f}%)")
print(f"  ROC-AUC:   {engineered_roc:.4f} ({engineered_roc*100:.2f}%)")

# %% [markdown]
# ## 6. Performance Comparison
#
# Compare baseline vs engineered features.

# %%
print("\n" + "="*80)
print("BASELINE vs ENGINEERED FEATURES COMPARISON")
print("="*80)

comparison_results = {
    'Baseline': (baseline_pred, baseline_proba),
    'Engineered Features': (engineered_pred, engineered_proba)
}

comparison_df = compare_models_results(comparison_results, y_test)
print("\n")
print(comparison_df.to_string(index=False))

# %% [markdown]
# ## 7. Feature Importance Analysis
#
# Identify which engineered features contribute most.

# %%
# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_engineered.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n" + "="*80)
print("TOP 15 MOST IMPORTANT FEATURES")
print("="*80)
print(feature_importance.head(15).to_string(index=False))

# Highlight engineered features
print("\n" + "="*80)
print("ENGINEERED FEATURES IMPORTANCE")
print("="*80)
engineered_features = [col for col in X_train.columns if col not in X_train_base.columns]
engineered_importance = feature_importance[feature_importance['Feature'].isin(engineered_features)]
if len(engineered_importance) > 0:
    print(engineered_importance.to_string(index=False))
else:
    print("No engineered features in top features")

# %% [markdown]
# ## 8. Visualizations
#
# Create visual comparisons.

# %% [markdown]
# ### 8.1 Confusion Matrix Comparison

# %%
# Side-by-side confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

for idx, (name, pred) in enumerate([('Baseline', baseline_pred), ('Engineered', engineered_pred)]):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
               xticklabels=['Not Churn', 'Churn'],
               yticklabels=['Not Churn', 'Churn'])
    axes[idx].set_title(f'{name}')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 8.2 ROC Curve Comparison

# %%
fig, ax = plt.subplots(figsize=(8, 6))

for name, proba in [('Baseline', baseline_proba), ('Engineered Features', engineered_proba)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves: Baseline vs Engineered Features')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 8.3 Metrics Bar Chart

# %%
# Create bar chart comparing metrics
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

metrics_data = pd.DataFrame({
    'Approach': ['Baseline', 'Engineered'],
    'Accuracy': [baseline_accuracy, engineered_accuracy],
    'Precision': [baseline_precision, engineered_precision],
    'Recall': [baseline_recall, engineered_recall],
    'F1-Score': [baseline_f1, engineered_f1],
    'ROC-AUC': [baseline_roc, engineered_roc]
})

metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
colors = ['steelblue', 'seagreen']

for idx, metric in enumerate(metric_cols):
    ax = axes[idx // 3, idx % 3]
    bars = ax.bar(metrics_data['Approach'], metrics_data[metric], color=colors)
    ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric)
    ax.set_ylim([0, 1])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Detailed Metrics Comparison

# %%
print("\n" + "="*80)
print("DETAILED METRICS COMPARISON")
print("="*80)

metrics_comparison = pd.DataFrame({
    'Approach': ['Baseline', 'Engineered Features', 'Difference'],
    'Accuracy': [baseline_accuracy, engineered_accuracy, engineered_accuracy - baseline_accuracy],
    'Precision': [baseline_precision, engineered_precision, engineered_precision - baseline_precision],
    'Recall': [baseline_recall, engineered_recall, engineered_recall - baseline_recall],
    'F1-Score': [baseline_f1, engineered_f1, engineered_f1 - baseline_f1],
    'ROC-AUC': [baseline_roc, engineered_roc, engineered_roc - baseline_roc]
})

print("\n")
print(metrics_comparison.to_string(index=False))

# %% [markdown]
# ## 10. Save Results
#
# Save experiment results for future reference.

# %%
# Prepare results dictionary
results = {
    'models': {
        'Baseline': rf_baseline,
        'Engineered': rf_engineered
    },
    'predictions': {
        'Baseline': baseline_pred,
        'Engineered': engineered_pred
    },
    'probabilities': {
        'Baseline': baseline_proba,
        'Engineered': engineered_proba
    },
    'metrics': comparison_df.to_dict('records'),
    'feature_importance': feature_importance.to_dict('records'),
    'engineered_features': engineered_features,
    'feature_count': {
        'baseline': X_train_base.shape[1],
        'engineered': X_train.shape[1],
        'new_features': len(engineered_features)
    },
    'y_test': y_test.values  # Save actual y_test for future plotting
}

# Save results
save_experiment_results(results, 'feature_engineering_results.pkl')

# %% [markdown]
# ## 11. Conclusions & Recommendations
#
# Based on the comparison results:

# %%
print("\n" + "="*80)
print("CONCLUSIONS & RECOMMENDATIONS")
print("="*80)

# Calculate improvements
accuracy_change = engineered_accuracy - baseline_accuracy
precision_change = engineered_precision - baseline_precision
recall_change = engineered_recall - baseline_recall
f1_change = engineered_f1 - baseline_f1
roc_change = engineered_roc - baseline_roc

print(f"\n📊 Performance Changes:")
print(f"  Accuracy change:  {accuracy_change:+.4f} ({accuracy_change*100:+.2f}%)")
print(f"  Precision change: {precision_change:+.4f} ({precision_change*100:+.2f}%)")
print(f"  Recall change:    {recall_change:+.4f} ({recall_change*100:+.2f}%)")
print(f"  F1-Score change:  {f1_change:+.4f} ({f1_change*100:+.2f}%)")
print(f"  ROC-AUC change:   {roc_change:+.4f} ({roc_change*100:+.2f}%)")

print(f"\n{'='*80}")
print("RECOMMENDATION:")
print(f"{'='*80}")

if f1_change > 0.01:  # More than 1% improvement
    print("\n✓ Feature engineering significantly improves performance.")
    print(f"  Improvement: +{f1_change*100:.2f}% F1-Score")
    print("\n  Consider adopting engineered features:")
    print("    - Performance gain justifies added complexity")
    print("    - Model can handle additional features")
    print("    - Top engineered features contribute meaningfully")
elif f1_change > 0 and f1_change <= 0.01:
    print("\n⚠ Feature engineering provides marginal improvement.")
    print(f"  Improvement: +{f1_change*100:.2f}% F1-Score")
    print("\n  Consider trade-offs:")
    print("    - Marginal benefit vs added complexity")
    print("    - Evaluate if specific engineered features are valuable")
    print("    - May be worth adopting only top contributing features")
else:
    print("\n✗ Feature engineering does not improve performance.")
    print(f"  F1-Score change: {f1_change*100:.2f}%")
    print("\n  Recommendation: Stick with baseline features")
    print("    - Baseline features are already optimized")
    print("    - Additional features may introduce noise")
    print("    - Random Forest captures non-linearities well")
    print("    - Simpler model is easier to interpret and maintain")

print(f"\n{'='*80}")
print("END OF FEATURE ENGINEERING EXPERIMENT")
print(f"{'='*80}")

