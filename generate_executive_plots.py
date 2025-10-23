"""
Generate all plots for Executive Summary
Saves high-quality PNGs to executive_summary_assets/

Usage:
    python generate_executive_plots.py --section eda
    python generate_executive_plots.py --section survival
    python generate_executive_plots.py --section prediction
    python generate_executive_plots.py --all
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent plots from opening
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
sns.set_style('darkgrid')
plt.rcParams['figure.dpi'] = 300  # High quality for reports
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.ioff()  # Turn off interactive mode

# Output directory
OUTPUT_DIR = Path('executive_summary_assets')
OUTPUT_DIR.mkdir(exist_ok=True)

# Import custom utilities
sys.path.append('01_exploratory_data_analysis')
from eda_utils import (
    stacked_plot, 
    print_churn_summary,
    correlation_heatmap,
    plot_demographic_grid
)

print("="*80)
print("EXECUTIVE SUMMARY PLOT GENERATOR")
print("="*80)


def load_and_prep_data():
    """Load and preprocess data for EDA plots"""
    print("\n📂 Loading data...")
    df = pd.read_csv('data/Customer-Churn-Records.csv')
    
    # Drop identifiers
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(' ', '')
    
    # Drop weak features (based on EDA findings)
    weak_features = ['card_type', 'hascrcard', 'satisfaction_score', 
                     'point_earned', 'estimatedsalary', 'creditscore']
    existing_weak = [f for f in weak_features if f in df.columns]
    if existing_weak:
        df = df.drop(existing_weak, axis=1)
    
    # Convert binary columns
    binary_cols = ['isactivemember', 'exited', 'complain']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    
    # Create age_group
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 30, 40, 50, 60, 70, 100],
        labels=['18-30', '31-40', '41-50', '51-60', '61-70', '70+']
    )
    
    print(f"✓ Data loaded: {df.shape[0]:,} customers, {df.shape[1]} features")
    return df


def generate_eda_plots(df):
    """Generate all EDA plots for executive summary"""
    print("\n" + "="*80)
    print("SECTION 1: EXPLORATORY DATA ANALYSIS PLOTS")
    print("="*80)
    
    # Plot 1: Overall Churn Rate
    print("\n📊 Generating: 01_overall_churn_rate.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    churn_counts = df['exited'].value_counts()
    churn_pct = df['exited'].value_counts(normalize=True) * 100
    
    colors = ['#2ecc71', '#e74c3c']  # Green for retained, red for churned
    bars = ax.bar(['Retained', 'Churned'], churn_counts.values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars, churn_pct.values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
    ax.set_title('Overall Customer Churn Distribution', fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_overall_churn_rate.png')
    plt.close()
    print("✓ Saved: 01_overall_churn_rate.png")
    
    # Plot 2: Demographic Grid (Gender, Geography, Age Groups)
    print("\n📊 Generating: 02_demographic_overview.png")
    plot_demographic_grid(df)
    plt.savefig(OUTPUT_DIR / '02_demographic_overview.png')
    plt.close()
    print("✓ Saved: 02_demographic_overview.png")
    
    # Plot 3: Age Distribution with Churn
    print("\n📊 Generating: 03_age_distribution_churn.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # KDE plot
    for label, exited in [('Retained', False), ('Churned', True)]:
        axes[0].hist(df[df['exited'] == exited]['age'], bins=30, alpha=0.6, 
                     label=label, edgecolor='black')
    axes[0].set_xlabel('Age', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[0].set_title('Age Distribution by Churn Status', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Churn rate by age group
    age_churn = df.groupby('age_group')['exited'].agg(['sum', 'count'])
    age_churn['churn_rate'] = (age_churn['sum'] / age_churn['count'] * 100)
    
    bars = axes[1].bar(age_churn.index.astype(str), age_churn['churn_rate'], 
                       color='#e74c3c', alpha=0.8, edgecolor='black')
    axes[1].set_xlabel('Age Group', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Churn Rate (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Churn Rate by Age Group', fontsize=13, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_age_distribution_churn.png')
    plt.close()
    print("✓ Saved: 03_age_distribution_churn.png")
    
    # Plot 4: Number of Products Analysis
    print("\n📊 Generating: 04_products_churn_analysis.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Distribution
    product_counts = df['numofproducts'].value_counts().sort_index()
    axes[0].bar(product_counts.index, product_counts.values, 
                color='#3498db', alpha=0.8, edgecolor='black')
    axes[0].set_xlabel('Number of Products', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribution of Products per Customer', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(product_counts.values):
        axes[0].text(product_counts.index[i], v, f'{v:,}', 
                     ha='center', va='bottom', fontweight='bold')
    
    # Churn rate by products
    product_churn = df.groupby('numofproducts')['exited'].agg(['sum', 'count'])
    product_churn['churn_rate'] = (product_churn['sum'] / product_churn['count'] * 100)
    
    colors_products = ['#3498db', '#2ecc71', '#e74c3c', '#e74c3c']
    bars = axes[1].bar(product_churn.index, product_churn['churn_rate'], 
                       color=colors_products[:len(product_churn)], alpha=0.8, edgecolor='black')
    axes[1].set_xlabel('Number of Products', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Churn Rate (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Churn Rate by Number of Products (U-Shape)', fontsize=13, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_products_churn_analysis.png')
    plt.close()
    print("✓ Saved: 04_products_churn_analysis.png")
    
    # Plot 5: Active Member Status
    print("\n📊 Generating: 05_active_member_impact.png")
    stacked_plot(df, 'isactivemember', 'exited')
    plt.suptitle('Active Member Status Impact on Churn', fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(OUTPUT_DIR / '05_active_member_impact.png')
    plt.close()
    print("✓ Saved: 05_active_member_impact.png")
    
    # Plot 6: Geography Impact
    print("\n📊 Generating: 06_geography_churn.png")
    stacked_plot(df, 'geography', 'exited')
    plt.suptitle('Geographic Distribution and Churn Rates', fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(OUTPUT_DIR / '06_geography_churn.png')
    plt.close()
    print("✓ Saved: 06_geography_churn.png")
    
    # Plot 7: Complaint Analysis (Critical Finding)
    print("\n📊 Generating: 07_complaint_impact.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    complaint_churn = df.groupby('complain')['exited'].agg(['sum', 'count'])
    complaint_churn['churn_rate'] = (complaint_churn['sum'] / complaint_churn['count'] * 100)
    
    x = ['No Complaint', 'Complaint Filed']
    churn_rates = complaint_churn['churn_rate'].values
    colors = ['#2ecc71', '#e74c3c']
    
    bars = ax.bar(x, churn_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add dramatic value labels
    for i, (bar, rate) in enumerate(zip(bars, churn_rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', 
                fontsize=16, fontweight='bold')
    
    ax.set_ylabel('Churn Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Complaint Status: The Dominant Churn Predictor', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
    # Add annotation
    ax.annotate(f'{churn_rates[1]/churn_rates[0]:.0f}× Higher Churn Rate!',
                xy=(1, churn_rates[1]), xytext=(0.5, 80),
                fontsize=14, fontweight='bold', color='#000000',
                arrowprops=dict(arrowstyle='->', color='#000000', lw=2))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '07_complaint_impact.png')
    plt.close()
    print("✓ Saved: 07_complaint_impact.png")
    
    # Plot 8: Correlation Heatmap
    print("\n📊 Generating: 08_correlation_heatmap.png")
    
    # Prepare correlation data
    corr_df = df.copy()
    # Convert boolean to int for correlation
    for col in corr_df.select_dtypes(include='bool').columns:
        corr_df[col] = corr_df[col].astype(int)
    # Drop non-numeric and categorical
    corr_df = corr_df.select_dtypes(include=[np.number])
    
    # Create correlation heatmap focused on top correlations with churn
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get correlations with exited
    churn_corr = corr_df.corr()['exited'].abs().sort_values(ascending=False)
    # Select top features (excluding exited itself)
    top_features = churn_corr[1:13].index.tolist() + ['exited']
    
    # Create smaller correlation matrix
    corr_matrix = corr_df[top_features].corr()
    
    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, ax=ax)
    ax.set_title('Feature Correlation Heatmap (Top 12 + Churn)', 
                 fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '08_correlation_heatmap.png')
    plt.close()
    print("✓ Saved: 08_correlation_heatmap.png")
    
    print(f"\n✅ Generated {8} EDA plots in {OUTPUT_DIR}/")


def generate_survival_plots(df):
    """Generate survival analysis plots for executive summary"""
    print("\n" + "="*80)
    print("SECTION 2: SURVIVAL ANALYSIS PLOTS")
    print("="*80)
    
    # Import survival analysis libraries
    from lifelines import KaplanMeierFitter, CoxPHFitter
    sys.path.append('02_survival_analysis')
    from survival_utils import plot_survival_analysis_2groups, plot_survival_analysis_multigroup
    
    # Define event and time variables
    eventvar = df['exited'].astype(int)
    timevar = df['tenure']
    
    # Plot 1: Overall Kaplan-Meier Survival Curve
    print("\n📊 Generating: 09_overall_survival_curve.png")
    kmf = KaplanMeierFitter()
    kmf.fit(timevar, event_observed=eventvar, label="All Customers")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    kmf.plot(ax=ax, ci_show=True, color='#3498db', linewidth=2.5)
    ax.set_ylabel('Probability of Customer Retention', fontsize=12, fontweight='bold')
    ax.set_xlabel('Tenure (years)', fontsize=12, fontweight='bold')
    ax.set_title('Kaplan-Meier Survival Curve: Customer Retention Over Time', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add median survival time annotation
    median_survival = kmf.median_survival_time_
    if not np.isnan(median_survival):
        ax.axvline(x=median_survival, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(median_survival + 0.2, 0.5, f'Median: {median_survival:.1f} years', 
                rotation=0, verticalalignment='center', fontsize=11, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '09_overall_survival_curve.png')
    plt.close()
    print("✓ Saved: 09_overall_survival_curve.png")
    
    # Plot 2: Age Groups Survival Comparison (most critical demographic)
    print("\n📊 Generating: 10_survival_age_groups.png")
    age_18_30 = (df['age_group'] == '18-30')
    age_31_40 = (df['age_group'] == '31-40')
    age_41_50 = (df['age_group'] == '41-50')
    age_51_60 = (df['age_group'] == '51-60')
    age_61_70 = (df['age_group'] == '61-70')
    age_70_plus = (df['age_group'] == '70+')
    
    kmf_age = KaplanMeierFitter()
    result_age = plot_survival_analysis_multigroup(
        timevar, eventvar, df, 'age_group',
        [age_18_30, age_31_40, age_41_50, age_51_60, age_61_70, age_70_plus],
        ['18-30', '31-40', '41-50', '51-60', '61-70', '70+'],
        'Survival Analysis by Age Group',
        kmf_age,
        show_at_risk=True
    )
    plt.savefig(OUTPUT_DIR / '10_survival_age_groups.png')
    plt.close()
    print("✓ Saved: 10_survival_age_groups.png")
    
    # Plot 3: Number of Products Survival Comparison (critical finding from EDA)
    print("\n📊 Generating: 11_survival_products.png")
    prod_1 = (df['numofproducts'] == 1)
    prod_2 = (df['numofproducts'] == 2)
    prod_3 = (df['numofproducts'] == 3)
    prod_4 = (df['numofproducts'] == 4)
    
    kmf_products = KaplanMeierFitter()
    result_products = plot_survival_analysis_multigroup(
        timevar, eventvar, df, 'numofproducts',
        [prod_1, prod_2, prod_3, prod_4],
        ['1 Product', '2 Products', '3 Products', '4 Products'],
        'Survival Analysis by Number of Products',
        kmf_products,
        show_at_risk=True
    )
    plt.savefig(OUTPUT_DIR / '11_survival_products.png')
    plt.close()
    print("✓ Saved: 11_survival_products.png")
    
    # Plot 4: Active Member Status Comparison
    print("\n📊 Generating: 12_survival_active_status.png")
    active_mask = (df['isactivemember'] == 1)
    inactive_mask = (df['isactivemember'] == 0)
    
    kmf_active = KaplanMeierFitter()
    result_active = plot_survival_analysis_2groups(
        timevar, eventvar, active_mask, inactive_mask,
        "Active Member", "Inactive Member",
        'Survival Analysis by Activity Status',
        kmf_active,
        show_at_risk=True
    )
    plt.savefig(OUTPUT_DIR / '12_survival_active_status.png')
    plt.close()
    print("✓ Saved: 12_survival_active_status.png")
    
    # Plot 5: Geography Comparison
    print("\n📊 Generating: 13_survival_geography.png")
    france_mask = (df['geography'] == 'France')
    spain_mask = (df['geography'] == 'Spain')
    germany_mask = (df['geography'] == 'Germany')
    
    kmf_geo = KaplanMeierFitter()
    result_geo = plot_survival_analysis_multigroup(
        timevar, eventvar, df, 'geography',
        [france_mask, spain_mask, germany_mask],
        ['France', 'Spain', 'Germany'],
        'Survival Analysis by Geography',
        kmf_geo,
        show_at_risk=True
    )
    plt.savefig(OUTPUT_DIR / '13_survival_geography.png')
    plt.close()
    print("✓ Saved: 13_survival_geography.png")
    
    # Plot 6: Cox PH Model Coefficients
    print("\n📊 Generating: 14_cox_ph_coefficients.png")
    
    # Prepare data for Cox PH model
    cox_df = df.copy()
    
    # Drop age and complain (as done in original analysis)
    # age: multicollinearity with age_group
    # complain: too dominant (r=0.996 with churn)
    drop_cols = []
    if 'age' in cox_df.columns:
        drop_cols.append('age')
    if 'complain' in cox_df.columns:
        drop_cols.append('complain')
    
    if drop_cols:
        cox_df = cox_df.drop(drop_cols, axis=1)
    
    # Encode gender as numeric (Male=0, Female=1)
    if 'gender' in cox_df.columns:
        cox_df['gender'] = (cox_df['gender'] == 'Female').astype(int)
    
    # One-hot encode geography (drop first to avoid multicollinearity)
    cox_df = pd.get_dummies(cox_df, columns=['geography'], drop_first=True, dtype=int)
    
    # One-hot encode age_group (drop first to avoid multicollinearity)
    cox_df = pd.get_dummies(cox_df, columns=['age_group'], drop_first=True, dtype=int)
    
    # Drop balance_group and tenure_group (keep continuous versions)
    for col in ['balance_group', 'tenure_group']:
        if col in cox_df.columns:
            cox_df = cox_df.drop(col, axis=1)
    
    # Drop any remaining non-numeric columns (safety check)
    non_numeric_cols = cox_df.select_dtypes(include=['object', 'category']).columns.tolist()
    if non_numeric_cols:
        print(f"  ⚠️  Dropping non-numeric columns: {non_numeric_cols}")
        cox_df = cox_df.drop(non_numeric_cols, axis=1)
    
    # Convert boolean columns to int
    bool_cols = cox_df.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        cox_df[col] = cox_df[col].astype(int)
    
    # Fit Cox PH model
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col='tenure', event_col='exited')
    
    # Plot coefficients
    fig, ax = plt.subplots(figsize=(10, 8))
    cph.plot(ax=ax, hazard_ratios=True)
    ax.set_title('Cox Proportional Hazards Model - Hazard Ratios', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Hazard Ratio (log scale)', fontsize=12, fontweight='bold')
    ax.axvline(x=1, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '14_cox_ph_coefficients.png')
    plt.close()
    print("✓ Saved: 14_cox_ph_coefficients.png")
    
    print(f"\n✅ Generated {6} survival analysis plots in {OUTPUT_DIR}/")


def generate_prediction_plots():
    """Generate churn prediction model plots for executive summary"""
    import pickle
    import shap
    from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
    from sklearn.inspection import PartialDependenceDisplay
    
    print("\n" + "="*80)
    print("SECTION 3: CHURN PREDICTION PLOTS")
    print("="*80)
    
    # Load trained model and data
    print("\n📂 Loading model and data...")
    model_path = Path('03_churn_prediction/churn_model.pkl')
    features_path = Path('03_churn_prediction/feature_names.pkl')
    perm_imp_path = Path('03_churn_prediction/checkpoints/permutation_importance.pkl')
    shap_path = Path('03_churn_prediction/checkpoints/shap_data.pkl')
    
    if not model_path.exists():
        print("⚠️  Model file not found. Please train the model first.")
        return
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(features_path, 'rb') as f:
        feature_cols = pickle.load(f)
    
    print("✓ Model and features loaded")
    
    # Prepare test data
    df = pd.read_csv('data/Customer-Churn-Records.csv')
    
    # Drop identifiers
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
    
    # Drop weak features
    weak_features = ['card_type', 'hascrcard', 'satisfaction_score', 
                     'point_earned', 'estimatedsalary', 'creditscore', 'complain']
    existing_weak = [f for f in weak_features if f in df.columns]
    if existing_weak:
        df = df.drop(existing_weak, axis=1)
    
    # Convert and encode
    binary_cols = ['isactivemember', 'exited']
    for col in binary_cols:
        df[col] = df[col].astype(int)
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    
    # Create age_group
    df['age_group'] = pd.cut(df['age'], 
                              bins=[0, 30, 40, 50, 60, 70, 100],
                              labels=['18-30', '31-40', '41-50', '51-60', '61-70', '70+'])
    
    # One-hot encode
    df_encoded = pd.get_dummies(df, columns=['geography', 'age_group'], drop_first=True, dtype=int)
    
    # Split data (same as training)
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df_encoded, test_size=0.2, random_state=42, stratify=df_encoded['exited'])
    
    X_test = test[feature_cols]
    y_test = test['exited']
    
    print(f"✓ Test data prepared: {X_test.shape}")
    
    # Plot 1: Feature Importance Comparison
    print("\n📊 Generating: 15_feature_importance_comparison.png")
    
    # Get built-in importance
    builtin_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Load permutation importance
    if perm_imp_path.exists():
        with open(perm_imp_path, 'rb') as f:
            perm_imp = pickle.load(f)
    else:
        print("  ⚠️  Permutation importance not found, using built-in only")
        perm_imp = builtin_imp.copy()
        perm_imp.columns = ['feature', 'importance_mean']
    
    # Plot top 10
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Built-in importance
    top_builtin = builtin_imp.head(10)
    ax1.barh(range(len(top_builtin)), top_builtin['importance'], color='steelblue')
    ax1.set_yticks(range(len(top_builtin)))
    ax1.set_yticklabels(top_builtin['feature'])
    ax1.invert_yaxis()
    ax1.set_xlabel('Importance', fontweight='bold', fontsize=11)
    ax1.set_title('Built-in Feature Importance', fontweight='bold', fontsize=13)
    ax1.grid(axis='x', alpha=0.3)
    
    # Permutation importance
    top_perm = perm_imp.head(10)
    ax2.barh(range(len(top_perm)), top_perm['importance_mean'], color='coral')
    ax2.set_yticks(range(len(top_perm)))
    ax2.set_yticklabels(top_perm['feature'])
    ax2.invert_yaxis()
    ax2.set_xlabel('Permutation Importance', fontweight='bold', fontsize=11)
    ax2.set_title('Permutation Feature Importance', fontweight='bold', fontsize=13)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Feature Importance: Built-in vs Permutation', 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '15_feature_importance_comparison.png')
    plt.close()
    print("✓ Saved: 15_feature_importance_comparison.png")
    
    # Plot 2: Confusion Matrix
    print("\n📊 Generating: 16_confusion_matrix.png")
    
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Labels
    classes = ['Retained', 'Churned']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True Label', xlabel='Predicted Label')
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center", fontsize=16,
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set_title('Confusion Matrix - Final Model', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '16_confusion_matrix.png')
    plt.close()
    print("✓ Saved: 16_confusion_matrix.png")
    
    # Plot 3: ROC Curve
    print("\n📊 Generating: 17_roc_curve.png")
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
    ax.set_title('ROC Curve - Churn Prediction Model', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '17_roc_curve.png')
    plt.close()
    print("✓ Saved: 17_roc_curve.png")
    
    # Plot 4: Partial Dependence - Age
    print("\n📊 Generating: 18_pdp_age.png")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    PartialDependenceDisplay.from_estimator(
        model, X_test, ['age'],
        kind='average',
        ax=ax,
        grid_resolution=50
    )
    ax.set_title('Partial Dependence Plot: Age', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Partial Dependence (Churn Probability)', fontweight='bold', fontsize=11)
    ax.set_xlabel('Age (years)', fontweight='bold', fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '18_pdp_age.png')
    plt.close()
    print("✓ Saved: 18_pdp_age.png")
    
    # Plot 5: Partial Dependence - NumOfProducts
    print("\n📊 Generating: 19_pdp_numofproducts.png")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    PartialDependenceDisplay.from_estimator(
        model, X_test, ['numofproducts'],
        kind='average',
        ax=ax,
        grid_resolution=4
    )
    ax.set_title('Partial Dependence Plot: Number of Products (U-Shape)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Partial Dependence (Churn Probability)', fontweight='bold', fontsize=11)
    ax.set_xlabel('Number of Products', fontweight='bold', fontsize=11)
    ax.set_xticks([1, 2, 3, 4])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '19_pdp_numofproducts.png')
    plt.close()
    print("✓ Saved: 19_pdp_numofproducts.png")
    
    # Plot 6: SHAP Summary
    if shap_path.exists():
        print("\n📊 Generating: 20_shap_summary.png")
        
        with open(shap_path, 'rb') as f:
            shap_data = pickle.load(f)
        
        shap_values_churn = shap_data['shap_values_churn']
        X_test_sample = shap_data['X_test_sample']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values_churn, X_test_sample, 
                         feature_names=feature_cols, show=False)
        plt.title('SHAP Summary: Feature Impact on Churn Prediction', 
                 fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / '20_shap_summary.png')
        plt.close()
        print("✓ Saved: 20_shap_summary.png")
    else:
        print("  ⚠️  SHAP data not found, skipping SHAP plot")
    
    print(f"\n✅ Generated 6 churn prediction plots in {OUTPUT_DIR}/")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate executive summary plots')
    parser.add_argument('--section', choices=['eda', 'survival', 'prediction'], 
                        help='Generate plots for specific section')
    parser.add_argument('--all', action='store_true', 
                        help='Generate all plots')
    
    args = parser.parse_args()
    
    # Load data
    df = load_and_prep_data()
    
    # Generate plots based on arguments
    if args.all or args.section == 'eda':
        generate_eda_plots(df)
    
    if args.all or args.section == 'survival':
        generate_survival_plots(df)
    
    if args.all or args.section == 'prediction':
        generate_prediction_plots()
    
    print("\n" + "="*80)
    print("✅ PLOT GENERATION COMPLETE")
    print("="*80)
    print(f"📁 Output directory: {OUTPUT_DIR.absolute()}")
    print("📊 Generated plots ready for executive summary")


if __name__ == '__main__':
    main()

