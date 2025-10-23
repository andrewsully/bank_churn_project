"""
Generate model comparison plots for LaTeX report with plain matplotlib styling.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Configure plotting - Reset to plain matplotlib style
sns.reset_orig()
plt.style.use('default')

# Set rcParams for LaTeX report plots
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.ioff()

# Output directory
OUTPUT_DIR = Path('img')
OUTPUT_DIR.mkdir(exist_ok=True)

def generate_model_comparison_roc():
    """Generate ROC curve comparison plot for Random Forest vs XGBoost vs LightGBM"""
    print("\n📊 Generating: 21_model_comparison_roc.png")
    
    # Load models and results
    results_dir = Path('../03_churn_prediction/model_validation_experiments/results')
    checkpoints_dir = Path('../03_churn_prediction/checkpoints')
    
    try:
        # Load results from comparison experiment
        rf_path = checkpoints_dir / 'grid1_results.pkl'
        xgb_path = checkpoints_dir / 'grid2_results.pkl'
        lgbm_path = checkpoints_dir / 'grid3_results.pkl'
        
        if not all([rf_path.exists(), xgb_path.exists(), lgbm_path.exists()]):
            print("⚠️  Model comparison results not found, skipping...")
            return
        
        # Load test data predictions
        with open(rf_path, 'rb') as f:
            rf_results = pickle.load(f)
        with open(xgb_path, 'rb') as f:
            xgb_results = pickle.load(f)
        with open(lgbm_path, 'rb') as f:
            lgbm_results = pickle.load(f)
        
        # Get test data
        df = pd.read_csv('../data/Customer-Churn-Records.csv')
        df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
        df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
        
        # Prepare features (same as training)
        weak_features = ['card_type', 'hascrcard', 'satisfaction_score', 
                         'point_earned', 'estimatedsalary', 'creditscore', 'complain']
        existing_weak = [f for f in weak_features if f in df.columns]
        if existing_weak:
            df = df.drop(existing_weak, axis=1)
        
        df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
        df['age_group'] = pd.cut(df['age'], 
                                  bins=[0, 30, 40, 50, 60, 70, 100],
                                  labels=['18-30', '31-40', '41-50', '51-60', '61-70', '70+'])
        df_encoded = pd.get_dummies(df, columns=['geography', 'age_group'], drop_first=True, dtype=int)
        
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(df_encoded, test_size=0.2, random_state=42, stratify=df_encoded['exited'])
        
        X_test = test.drop('exited', axis=1)
        y_test = test['exited']
        
        # Get predictions from each model
        # Handle both dictionary and GridSearchCV objects
        models = {}
        for name, result in [('Random Forest', rf_results), ('XGBoost', xgb_results), ('LightGBM', lgbm_results)]:
            if hasattr(result, 'best_estimator_'):
                models[name] = result.best_estimator_
            elif isinstance(result, dict) and 'model' in result:
                models[name] = result['model']
            else:
                models[name] = result
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for name, model in models.items():
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = model.decision_function(X_test)
            
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
        ax.set_title('ROC Curves: RF vs XGBoost vs LightGBM', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='lower right', fontsize=11)
        
        # Ensure all spines are visible and black
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / '21_model_comparison_roc.png')
        plt.close()
        print("✓ Saved: 21_model_comparison_roc.png")
        
    except Exception as e:
        print(f"⚠️  Error generating model comparison ROC: {e}")


def generate_smote_class_distribution():
    """Generate SMOTE class distribution plot (before and after)"""
    print("\n📊 Generating: 22_smote_class_distribution.png")
    
    try:
        # Get data
        df = pd.read_csv('../data/Customer-Churn-Records.csv')
        df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
        df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
        
        # Prepare features
        weak_features = ['card_type', 'hascrcard', 'satisfaction_score', 
                         'point_earned', 'estimatedsalary', 'creditscore', 'complain']
        existing_weak = [f for f in weak_features if f in df.columns]
        if existing_weak:
            df = df.drop(existing_weak, axis=1)
        
        df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
        df['age_group'] = pd.cut(df['age'], 
                                   bins=[0, 30, 40, 50, 60, 70, 100],
                                   labels=['18-30', '31-40', '41-50', '51-60', '61-70', '70+'])
        df_encoded = pd.get_dummies(df, columns=['geography', 'age_group'], drop_first=True, dtype=int)
        
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(df_encoded, test_size=0.2, random_state=42, stratify=df_encoded['exited'])
        
        X_train = train.drop('exited', axis=1)
        y_train = train['exited']
        
        # Apply SMOTE to get balanced distribution
        from imblearn.over_sampling import SMOTE
        from collections import Counter
        
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Color scheme for churn classes
        colors = ['#55A868', '#C44E52']  # Green for non-churned, red for churned
        
        # Before SMOTE
        before_counts = Counter(y_train)
        bars_before = axes[0].bar(before_counts.keys(), before_counts.values(), color=[colors[idx] for idx in before_counts.keys()], edgecolor='black')
        axes[0].set_title('Class Distribution Before SMOTE', fontweight='bold', fontsize=12)
        axes[0].set_xlabel('Churn Status', fontweight='bold', fontsize=11)
        axes[0].set_ylabel('Count', fontweight='bold', fontsize=11)
        axes[0].set_xticks([0, 1])
        axes[0].set_xticklabels(['Not Churn', 'Churn'])
        
        # After SMOTE
        after_counts = Counter(y_train_smote)
        bars_after = axes[1].bar(after_counts.keys(), after_counts.values(), color=[colors[idx] for idx in after_counts.keys()], edgecolor='black')
        axes[1].set_title('Class Distribution After SMOTE', fontweight='bold', fontsize=12)
        axes[1].set_xlabel('Churn Status', fontweight='bold', fontsize=11)
        axes[1].set_ylabel('Count', fontweight='bold', fontsize=11)
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(['Not Churn', 'Churn'])
        
        # Ensure all spines are visible and black for both subplots
        for ax in axes:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / '22_smote_class_distribution.png')
        plt.close()
        print("✓ Saved: 22_smote_class_distribution.png")
        
    except Exception as e:
        print(f"⚠️  Error generating SMOTE class distribution: {e}")


def generate_smote_roc_comparison():
    """Generate SMOTE vs class weight ROC comparison plot"""
    print("\n📊 Generating: 25_smote_roc_comparison.png")
    
    results_dir = Path('../03_churn_prediction/model_validation_experiments/results')
    checkpoints_dir = Path('../03_churn_prediction/checkpoints')
    
    try:
        # Load SMOTE experiment results
        smote_path = results_dir / 'smote_comparison_results.pkl'
        
        if not smote_path.exists():
            print("⚠️  SMOTE results not found, skipping...")
            return
        
        with open(smote_path, 'rb') as f:
            smote_results = pickle.load(f)
        
        # Get test data
        df = pd.read_csv('../data/Customer-Churn-Records.csv')
        df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
        df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
        
        # Prepare features
        weak_features = ['card_type', 'hascrcard', 'satisfaction_score', 
                         'point_earned', 'estimatedsalary', 'creditscore', 'complain']
        existing_weak = [f for f in weak_features if f in df.columns]
        if existing_weak:
            df = df.drop(existing_weak, axis=1)
        
        df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
        df['age_group'] = pd.cut(df['age'], 
                                  bins=[0, 30, 40, 50, 60, 70, 100],
                                  labels=['18-30', '31-40', '41-50', '51-60', '61-70', '70+'])
        df_encoded = pd.get_dummies(df, columns=['geography', 'age_group'], drop_first=True, dtype=int)
        
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(df_encoded, test_size=0.2, random_state=42, stratify=df_encoded['exited'])
        
        X_test = test.drop('exited', axis=1)
        y_test = test['exited']
        
        # Get predictions for both approaches
        if isinstance(smote_results, dict) and 'models' in smote_results:
            rf_cw_model = smote_results['models'].get('Class Weight')
            rf_smote_model = smote_results['models'].get('SMOTE')
        else:
            rf_cw_model = None
            rf_smote_model = None
        
        if rf_cw_model is None or rf_smote_model is None:
            print("⚠️  Model results incomplete, skipping...")
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for name, model in [('Class Weight', rf_cw_model), ('SMOTE', rf_smote_model)]:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = model.decision_function(X_test)
            
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
        ax.set_title('ROC Curves: Class Weight vs SMOTE', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='lower right', fontsize=11)
        
        # Ensure all spines are visible and black
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / '25_smote_roc_comparison.png')
        plt.close()
        print("✓ Saved: 25_smote_roc_comparison.png")
        
    except Exception as e:
        print(f"⚠️  Error generating SMOTE ROC comparison: {e}")


def main():
    """Main execution"""
    print("="*80)
    print("GENERATING MODEL COMPARISON PLOTS FOR LATEX REPORT")
    print("="*80)
    
    generate_model_comparison_roc()
    generate_smote_class_distribution()
    generate_smote_roc_comparison()
    
    print("\n" + "="*80)
    print("✅ PLOT GENERATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()

