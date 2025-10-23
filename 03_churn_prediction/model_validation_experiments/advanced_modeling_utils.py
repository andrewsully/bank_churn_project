"""
Advanced Modeling Utilities

Shared functions for advanced modeling experiments in 04_advanced_modeling/
Ensures consistent data preprocessing and evaluation across all experiments.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path


def load_preprocessed_data(data_path='data/Customer-Churn-Records.csv', 
                          random_state=42):
    """
    Load and preprocess data identically to 03_churn_prediction.
    
    This function replicates the exact preprocessing steps from the main
    churn prediction notebook to ensure consistency across all experiments.
    
    Parameters:
    -----------
    data_path : str
        Path to the Customer-Churn-Records.csv file
    random_state : int
        Random seed for train/test split (must be 42 to match 03_churn_prediction)
    
    Returns:
    --------
    X_train : DataFrame
        Training features
    X_test : DataFrame
        Test features
    y_train : Series
        Training target
    y_test : Series
        Test target
    feature_names : list
        List of feature column names
    
    Notes:
    ------
    Preprocessing steps (matching 03_churn_prediction exactly):
    1. Drop identifiers: RowNumber, CustomerId, Surname
    2. Drop weak features: card_type, hascrcard, satisfaction_score, 
       point_earned, estimatedsalary, creditscore
    3. Drop dominant feature: complain (too strong: 99.5% churn)
    4. Convert binary columns to int (isactivemember, exited)
    5. Encode gender: Male=0, Female=1
    6. Create age_group feature (bins: 0-30, 31-40, 41-50, 51-60, 61-70, 70+)
    7. One-hot encode: geography, age_group
    8. Stratified train/test split (80/20, random_state=42)
    """
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Step 1: Drop identifier columns
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    
    # Standardize column names
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
    
    # Step 2: Drop weak features
    weak_features = ['card_type', 'hascrcard', 'satisfaction_score', 
                     'point_earned', 'estimatedsalary', 'creditscore']
    df = df.drop(weak_features, axis=1)
    
    # Step 3: Drop dominant feature (complain)
    df = df.drop('complain', axis=1)
    
    # Step 4: Convert binary columns to int
    binary_cols = ['isactivemember', 'exited']
    for col in binary_cols:
        df[col] = df[col].astype(int)
    
    # Step 5: Encode gender
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    
    # Step 6: Create age_group feature
    df['age_group'] = pd.cut(df['age'], 
                              bins=[0, 30, 40, 50, 60, 70, 100],
                              labels=['18-30', '31-40', '41-50', '51-60', '61-70', '70+'])
    
    # Step 7: One-hot encode categorical features
    categorical_cols = ['geography', 'age_group']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    
    # Step 8: Stratified train/test split
    train, test = train_test_split(df_encoded, test_size=0.2, random_state=random_state, 
                                   stratify=df_encoded['exited'])
    
    # Define feature matrix and target
    feature_cols = [col for col in df_encoded.columns if col != 'exited']
    X_train = train[feature_cols]
    y_train = train['exited']
    X_test = test[feature_cols]
    y_test = test['exited']
    
    return X_train, X_test, y_train, y_test, feature_cols


def compare_models_results(model_results, y_test, model_names=None):
    """
    Compare multiple model results and create summary table.
    
    Parameters:
    -----------
    model_results : dict
        Dictionary with model names as keys and tuples of (predictions, probabilities) as values
        Example: {'RF': (y_pred, y_proba), 'XGB': (y_pred, y_proba)}
    y_test : array-like
        True labels
    model_names : list, optional
        List of model names. If None, uses dict keys
    
    Returns:
    --------
    results_df : DataFrame
        Comparison table with metrics for each model
    """
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                f1_score, roc_auc_score)
    
    if model_names is None:
        model_names = list(model_results.keys())
    
    results = []
    for name in model_names:
        y_pred, y_proba = model_results[name]
        
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_proba)
        })
    
    results_df = pd.DataFrame(results)
    return results_df


def visualize_smote_impact(y_before, y_after, save_path=None):
    """
    Visualize class distribution before and after SMOTE.
    
    Parameters:
    -----------
    y_before : array-like
        Labels before SMOTE
    y_after : array-like
        Labels after SMOTE
    save_path : str, optional
        Path to save the plot
    """
    import matplotlib.pyplot as plt
    from collections import Counter
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Before SMOTE
    before_counts = Counter(y_before)
    axes[0].bar(before_counts.keys(), before_counts.values(), color='steelblue')
    axes[0].set_title('Class Distribution Before SMOTE')
    axes[0].set_xlabel('Churn Status')
    axes[0].set_ylabel('Count')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Not Churn', 'Churn'])
    
    # After SMOTE
    after_counts = Counter(y_after)
    axes[1].bar(after_counts.keys(), after_counts.values(), color='seagreen')
    axes[1].set_title('Class Distribution After SMOTE')
    axes[1].set_xlabel('Churn Status')
    axes[1].set_ylabel('Count')
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Not Churn', 'Churn'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.show()


def feature_importance_comparison(models_dict, feature_names, top_n=10):
    """
    Compare feature importance across multiple models.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with model names as keys and fitted models as values
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to display
    
    Returns:
    --------
    importance_df : DataFrame
        DataFrame with feature importance for each model
    """
    importance_data = {}
    
    for name, model in models_dict.items():
        if hasattr(model, 'feature_importances_'):
            importance_data[name] = model.feature_importances_
        else:
            print(f"⚠ {name} model does not have feature_importances_ attribute")
            continue
    
    importance_df = pd.DataFrame(importance_data, index=feature_names)
    
    # Plot comparison
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get top N features by average importance
    avg_importance = importance_df.mean(axis=1).sort_values(ascending=False)
    top_features = avg_importance.head(top_n).index
    
    # Plot
    plt.figure(figsize=(10, 6))
    importance_df.loc[top_features].plot(kind='barh', figsize=(10, 6))
    plt.title(f'Feature Importance Comparison (Top {top_n} Features)')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.legend(title='Model')
    plt.tight_layout()
    plt.show()
    
    return importance_df


def save_experiment_results(results, filename, results_dir='results'):
    """
    Save experiment results to pickle file.
    
    Parameters:
    -----------
    results : dict
        Dictionary of results to save
    filename : str
        Name of the pickle file
    results_dir : str
        Directory to save results
    """
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)
    
    filepath = results_path / filename
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"✓ Results saved to {filepath}")


def load_experiment_results(filename, results_dir='results'):
    """
    Load experiment results from pickle file.
    
    Parameters:
    -----------
    filename : str
        Name of the pickle file
    results_dir : str
        Directory containing results
    
    Returns:
    --------
    results : dict
        Loaded results dictionary
    """
    filepath = Path(results_dir) / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    
    print(f"✓ Results loaded from {filepath}")
    return results


def plot_roc_comparison(models_dict, X_test, y_test, title="ROC Curve Comparison"):
    """
    Plot ROC curves for multiple models.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with model names as keys and fitted models as values
    X_test : DataFrame
        Test features
    y_test : array-like
        True labels
    title : str
        Plot title
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    plt.figure(figsize=(8, 6))
    
    for name, model in models_dict.items():
        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
    
    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices_comparison(models_dict, X_test, y_test, 
                                       figsize=(15, 4)):
    """
    Plot side-by-side confusion matrices for multiple models.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with model names as keys and fitted models as values
    X_test : DataFrame
        Test features
    y_test : array-like
        True labels
    figsize : tuple
        Figure size
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, model) in enumerate(models_dict.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Not Churn', 'Churn'],
                   yticklabels=['Not Churn', 'Churn'])
        axes[idx].set_title(f'{name}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test the load_preprocessed_data function
    print("Testing load_preprocessed_data function...")
    X_train, X_test, y_train, y_test, feature_names = load_preprocessed_data()
    
    print(f"\n✓ Data loaded successfully!")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Features: {len(feature_names)}")
    print(f"Train churn rate: {y_train.mean():.2%}")
    print(f"Test churn rate: {y_test.mean():.2%}")
    print(f"\nFirst 5 features: {feature_names[:5]}")

