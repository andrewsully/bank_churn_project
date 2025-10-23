"""
Machine Learning Modeling Utility Functions
============================================
Reusable functions for model training, evaluation, and interpretation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report,
    roc_auc_score, roc_curve, precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score
)


def evaluate_classification_model(
    model, 
    train_x, train_y, 
    test_x, test_y, 
    feature_names,
    feature_importance_type='feature_importances',
    figsize=(16, 10),
    model_name="Model"
):
    """
    Comprehensive evaluation of a classification model with visualizations.
    
    Parameters
    ----------
    model : sklearn estimator
        Trained classification model
    train_x : pd.DataFrame
        Training features
    train_y : pd.Series
        Training target
    test_x : pd.DataFrame
        Test features
    test_y : pd.Series
        Test target
    feature_names : list
        List of feature names
    feature_importance_type : str
        'feature_importances' for tree models, 'coef' for linear models
    figsize : tuple
        Figure size for plots
    model_name : str
        Name of the model for display
        
    Returns
    -------
    dict
        Dictionary containing evaluation metrics
    """
    # Make predictions
    train_pred = model.predict(train_x)
    test_pred = model.predict(test_x)
    test_proba = model.predict_proba(test_x)[:, 1]
    
    # Calculate metrics
    metrics = {
        'train_accuracy': accuracy_score(train_y, train_pred),
        'test_accuracy': accuracy_score(test_y, test_pred),
        'test_precision': precision_score(test_y, test_pred),
        'test_recall': recall_score(test_y, test_pred),
        'test_f1': f1_score(test_y, test_pred),
        'test_roc_auc': roc_auc_score(test_y, test_proba)
    }
    
    # Feature importances
    if feature_importance_type == 'feature_importances':
        importances = model.feature_importances_
    elif feature_importance_type == 'coef':
        importances = np.abs(model.coef_.ravel())
    else:
        importances = None
    
    if importances is not None:
        feature_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
    else:
        feature_imp_df = None
    
    # Create visualizations
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(test_y, test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False)
    ax1.set_title('Confusion Matrix (Test Set)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # 2. ROC Curve
    ax2 = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(test_y, test_proba)
    ax2.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC (AUC = {metrics["test_roc_auc"]:.3f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve', fontweight='bold', fontsize=12)
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)
    
    # 3. Precision-Recall Curve
    ax3 = fig.add_subplot(gs[1, 0])
    precision_vals, recall_vals, _ = precision_recall_curve(test_y, test_proba)
    avg_precision = average_precision_score(test_y, test_proba)
    ax3.plot(recall_vals, precision_vals, color='blue', lw=2,
             label=f'PR (AP = {avg_precision:.3f})')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision-Recall Curve', fontweight='bold', fontsize=12)
    ax3.legend(loc="best")
    ax3.grid(alpha=0.3)
    
    # 4. Metrics Summary Table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    metrics_data = [
        ['Train Accuracy', f"{metrics['train_accuracy']:.4f}"],
        ['Test Accuracy', f"{metrics['test_accuracy']:.4f}"],
        ['Precision', f"{metrics['test_precision']:.4f}"],
        ['Recall', f"{metrics['test_recall']:.4f}"],
        ['F1 Score', f"{metrics['test_f1']:.4f}"],
        ['ROC AUC', f"{metrics['test_roc_auc']:.4f}"],
    ]
    table = ax4.table(cellText=metrics_data, colLabels=['Metric', 'Value'],
                      cellLoc='left', loc='center',
                      colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#6B7280')
        else:
            cell.set_edgecolor('#E5E7EB')
    ax4.set_title('Performance Metrics', fontweight='bold', fontsize=12, pad=20)
    
    # 5. Feature Importance (if available)
    if feature_imp_df is not None:
        ax5 = fig.add_subplot(gs[2, :])
        top_n = min(20, len(feature_imp_df))
        top_features = feature_imp_df.head(top_n)
        ax5.barh(range(len(top_features)), top_features['importance'])
        ax5.set_yticks(range(len(top_features)))
        ax5.set_yticklabels(top_features['feature'])
        ax5.invert_yaxis()
        ax5.set_xlabel('Importance', fontweight='bold')
        ax5.set_title(f'Top {top_n} Feature Importances', fontweight='bold', fontsize=12)
        ax5.grid(axis='x', alpha=0.3)
    
    fig.suptitle(f'{model_name} - Comprehensive Evaluation', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()
    
    # Print classification report
    print("="*80)
    print(f"{model_name} - CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(test_y, test_pred, target_names=['Retained', 'Churned']))
    print("="*80)
    
    return metrics, feature_imp_df


def plot_grid_search_heatmap(grid_search, param1, param2, 
                              title="Grid Search Results",
                              score_name="mean_test_score",
                              figsize=(10, 6)):
    """
    Plot heatmap of grid search results for two parameters.
    
    Parameters
    ----------
    grid_search : GridSearchCV
        Fitted grid search object
    param1 : str
        First parameter name (for y-axis)
    param2 : str
        Second parameter name (for x-axis)
    title : str
        Plot title
    score_name : str
        Score metric to display
    figsize : tuple
        Figure size
    """
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    # Convert param columns to string for pivot
    param1_col = f'param_{param1}'
    param2_col = f'param_{param2}'
    results_df[param1_col] = results_df[param1_col].astype(str)
    results_df[param2_col] = results_df[param2_col].astype(str)
    
    # Create pivot table
    pivot_table = pd.pivot_table(
        results_df, 
        values=score_name,
        index=param1_col,
        columns=param2_col
    )
    
    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='YlOrRd', cbar_kws={'label': 'Score'})
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(param2.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    plt.ylabel(param1.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


def quick_pdp(
    model, test_x, feature_name,
    centered=True,
    n_samples=None,
    feature_type="auto",
    title=None,
    figsize=(10, 6)
):
    """
    Create Partial Dependence Plot (PDP) for a feature.
    
    Parameters
    ----------
    model : sklearn estimator
        Trained model with predict_proba method
    test_x : pd.DataFrame
        Test features
    feature_name : str
        Name of feature to plot
    centered : bool
        If True, center the PDP at baseline (first grid point)
    n_samples : int or None
        Number of samples to use (None = all)
    feature_type : str
        'auto', 'binary', or 'continuous'
    title : str or None
        Custom title
    figsize : tuple
        Figure size
    """
    # Sample data if requested
    X_plot = (
        test_x.sample(n=n_samples, random_state=42)
        if (n_samples and len(test_x) > n_samples)
        else test_x.copy()
    )
    
    # Determine feature type
    if feature_type == "auto":
        is_binary = (X_plot[feature_name].nunique() == 2)
    else:
        is_binary = (feature_type == "binary")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Binary feature
    if is_binary:
        X0 = X_plot.copy()
        X0[feature_name] = 0
        X1 = X_plot.copy()
        X1[feature_name] = 1
        
        p0 = model.predict_proba(X0)[:, 1]
        p1 = model.predict_proba(X1)[:, 1]
        
        y = np.array([p0.mean(), p1.mean()])
        ice = np.vstack([p0, p1])
        
        if centered:
            y_main = y - y[0]
            ice_center = ice - ice[0, :]
            ci = ice_center.std(axis=1)
            ax.axhline(0, color="black", ls="--", lw=1, alpha=0.6)
            ax.set_ylabel("Δ Predicted Probability (centered)", fontweight='bold')
            subtitle = "Change in churn probability: P(1) − P(0). Shaded = ±1 SD"
        else:
            y_main = y
            ci = ice.std(axis=1)
            ax.set_ylabel("Predicted Churn Probability", fontweight='bold')
            subtitle = "Predicted churn probability. Shaded = ±1 SD"
        
        x = np.array([0, 1])
        ax.fill_between(x, y_main - ci, y_main + ci, alpha=0.25, label="±1 std")
        ax.plot(x, y_main, marker="o", lw=2.5, markersize=8, label="Mean PDP")
        ax.set_xticks([0, 1])
        ax.set_xlim(-0.1, 1.1)
        ax.set_xlabel(feature_name.replace('_', ' ').title(), fontweight='bold')
    
    # Continuous feature
    else:
        xraw = X_plot[feature_name].to_numpy()
        qs = np.linspace(0.01, 0.99, 30)
        grid = np.quantile(xraw, qs)
        
        pdp_vals = []
        ice_rows = []
        for v in grid:
            tmp = X_plot.copy()
            tmp[feature_name] = v
            proba = model.predict_proba(tmp)[:, 1]
            ice_rows.append(proba)
            pdp_vals.append(proba.mean())
        
        pdp_vals = np.array(pdp_vals)
        ice = np.array(ice_rows)
        
        if centered:
            y_main = pdp_vals - pdp_vals[0]
            ice_center = ice - ice[0, :]
            ci = ice_center.std(axis=1)
            ax.axhline(0, color="black", ls="--", lw=1, alpha=0.6)
            ax.set_ylabel("Δ Predicted Probability (centered)", fontweight='bold')
            subtitle = "Change in churn probability vs baseline. Shaded = ±1 SD"
        else:
            y_main = pdp_vals
            ci = ice.std(axis=1)
            ax.set_ylabel("Predicted Churn Probability", fontweight='bold')
            subtitle = "Predicted churn probability. Shaded = ±1 SD"
        
        ax.fill_between(grid, y_main - ci, y_main + ci, alpha=0.25, label="±1 std")
        ax.plot(grid, y_main, lw=2.5, label="Mean PDP")
        ax.set_xlabel(feature_name.replace('_', ' ').title(), fontweight='bold')
    
    # Formatting
    ttl = title or f'Partial Dependence: {feature_name.replace("_", " ").title()}'
    ax.set_title(ttl, fontsize=13, fontweight='bold', pad=25)
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, color="dimgray", style='italic')
    ax.legend(loc="best", frameon=True, fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def print_grid_search_summary(grid_search, grid_name="Grid Search"):
    """
    Print summary of grid search results.
    
    Parameters
    ----------
    grid_search : GridSearchCV
        Fitted grid search object
    grid_name : str
        Name of the grid search for display
    """
    print("="*80)
    print(f"{grid_name} - RESULTS")
    print("="*80)
    print(f"Best Score (CV): {grid_search.best_score_:.4f}")
    print(f"\nBest Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  - {param}: {value}")
    print("="*80)
    print()

