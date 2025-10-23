"""
EDA Utility Functions for Bank Customer Churn Analysis

This module contains standardized plotting and analysis functions for exploratory data analysis.
All functions follow consistent styling with a green/red color scheme for churn visualization.

Color Scheme:
    - NO_CHURN (Green): #66BB6A - Represents customers who did not churn
    - CHURN (Red): #EF5350 - Represents customers who churned

Author: Data Science Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter, MultipleLocator
from matplotlib.patches import Patch

# ============================================================================
# COLOR SCHEME CONSTANTS
# ============================================================================

NO_CHURN = "#55A868"   # green for non-churned customers
CHURN = "#C44E52"      # red for churned customers


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def stacked_plot(
    df, group, target, ax=None, title=None, show_legend=False,
    palette=(NO_CHURN, CHURN), label_threshold=0.05, rotation=0
):
    """
    Create clean stacked percentage bar charts for churn analysis.
    
    This function creates a stacked bar chart showing the proportion of churned vs
    non-churned customers for each category in a grouping variable. Percentages are
    displayed inside each colored segment.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data
    group : str
        Column name to group by (x-axis categories)
    target : str
        Target column name (should be binary, e.g., 'exited', 'Churn')
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure
    title : str, optional
        Title for the plot
    show_legend : bool, default False
        Whether to show the legend (typically only on one subplot in a grid)
    palette : tuple, default (NO_CHURN, CHURN)
        Colors for (No, Yes) categories
    label_threshold : float, default 0.05
        Minimum proportion to show percentage label (hides tiny segments)
    rotation : int, default 0
        Rotation angle for x-axis labels
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot
        
    Examples
    --------
    >>> stacked_plot(df, "geography", "exited", title="Churn by Country", show_legend=True)
    >>> 
    >>> # Multiple subplots
    >>> fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    >>> stacked_plot(df, "gender", "exited", ax=axes[0,0], title="Gender")
    >>> stacked_plot(df, "geography", "exited", ax=axes[0,1], title="Geography", show_legend=True)
    """
    is_standalone = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    fig = ax.figure
    ax.set_facecolor("white")
    fig.set_facecolor("white")

    # Calculate proportions
    counts = df.groupby([group, target], observed=True).size()
    totals = counts.groupby(level=0, observed=True).sum()
    prop = (counts / totals).reset_index().pivot(index=group, columns=target, values=0).fillna(0)

    # Normalize target columns to "No"/"Yes" and ensure proper order
    colmap = {}
    for c in prop.columns:
        if c is False or c == 0:
            colmap[c] = "No"
        elif c is True or c == 1:
            colmap[c] = "Yes"
        else:
            colmap[c] = str(c)

    prop = prop.rename(columns=colmap)

    # Ensure both columns exist and are ordered as ["No", "Yes"]
    if "No" not in prop.columns:
        prop["No"] = 0.0
    if "Yes" not in prop.columns:
        prop["Yes"] = 0.0
    prop = prop[["No", "Yes"]]

    # Create stacked bar plot
    prop.plot(
        kind="bar", stacked=True, ax=ax, legend=False,
        color=palette, width=0.7, edgecolor="white", linewidth=1
    )

    # Format axes
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.tick_params(axis="x", rotation=rotation)
    ax.set_xlabel(group.replace("_", " ").title(), fontsize=10, fontweight='bold')
    ax.set_ylabel("Proportion", fontsize=10, fontweight='bold')
    
    if title:
        title_pad = 30 if (show_legend and is_standalone) else 6
        ax.set_title(title, fontsize=12, weight="bold", pad=title_pad)

    # Black axis lines for framed appearance
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(1.0)
        ax.spines[s].set_color("black")

    # Add percentage labels inside each segment
    for container in ax.containers:
        heights = [patch.get_height() for patch in container]
        labels = [f"{h:.0%}" if h >= label_threshold else "" for h in heights]
        ax.bar_label(
            container, labels=labels, label_type="center",
            fontsize=9, color="white", fontweight="bold", padding=0
        )

    # Add shared legend for grid layouts
    if show_legend and len(ax.containers) >= 2:
        for lg in fig.legends:
            lg.remove()
        handles = [
            Patch(facecolor=palette[0], label="No Churn"),
            Patch(facecolor=palette[1], label="Churn")
        ]
        fig.legend(
            handles=handles, ncol=2, frameon=True, facecolor="white",
            edgecolor="#757575", framealpha=0.95,
            loc="upper center", bbox_to_anchor=(0.5, 0.98), fontsize=10
        )

    ax.margins(x=0.02)
    return ax


def countplot_enhanced(
    df, x, hue=None, ax=None, title=None, 
    xlabel=None, ylabel="Count", figsize=(18, 6),
    palette=None
):
    """
    Create enhanced count plots with consistent styling for churn analysis.
    
    This function creates a seaborn countplot with consistent styling matching
    the stacked_plot function, including proper color schemes, legends, and grids.
    
    Parameters
    ----------
    df : pd.DataFrame
        The data to plot
    x : str
        Column name for x-axis
    hue : str, optional
        Column name for grouping (e.g., 'exited', 'Churn')
    ax : matplotlib.axes.Axes, optional
        If provided, plot on this axis. Otherwise create new figure
    title : str, optional
        Custom title for the plot
    xlabel : str, optional
        Custom x-axis label (defaults to x column name)
    ylabel : str, default "Count"
        Y-axis label
    figsize : tuple, default (18, 6)
        Figure size if creating new figure
    palette : dict, optional
        Custom color palette. If None and hue relates to churn, uses standard colors
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot
        
    Examples
    --------
    >>> countplot_enhanced(df, x="tenure", hue="exited", 
    ...                    title="Tenure Distribution by Churn Status")
    >>> 
    >>> countplot_enhanced(df, x="numofproducts", hue="geography",
    ...                    palette={"France": "#1E88E5", "Spain": "#FFA726", "Germany": "#66BB6A"})
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Set default palette for Churn analysis
    if palette is None and hue in ["Churn", "exited"]:
        palette = {"No": NO_CHURN, "Yes": CHURN, False: NO_CHURN, True: CHURN, 0: NO_CHURN, 1: CHURN}
    
    # Create countplot
    sns.countplot(data=df, x=x, hue=hue, palette=palette, ax=ax)
    
    # Set title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Set labels
    if xlabel is None:
        xlabel = x.replace("_", " ").title()
    ax.set_xlabel(xlabel, fontsize=12, fontweight='semibold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='semibold')
    
    # Enhanced legend if hue is used
    if hue:
        legend_title = hue.replace("_", " ").title()
        if hue in ["Churn", "exited"]:
            legend_labels = ["No Churn", "Churn"]
            legend_title = "Customer Status"
        else:
            legend_labels = None
        
        ax.legend(
            title=legend_title, labels=legend_labels,
            title_fontsize=11, fontsize=10, frameon=True,
            facecolor="white", edgecolor="#757575", framealpha=0.95,
            loc='upper right'
        )
    
    # Add grid for easier reading
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='#BDBDBD', linewidth=1)
    ax.set_axisbelow(True)
    
    # Enhance tick parameters
    ax.tick_params(
        axis='both', which='major', direction='out',
        length=6, width=1.5, labelsize=9, pad=6
    )
    
    return ax


def density_plot_enhanced(
    df, x_col, group_col, title=None, 
    xlabel=None, ylabel="Density", figsize=(12, 6),
    alpha=0.3, bins=30
):
    """
    Create enhanced density/distribution plots with histograms and KDE curves.
    
    This function creates overlapping histograms with KDE curves for different groups,
    useful for comparing distributions across categories.
    
    Parameters
    ----------
    df : pd.DataFrame
        The data to plot
    x_col : str
        Column name for x-axis (continuous variable)
    group_col : str
        Column name for grouping/coloring (categorical variable)
    title : str, optional
        Custom title for the plot
    xlabel : str, optional
        Custom x-axis label (defaults to x_col name)
    ylabel : str, default "Density"
        Y-axis label
    figsize : tuple, default (12, 6)
        Figure size
    alpha : float, default 0.3
        Transparency of histograms
    bins : int, default 30
        Number of bins for histogram
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot
        
    Examples
    --------
    >>> density_plot_enhanced(df, x_col='tenure', group_col='isactivemember',
    ...                       title="Tenure Distribution by Active Status")
    >>> 
    >>> density_plot_enhanced(df, x_col='age', group_col='geography')
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#E8E8E8")
    
    groups = df[group_col].unique()
    
    # Define color mapping
    color_map = {
        "No": "#2E7D32", "Yes": "#C62828",
        False: "#2E7D32", True: "#C62828",
        0: "#2E7D32", 1: "#C62828",
        "France": "#1565C0", "Spain": "#EF6C00", "Germany": "#6A1B9A",
        "SILVER": "#9E9E9E", "GOLD": "#FFD700", "PLATINUM": "#E5E4E2", "DIAMOND": "#B9F2FF"
    }
    
    # Calculate common bins for alignment
    data_min = df[x_col].min()
    data_max = df[x_col].max()
    bin_edges = np.linspace(data_min, data_max, bins + 1)
    
    # Plot each group
    for group in sorted(groups, key=str):
        group_data = df[df[group_col] == group][x_col]
        color = color_map.get(group, "#9E9E9E")
        
        sns.histplot(
            group_data, kde=True, stat='density', alpha=alpha,
            label=str(group), color=color, bins=bin_edges, ax=ax
        )
    
    # Set title and labels
    if title is None:
        title = f"{x_col.replace('_', ' ').title()} Distribution by {group_col.replace('_', ' ').title()}"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    if xlabel is None:
        xlabel = x_col.replace("_", " ").title()
        if x_col.lower() == "tenure":
            xlabel += " (years)"
    ax.set_xlabel(xlabel, fontsize=12, fontweight='semibold', labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, fontweight='semibold', labelpad=10)
    
    # Enhanced legend
    ax.legend(
        title=group_col.replace("_", " ").title(),
        title_fontsize=11, fontsize=10, frameon=True,
        facecolor="white", edgecolor="#757575", framealpha=0.95,
        loc='upper center', ncol=len(groups), columnspacing=2.0
    )
    
    # Add grid
    ax.xaxis.grid(True, linestyle='-', alpha=0.7, color='white', linewidth=1.5)
    ax.yaxis.grid(True, linestyle='-', alpha=0.7, color='white', linewidth=1.5)
    ax.set_axisbelow(True)
    
    # Enhance ticks
    ax.tick_params(
        axis='both', which='major', direction='out',
        length=6, width=1.5, labelsize=10, pad=8
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return ax


def kde_comparison_plot(
    df_group1, df_group2, column, 
    group1_label="Churn", group2_label="Not Churn",
    title=None, xlabel=None, ylabel="Density",
    figsize=(12, 6), linewidth=2.5
):
    """
    Create KDE comparison plots for two groups (typically churned vs non-churned).
    
    This function creates smooth density curves comparing a continuous variable
    between two groups, ideal for comparing churned vs retained customers.
    
    Parameters
    ----------
    df_group1 : pd.DataFrame
        First group data (typically churned customers)
    df_group2 : pd.DataFrame
        Second group data (typically non-churned customers)
    column : str
        Column name to plot (continuous variable)
    group1_label : str, default "Churn"
        Label for first group
    group2_label : str, default "Not Churn"
        Label for second group
    title : str, optional
        Custom title for the plot
    xlabel : str, optional
        Custom x-axis label (defaults to column name)
    ylabel : str, default "Density"
        Y-axis label
    figsize : tuple, default (12, 6)
        Figure size
    linewidth : float, default 2.5
        Width of KDE lines
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot
        
    Examples
    --------
    >>> churned = df[df['exited'] == True]
    >>> not_churned = df[df['exited'] == False]
    >>> kde_comparison_plot(churned, not_churned, 'balance',
    ...                     title="Account Balance Distribution by Churn Status")
    >>> 
    >>> kde_comparison_plot(churned, not_churned, 'creditscore',
    ...                     xlabel="Credit Score")
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#FAFAFA")
    fig.set_facecolor("white")
    
    # Plot KDE curves
    sns.kdeplot(
        df_group1[column], label=group1_label, ax=ax, 
        color=CHURN, linewidth=linewidth, fill=True, alpha=0.2
    )
    sns.kdeplot(
        df_group2[column], label=group2_label, ax=ax, 
        color=NO_CHURN, linewidth=linewidth, fill=True, alpha=0.2
    )
    
    # Set title and labels
    if title is None:
        title = f"Distribution of {column.replace('_', ' ').title()} by Churn Status"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    if xlabel is None:
        xlabel = column.replace("_", " ").title()
        if any(word in column.lower() for word in ['salary', 'balance', 'charge']):
            xlabel += " ($)"
    ax.set_xlabel(xlabel, fontsize=12, fontweight='semibold', labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, fontweight='semibold', labelpad=10)
    
    # Enhanced legend
    ax.legend(
        title="Customer Status", title_fontsize=11, fontsize=10,
        frameon=True, facecolor="white", edgecolor="#757575",
        framealpha=0.95, loc='upper right'
    )
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='#BDBDBD', linewidth=1)
    ax.set_axisbelow(True)
    
    # Enhance ticks
    ax.tick_params(
        axis='both', which='major', direction='out',
        length=6, width=1.5, labelsize=10, pad=8
    )
    
    plt.tight_layout()
    return ax


# ============================================================================
# ANALYSIS HELPER FUNCTIONS
# ============================================================================

def print_churn_summary(df, segment_col, target_col='exited', segment_label=None):
    """
    Print a formatted summary table of churn rates by segment.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to analyze
    segment_col : str
        Column name to segment by
    target_col : str, default 'exited'
        Target column (churn indicator)
    segment_label : str, optional
        Custom label for the segment column
        
    Examples
    --------
    >>> print_churn_summary(df, 'geography', 'exited')
    >>> print_churn_summary(df, 'age_group', 'exited', segment_label='Age Group')
    """
    print("=" * 90)
    if segment_label is None:
        segment_label = segment_col.replace("_", " ").title()
    print(f"CHURN ANALYSIS: {segment_label}")
    print("=" * 90)
    
    # Ensure target is boolean
    if df[target_col].dtype != bool:
        target = df[target_col].astype(bool)
    else:
        target = df[target_col]
    
    # Calculate statistics
    summary_data = []
    for segment in sorted(df[segment_col].unique(), key=str):
        segment_df = df[df[segment_col] == segment]
        total = len(segment_df)
        # vc = segment_df[target_col].value_counts()
        seg_y = segment_df[target_col].astype(bool)   # force boolean labels: {False, True}
        vc = seg_y.value_counts()                      # index is exactly {False, True}
        
        # Convert to dict to avoid pandas FutureWarning about positional indexing
        # vc_dict = vc.to_dict()
        
        # # Handle both boolean (True/False) and integer (1/0) target columns
        # churned = int(vc_dict.get(True, vc_dict.get(1, 0)))
        # retained = int(vc_dict.get(False, vc_dict.get(0, 0)))
        # (new)
        churned  = int(vc.get(True, 0))
        retained = int(vc.get(False, 0))

        churn_rate = (churned / total * 100.0) if total > 0 else 0.0
        
        summary_data.append({
            'segment': str(segment),
            'total': total,
            'churned': churned,
            'retained': retained,
            'churn_rate': churn_rate
        })
    
    # Print table
    print(f"\n{segment_label:<25} {'Total':>8} {'Churned':>8} {'Retained':>9} {'Churn Rate':>12}")
    print("-" * 90)
    
    for row in summary_data:
        print(f"{row['segment']:<25} {row['total']:>8} {row['churned']:>8} "
              f"{row['retained']:>9} {row['churn_rate']:>11.2f}%")
    
    print("-" * 90)
    
    # Key insights
    print("\n📊 KEY INSIGHTS:")
    print("-" * 90)
    
    lowest = min(summary_data, key=lambda x: x['churn_rate'])
    highest = max(summary_data, key=lambda x: x['churn_rate'])
    
    print(f"✓ LOWEST Churn:  {lowest['segment']} ({lowest['churn_rate']:.2f}%)")
    print(f"✗ HIGHEST Churn: {highest['segment']} ({highest['churn_rate']:.2f}%)")
    
    if lowest['churn_rate'] > 0:
        ratio = highest['churn_rate'] / lowest['churn_rate']
        print(f"\n  → Relative risk: {ratio:.1f}× higher churn in {highest['segment']} vs {lowest['segment']}")
    
    print("=" * 90)
    print()


def correlation_heatmap(df, figsize=(14, 10), annot=True, cmap='coolwarm'):
    """
    Create a correlation heatmap for numerical features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with numerical columns
    figsize : tuple, default (14, 10)
        Figure size
    annot : bool, default True
        Whether to annotate cells with correlation values
    cmap : str, default 'coolwarm'
        Colormap name
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the heatmap
        
    Examples
    --------
    >>> numerical_cols = ['creditscore', 'age', 'tenure', 'balance', 'numofproducts']
    >>> correlation_heatmap(df[numerical_cols])
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate correlation matrix
    corr = df.corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Create heatmap
    sns.heatmap(
        corr, mask=mask, annot=annot, fmt='.2f', cmap=cmap,
        center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    ax.set_title('Correlation Heatmap of Numerical Features', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return ax


def plot_demographic_grid(df, target_col='exited'):
    """
    Create a 1x3 grid of TRUE demographic feature stacked plots.
    
    Demographics are innate or relatively stable characteristics:
    - Gender (innate)
    - Geography (location, relatively stable)
    - Age_group (innate, changes slowly)
    
    Note: HasCrCard and IsActiveMember are NOT demographics - they are
    product ownership and behavioral features respectively.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to analyze (must have 'age_group' already created)
    target_col : str, default 'exited'
        Target column name for churn
        
    Returns
    -------
    tuple
        (fig, axes) matplotlib figure and axes objects
        
    Examples
    --------
    >>> # First create age_group in your dataframe
    >>> df['age_group'] = pd.cut(df['age'], bins=[18,30,40,50,60,70,100],
    ...                           labels=['18-30','31-40','41-50','51-60','61-70','70+'])
    >>> fig, axes = plot_demographic_grid(df)
    >>> plt.show()
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Churn Analysis by Demographic Features', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    features = [
        ("gender", "Gender"),
        ("geography", "Geography"),
        ("age_group", "Age Group")
    ]
    
    for idx, (feature, title) in enumerate(features):
        ax = axes[idx]
        
        if feature in df.columns:
            stacked_plot(
                df, feature, target_col,
                ax=ax, title=title,
                show_legend=(idx == 1)  # Show legend on middle plot
            )
        else:
            ax.text(0.5, 0.5, f"'{feature}' column not found\n(create age_group before calling this function)",
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(title)
    
    plt.tight_layout()
    return fig, axes


# ============================================================================
# DATA VALIDATION FUNCTIONS
# ============================================================================

def validate_data_consistency(df, info_text="Data Validation Check"):
    """
    Validate basic data consistency and print summary.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to validate
    info_text : str, default "Data Validation Check"
        Header text for the validation report
        
    Examples
    --------
    >>> validate_data_consistency(df, "Initial Data Check")
    """
    print("=" * 80)
    print(info_text)
    print("=" * 80)
    print(f"\nDataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  ✓ No missing values detected")
    else:
        print(missing[missing > 0])
    
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")
    print("=" * 80)
    print()

