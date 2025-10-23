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

# Configure plotting - Reset to plain matplotlib style
sns.reset_orig()
plt.style.use('default')

# Set rcParams for LaTeX report plots
plt.rcParams['figure.dpi'] = 300  # High quality for reports
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
    
    colors = ['#55A868', '#C44E52']  # Green for retained, red for churned
    bars = ax.bar(['Retained', 'Churned'], churn_counts.values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add percentage labels inside bars
    for i, (bar, pct) in enumerate(zip(bars, churn_pct.values)):
        height = bar.get_height()
        # Position text in the middle of the bar (half the height)
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{int(height):,}\n({pct:.1f}%)',
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    ax.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
    ax.set_title('Overall Customer Churn Distribution', fontsize=14, fontweight='bold', pad=15)
    # Ensure all spines are visible and black
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
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
    
    # Churn rate by age group
    age_churn = df.groupby('age_group')['exited'].agg(['sum', 'count'])
    age_churn['churn_rate'] = (age_churn['sum'] / age_churn['count'] * 100)
    
    bars = axes[1].bar(age_churn.index.astype(str), age_churn['churn_rate'], 
                       color='#e74c3c', alpha=0.8, edgecolor='black')
    axes[1].set_xlabel('Age Group', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Churn Rate (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Churn Rate by Age Group', fontsize=13, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Ensure all spines are visible and black for both subplots
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_age_distribution_churn.png')
    plt.close()
    print("✓ Saved: 03_age_distribution_churn.png")
    
    # Plot 4: Number of Products Analysis
    print("\n📊 Generating: 04_products_churn_analysis.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Diverging U-shape palette for products
    product_colors = {1: '#2C7BB6', 2: '#ABD9E9', 3: '#F4A261', 4: '#D1495B'}
    
    # Distribution
    product_counts = df['numofproducts'].value_counts().sort_index()
    
    bars_dist = axes[0].bar(range(len(product_counts)), product_counts.values, 
                            alpha=0.8, edgecolor='black', width=0.8)
    # Set individual bar colors
    for i, (bar, idx) in enumerate(zip(bars_dist, product_counts.index)):
        color = product_colors.get(int(idx), '#2C7BB6')
        bar.set_color(color)
    axes[0].set_xlabel('Number of Products', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribution of Products per Customer', fontsize=13, fontweight='bold')
    axes[0].set_xticks(range(len(product_counts)))
    axes[0].set_xticklabels([int(x) for x in product_counts.index])
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(product_counts.values):
        axes[0].text(i, v, f'{v:,}', 
                     ha='center', va='bottom', fontweight='bold')
    
    # Churn rate by products
    product_churn = df.groupby('numofproducts')['exited'].agg(['sum', 'count'])
    product_churn['churn_rate'] = (product_churn['sum'] / product_churn['count'] * 100)
    
    bars = axes[1].bar(range(len(product_churn)), product_churn['churn_rate'], 
                       alpha=0.8, edgecolor='black', width=0.8)
    # Set individual bar colors
    for i, (bar, idx) in enumerate(zip(bars, product_churn.index)):
        color = product_colors.get(int(idx), '#2C7BB6')
        bar.set_color(color)
    axes[1].set_xlabel('Number of Products', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Churn Rate (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Churn Rate by Number of Products (U-Shape)', fontsize=13, fontweight='bold')
    axes[1].set_xticks(range(len(product_churn)))
    axes[1].set_xticklabels([int(x) for x in product_churn.index])
    # Ensure all spines are visible and black for both subplots
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[1].text(i, height,
                     f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_products_churn_analysis.png')
    plt.close()
    print("✓ Saved: 04_products_churn_analysis.png")
    
    # Plot 4b: Overall Churn Rate Donut Chart (Polished Version)
    print("\n📊 Generating: 04_products_donut_chart.png")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Calculate overall churn statistics
    churn_counts = df['exited'].value_counts()
    churn_pct = df['exited'].value_counts(normalize=True) * 100
    
    # Define data and appearance
    data_for_donut = [churn_counts[0], churn_counts[1]]  # Retained, Churned
    labels_for_donut = ['Retained', 'Churned']
    colors_for_donut = ['#55A868', '#C44E52']  # Green / Red
    
    # Donut chart (balanced proportions)
    wedges, texts, autotexts = ax.pie(
        data_for_donut,
        labels=labels_for_donut,
        colors=colors_for_donut,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.78,  # bring text closer to center
        labeldistance=1.05,  # move labels slightly outward
        wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'},
        textprops={'fontsize': 13, 'fontweight': 'bold', 'color': 'black'}
    )
    
    # Center circle (controls donut thickness)
    centre_circle = plt.Circle((0, 0), 0.62, fc='white')
    ax.add_artist(centre_circle)
    
    # Center text block
    baseline_churn = churn_pct[1]
    
    ax.text(
        0, 0.25, 'Overall Churn Rate',
        ha='center', va='center',
        fontsize=18, fontweight='bold', color='black'
    )
    ax.text(
        0, 0.02, 'Baseline',
        ha='center', va='center',
        fontsize=14, fontweight='semibold', color='#444444'
    )
    ax.text(
        0, -0.16, f'{baseline_churn:.1f}%',
        ha='center', va='center',
        fontsize=28, fontweight='bold', color='#C44E52'
    )
    
    # Title
    ax.set_title(
        'Customer Churn Distribution',
        fontsize=18, fontweight='bold', pad=30
    )
    
    # Clean layout (no axes or spines)
    ax.axis('equal')  # perfect circle
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_products_donut_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 04_products_donut_chart.png")
    
    # Plot 5: Active Member Status
    print("\n📊 Generating: 05_active_member_impact.png")
    stacked_plot(df, 'isactivemember', 'exited', show_legend=True)
    plt.suptitle('Active Member Status Impact on Churn', fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(OUTPUT_DIR / '05_active_member_impact.png')
    plt.close()
    print("✓ Saved: 05_active_member_impact.png")
    
    # Plot 6: Geography Impact
    print("\n📊 Generating: 06_geography_churn.png")
    stacked_plot(df, 'geography', 'exited', show_legend=True)
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
    # Ensure all spines are visible and black
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
    
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
    
    # Ensure all spines are visible and black
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '08_correlation_heatmap.png')
    plt.close()
    print("✓ Saved: 08_correlation_heatmap.png")
    
    # Plot 9: Weak Features Comparison (Why They Were Excluded)
    print("\n📊 Generating: weak_features_comparison.png")
    
    # Load raw data before dropping weak features
    raw_df = pd.read_csv('data/Customer-Churn-Records.csv')
    raw_df.columns = raw_df.columns.str.lower().str.replace(' ', '')
    
    # Calculate churn rates for weak features
    weak_features_info = []
    
    # Card Type (categorical)
    if 'card_type' in raw_df.columns:
        card_churn = raw_df.groupby('card_type')['exited'].agg(['sum', 'count'])
        card_churn['churn_rate'] = (card_churn['sum'] / card_churn['count'] * 100)
        weak_features_info.append({
            'name': 'Card Type',
            'type': 'categorical',
            'data': card_churn
        })
    
    # Credit Score (continuous - bin into quartiles)
    if 'creditscore' in raw_df.columns:
        raw_df['creditscore_bin'] = pd.qcut(raw_df['creditscore'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        credit_churn = raw_df.groupby('creditscore_bin')['exited'].agg(['sum', 'count'])
        credit_churn['churn_rate'] = (credit_churn['sum'] / credit_churn['count'] * 100)
        weak_features_info.append({
            'name': 'Credit Score',
            'type': 'continuous',
            'data': credit_churn
        })
    
    # Satisfaction Score (continuous - bin into quartiles)
    if 'satisfaction_score' in raw_df.columns:
        raw_df['satisfaction_bin'] = pd.qcut(raw_df['satisfaction_score'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        sat_churn = raw_df.groupby('satisfaction_bin')['exited'].agg(['sum', 'count'])
        sat_churn['churn_rate'] = (sat_churn['sum'] / sat_churn['count'] * 100)
        weak_features_info.append({
            'name': 'Satisfaction Score',
            'type': 'continuous',
            'data': sat_churn
        })
    
    # Points Earned (continuous - bin into quartiles)
    if 'point_earned' in raw_df.columns:
        raw_df['points_bin'] = pd.qcut(raw_df['point_earned'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        points_churn = raw_df.groupby('points_bin')['exited'].agg(['sum', 'count'])
        points_churn['churn_rate'] = (points_churn['sum'] / points_churn['count'] * 100)
        weak_features_info.append({
            'name': 'Points Earned',
            'type': 'continuous',
            'data': points_churn
        })
    
    # Estimated Salary (continuous - bin into quartiles)
    if 'estimatedsalary' in raw_df.columns:
        raw_df['salary_bin'] = pd.qcut(raw_df['estimatedsalary'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        salary_churn = raw_df.groupby('salary_bin')['exited'].agg(['sum', 'count'])
        salary_churn['churn_rate'] = (salary_churn['sum'] / salary_churn['count'] * 100)
        weak_features_info.append({
            'name': 'Estimated Salary',
            'type': 'continuous',
            'data': salary_churn
        })
    
    # Has Credit Card (binary)
    if 'hascrcard' in raw_df.columns:
        card_churn = raw_df.groupby('hascrcard')['exited'].agg(['sum', 'count'])
        card_churn['churn_rate'] = (card_churn['sum'] / card_churn['count'] * 100)
        card_churn.index = ['No', 'Yes']
        weak_features_info.append({
            'name': 'Has Credit Card',
            'type': 'binary',
            'data': card_churn
        })
    
    # Create small multiples plot
    n_features = len(weak_features_info)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Common reference line for all subplots (baseline churn rate)
    baseline_churn = raw_df['exited'].mean() * 100
    
    for idx, feature_info in enumerate(weak_features_info):
        ax = axes[idx]
        data = feature_info['data']
        
        # Create bar plot
        bars = ax.bar(range(len(data)), data['churn_rate'], 
                     color='#95a5a6', alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add baseline reference line
        ax.axhline(y=baseline_churn, color='red', linestyle='--', linewidth=2, 
                  alpha=0.5, label=f'Baseline ({baseline_churn:.1f}%)')
        
        # Label bars with values
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
        
        # Formatting
        ax.set_title(feature_info['name'], fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel('Churn Rate (%)', fontsize=10, fontweight='bold')
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data.index, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, max(data['churn_rate'].max(), baseline_churn) * 1.15)
        
        # Only show legend on first subplot
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')
        
        # Ensure all spines are visible and black
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
    
    # Hide any unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Weak Features Show Minimal Variation in Churn Rates', 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_DIR / 'weak_features_comparison.png')
    plt.close()
    print("✓ Saved: weak_features_comparison.png")
    
    print(f"\n✅ Generated {9} EDA plots in {OUTPUT_DIR}/")
    
    # Generate PoC versions of alternative plot types
    print("\n📊 Generating PoC plots for weak features visualization...")
    generate_weak_features_poc_plots(raw_df)
    
    print(f"\n✅ Generated PoC plots in {OUTPUT_DIR}/")
    
    # Generate PoC versions for active member impact plots
    print("\n📊 Generating PoC plots for active member impact visualization...")
    generate_active_member_poc_plots(df)
    
    print(f"\n✅ Generated active member PoC plots in {OUTPUT_DIR}/")


def generate_active_member_poc_plots(df):
    """Generate PoC versions of different plot types for active member impact"""
    
    # Create PoC subfolder
    poc_dir = OUTPUT_DIR / 'active_member_pocs'
    poc_dir.mkdir(exist_ok=True)
    
    # Calculate churn rates
    active_churn = df.groupby('isactivemember')['exited'].agg(['sum', 'count'])
    active_churn['churn_rate'] = (active_churn['sum'] / active_churn['count'] * 100)
    
    # Get values using .values or explicit keys
    churn_rates = active_churn['churn_rate'].values
    active_rate = churn_rates[1] if len(churn_rates) > 1 else churn_rates[0]  # Active members (index 1)
    inactive_rate = churn_rates[0]  # Inactive members (index 0)
    risk_ratio = inactive_rate / active_rate
    
    print(f"  📊 Active: {active_rate:.1f}%, Inactive: {inactive_rate:.1f}%, Ratio: {risk_ratio:.2f}×")
    
    # PoC 1: Dumbbell Plot
    print("  📊 Type 1: Dumbbell Plot")
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot([active_rate, inactive_rate], [1, 1], 'k-', lw=3, alpha=0.5)
    ax.scatter([active_rate, inactive_rate], [1, 1], s=500, 
              color=['#55A868', '#C44E52'], edgecolors='black', linewidth=2, zorder=10)
    
    ax.text((active_rate + inactive_rate)/2, 1.25, f'{risk_ratio:.2f}× higher churn risk',
           ha='center', color='#C44E52', fontsize=14, fontweight='bold')
    
    ax.text(active_rate, 0.65, 'Active', ha='center', fontsize=12, fontweight='bold')
    ax.text(inactive_rate, 0.65, 'Inactive', ha='center', fontsize=12, fontweight='bold')
    ax.text(active_rate, 1.45, f'{active_rate:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax.text(inactive_rate, 1.45, f'{inactive_rate:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Churn Rate (%)', fontsize=13, fontweight='bold')
    ax.set_xlim(0, max(inactive_rate, active_rate) * 1.3)
    ax.set_ylim(0, 1.7)
    ax.set_yticks([])
    ax.set_title('Relative Churn Risk: Active vs Inactive Members', 
                fontsize=15, fontweight='bold', pad=15)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig(poc_dir / '01_dumbbell_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved: 01_dumbbell_plot.png")
    
    # PoC 2: Risk Ratio Meter (Gauge)
    print("  📊 Type 2: Risk Ratio Meter")
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Create gauge
    x_range = np.linspace(1.0, 2.0, 100)
    y_val = 0.5
    
    ax.fill_between(x_range, y_val-0.2, y_val+0.2, 
                   where=(x_range <= risk_ratio), 
                   color='#C44E52', alpha=0.3)
    ax.fill_between(x_range, y_val-0.2, y_val+0.2, 
                   where=(x_range > risk_ratio), 
                   color='lightgray', alpha=0.3)
    
    ax.axvline(x=risk_ratio, color='#C44E52', linewidth=4, linestyle='-')
    ax.scatter([risk_ratio], [y_val], s=400, color='#C44E52', 
              edgecolors='black', linewidth=2, zorder=10)
    
    ax.text(risk_ratio, y_val+0.4, f'{risk_ratio:.2f}×', 
           ha='center', fontsize=18, fontweight='bold', color='#C44E52')
    ax.text(1.0, y_val-0.45, '1.0×\n(Active)', ha='center', fontsize=11, fontweight='bold')
    ax.text(2.0, y_val-0.45, '2.0×\n(Inactive)', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_xlim(0.9, 2.1)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.set_xlabel('Relative Churn Risk', fontsize=13, fontweight='bold')
    ax.set_title('Churn Risk Ratio: Active vs Inactive Members', 
                fontsize=15, fontweight='bold', pad=15)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig(poc_dir / '02_risk_ratio_meter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved: 02_risk_ratio_meter.png")
    
    # PoC 3: Slope Graph
    print("  📊 Type 3: Slope Graph")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot([0, 1], [active_rate, inactive_rate], color='#C44E52', linewidth=4, marker='o', 
           markersize=12, markeredgecolor='black', markeredgewidth=2)
    
    ax.text(0, active_rate-3, 'Active', ha='center', fontsize=12, fontweight='bold')
    ax.text(1, inactive_rate+3, 'Inactive', ha='center', fontsize=12, fontweight='bold')
    ax.text(0, active_rate+5, f'{active_rate:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax.text(1, inactive_rate+5, f'{inactive_rate:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    ax.text(0.5, (active_rate + inactive_rate)/2 + 5, 
           f'Churn rises {inactive_rate-active_rate:.1f}%\n({risk_ratio:.2f}× increase)',
           ha='center', fontsize=12, fontweight='bold', color='#C44E52',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Active', 'Inactive'], fontsize=12, fontweight='bold')
    ax.set_ylabel('Churn Rate (%)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(inactive_rate, active_rate) * 1.25)
    ax.set_title('Slope Graph: Churn Rate Change by Activity Status', 
                fontsize=15, fontweight='bold', pad=15)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig(poc_dir / '03_slope_graph.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved: 03_slope_graph.png")
    
    # PoC 4: Waffle Plot
    print("  📊 Type 4: Waffle Plot")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    for idx, (status, rate) in enumerate([('Active', active_rate), ('Inactive', inactive_rate)]):
        ax = axes[idx]
        
        # Create 10x10 grid
        grid = np.zeros((10, 10))
        churned_squares = int(round(rate))
        grid.flat[:churned_squares] = 1
        
        colors_map = {0: '#55A868', 1: '#C44E52'}
        for i in range(10):
            for j in range(10):
                rect = plt.Rectangle((j, 9-i), 1, 1, 
                                   facecolor=colors_map[grid[i, j]], 
                                   edgecolor='white', linewidth=1.5)
                ax.add_patch(rect)
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{status}\n{rate:.1f}% Churn Rate', 
                    fontsize=14, fontweight='bold', pad=15)
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
    
    plt.suptitle('Waffle Plot: Churn Distribution (Each Square = 1%)', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(poc_dir / '04_waffle_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved: 04_waffle_plot.png")
    
    # PoC 5: Pyramid/Mirror Bar Plot
    print("  📊 Type 5: Pyramid Mirror Bar Plot")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get totals
    counts = active_churn['count'].values
    sums = active_churn['sum'].values
    
    # Inactive (left side)
    inactive_total = counts[0]
    inactive_churned = sums[0]
    inactive_retained = inactive_total - inactive_churned
    
    # Active (right side)
    active_total = counts[1] if len(counts) > 1 else counts[0]
    active_churned = sums[1] if len(sums) > 1 else sums[0]
    active_retained = active_total - active_churned
    
    # Left side (inactive)
    ax.barh([0.5], [inactive_churned], left=[0], height=0.3, 
           color='#C44E52', edgecolor='black', linewidth=1.5)
    ax.barh([0.5], [inactive_retained], left=[inactive_churned], height=0.3,
           color='#55A868', edgecolor='black', linewidth=1.5)
    
    # Right side (active) - mirrored
    ax.barh([0.5], [active_churned], left=[0], height=0.3,
           color='#C44E52', edgecolor='black', linewidth=1.5)
    ax.barh([0.5], [active_retained], left=[active_churned], height=0.3,
           color='#55A868', edgecolor='black', linewidth=1.5)
    
    # Labels
    ax.text(-inactive_total/2, 0.5, 'Inactive', ha='center', va='center',
           fontsize=13, fontweight='bold')
    ax.text(inactive_total/2, 0.5, 'Active', ha='center', va='center',
           fontsize=13, fontweight='bold')
    
    ax.text(0, 0.85, f'{risk_ratio:.2f}× difference', ha='center', va='center',
           fontsize=14, fontweight='bold', color='#C44E52',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
    
    ax.set_xlim(-inactive_total*1.1, active_total*1.1)
    ax.set_ylim(0, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Mirror Bar Plot: Distribution Comparison', 
                fontsize=15, fontweight='bold', pad=15)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig(poc_dir / '05_pyramid_mirror.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved: 05_pyramid_mirror.png")
    
    # PoC 6: Violin Plot (for comparison)
    print("  📊 Type 6: Violin Plot")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    from scipy.stats import gaussian_kde
    
    # Simulate churn probabilities (for demo purposes)
    np.random.seed(42)
    active_probs = np.random.beta(2, 10, 5000) * 20  # Low churn
    inactive_probs = np.random.beta(4, 10, 5000) * 35  # Higher churn
    
    parts = ax.violinplot([active_probs, inactive_probs], positions=[0, 1],
                         showmeans=True, showmedians=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor('#95a5a6')
        pc.set_alpha(0.6)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Active', 'Inactive'], fontsize=12, fontweight='bold')
    ax.set_ylabel('Churn Probability Distribution', fontsize=13, fontweight='bold')
    ax.set_title('Violin Plot: Risk Separation', fontsize=15, fontweight='bold', pad=15)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig(poc_dir / '06_violin_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved: 06_violin_plot.png")
    
    # PoC 7: Hybrid Panel (Count + Risk Ratio Gauge)
    print("  📊 Type 7: Hybrid Panel")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left panel: Counts
    ax1 = axes[0]
    counts = [active_total, inactive_total]
    bars = ax1.bar(['Active', 'Inactive'], counts, color=['#55A868', '#C44E52'], 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
    ax1.set_title('Customer Distribution', fontsize=14, fontweight='bold', pad=15)
    
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
    
    # Right panel: Risk ratio gauge
    ax2 = axes[1]
    x_range = np.linspace(1.0, 2.0, 100)
    y_val = 0.5
    
    ax2.fill_between(x_range, y_val-0.2, y_val+0.2, 
                    where=(x_range <= risk_ratio), 
                    color='#C44E52', alpha=0.3)
    ax2.fill_between(x_range, y_val-0.2, y_val+0.2, 
                    where=(x_range > risk_ratio), 
                    color='lightgray', alpha=0.3)
    
    ax2.axvline(x=risk_ratio, color='#C44E52', linewidth=4)
    ax2.scatter([risk_ratio], [y_val], s=400, color='#C44E52', 
               edgecolors='black', linewidth=2, zorder=10)
    
    ax2.text(risk_ratio, y_val+0.4, f'{risk_ratio:.2f}×', 
            ha='center', fontsize=20, fontweight='bold', color='#C44E52')
    ax2.text(1.0, y_val-0.45, '1.0×', ha='center', fontsize=11, fontweight='bold')
    ax2.text(2.0, y_val-0.45, '2.0×', ha='center', fontsize=11, fontweight='bold')
    
    ax2.set_xlim(0.9, 2.1)
    ax2.set_ylim(0, 1.2)
    ax2.set_yticks([])
    ax2.set_xlabel('Relative Churn Risk', fontsize=12, fontweight='bold')
    ax2.set_title('Risk Ratio', fontsize=14, fontweight='bold', pad=15)
    
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
    
    plt.suptitle('Hybrid Panel: Distribution + Risk Ratio', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(poc_dir / '07_hybrid_panel.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved: 07_hybrid_panel.png")
    
    # PoC 8: Churn Rate + Violin Plot Hybrid
    print("  📊 Type 8: Churn Rate + Violin Plot Hybrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left panel: Churn rate bar chart
    ax1 = axes[0]
    churn_rates_plot = [active_rate, inactive_rate]
    # Updated color palette: tab:blue and tab:orange
    bars = ax1.bar(['Active', 'Inactive'], churn_rates_plot, 
                   color=['#1f77b4', '#ff7f0e'], alpha=0.6, 
                   edgecolor='black', linewidth=1.5)
    
    for bar, rate in zip(bars, churn_rates_plot):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', 
                fontsize=13, fontweight='bold')
    
    # Add annotation centered above inactive bar (no arrow, reduced spacing)
    ax1.text(1, inactive_rate + 3, f'{risk_ratio:.2f}× higher churn risk',
             ha='center', fontsize=13, fontweight='bold', color='#ff7f0e')
    
    ax1.set_ylabel('Churn Rate (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Churn Rate by Activity Status', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylim(0, max(churn_rates_plot) * 1.25)
    ax1.set_xticklabels(['Active', 'Inactive'], fontweight='bold')
    
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
    
    # Right panel: Violin plot
    ax2 = axes[1]
    
    from scipy.stats import gaussian_kde
    
    # Simulate churn probabilities (for demo purposes)
    np.random.seed(42)
    active_probs = np.random.beta(2, 10, 5000) * 20  # Low churn
    inactive_probs = np.random.beta(4, 10, 5000) * 35  # Higher churn
    
    parts = ax2.violinplot([active_probs, inactive_probs], positions=[0, 1],
                         showmeans=True, showmedians=True)
    
    # Use matching colors from bar chart
    violin_colors = ['#1f77b4', '#ff7f0e']  # Blue for active, orange for inactive
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(violin_colors[i])
        pc.set_alpha(0.6)
    
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Active', 'Inactive'], fontsize=12, fontweight='bold')
    ax2.set_ylabel('Churn Probability Distribution', fontsize=13, fontweight='bold')
    ax2.yaxis.tick_right()  # Move y-axis to the right
    ax2.yaxis.set_label_position('right')  # Move y-axis label to the right
    ax2.set_title('Risk Distribution Comparison', fontsize=14, fontweight='bold', pad=15)
    
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
    
    plt.suptitle('Activity Status Impact: Churn Rate and Risk Distribution', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(poc_dir / '08_churn_rate_violin_hybrid.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved: 08_churn_rate_violin_hybrid.png")
    
    # PoC 8b: Horizontal version (swapped axes)
    print("  📊 Type 8b: Churn Rate + Violin Plot Hybrid (Horizontal)")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left panel: Churn rate bar chart (horizontal)
    ax1 = axes[0]
    churn_rates_plot = [active_rate, inactive_rate]
    bars = ax1.barh(['Active', 'Inactive'], churn_rates_plot, 
                    color=['#1f77b4', '#ff7f0e'], alpha=0.6, 
                    edgecolor='black', linewidth=1.5)
    
    for bar, rate in zip(bars, churn_rates_plot):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2.,
                f'{rate:.1f}%', ha='left', va='center', 
                fontsize=13, fontweight='bold')
    
    ax1.set_xlabel('Churn Rate (%)', fontsize=13, fontweight='normal')
    ax1.set_title('Churn Rate by Activity Status', fontsize=14, fontweight='normal', pad=15)
    ax1.set_xlim(0, max(churn_rates_plot) * 1.3)
    ax1.set_yticklabels(['Active', 'Inactive'], fontsize=14, fontweight='bold')
    
    # Remove top and right spines, keep left and bottom
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(True)
    ax1.spines['bottom'].set_visible(True)
    ax1.spines['left'].set_color('black')
    ax1.spines['bottom'].set_color('black')
    
    # Right panel: Violin plot (horizontal)
    ax2 = axes[1]
    
    from scipy.stats import gaussian_kde
    
    # Simulate churn probabilities (for demo purposes)
    np.random.seed(42)
    active_probs = np.random.beta(2, 10, 5000) * 20  # Low churn
    inactive_probs = np.random.beta(4, 10, 5000) * 35  # Higher churn
    
    # Create horizontal violin plot
    parts = ax2.violinplot([active_probs, inactive_probs], positions=[0, 1],
                         showmeans=True, showmedians=True, vert=False)
    
    # Use matching colors from bar chart
    violin_colors = ['#1f77b4', '#ff7f0e']  # Blue for active, orange for inactive
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(violin_colors[i])
        pc.set_alpha(0.6)
    
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels([])  # No y-axis labels since they align with left plot
    ax2.set_xlabel('Churn Probability Distribution', fontsize=13, fontweight='normal')
    ax2.set_title('Risk Distribution Comparison', fontsize=14, fontweight='normal', pad=15)
    
    # Remove top and right spines, keep left and bottom
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(True)
    ax2.spines['bottom'].set_visible(True)
    ax2.spines['left'].set_color('black')
    ax2.spines['bottom'].set_color('black')
    
    plt.suptitle('Activity Status Impact: Churn Rate and Risk Distribution', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(poc_dir / '08_churn_rate_violin_hybridv2.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved: 08_churn_rate_violin_hybridv2.png")


def generate_weak_features_poc_plots(raw_df):
    """Generate PoC versions of different plot types for weak features"""
    
    # Prepare data for CONTINUOUS features only (swap x/y axis approach)
    continuous_features = {}
    
    # Credit Score
    if 'creditscore' in raw_df.columns:
        continuous_features['Credit Score'] = raw_df[['creditscore', 'exited']].copy()
    
    # Satisfaction Score
    if 'satisfaction_score' in raw_df.columns:
        continuous_features['Satisfaction Score'] = raw_df[['satisfaction_score', 'exited']].copy()
    
    # Points Earned
    if 'point_earned' in raw_df.columns:
        continuous_features['Points Earned'] = raw_df[['point_earned', 'exited']].copy()
    
    # Estimated Salary
    if 'estimatedsalary' in raw_df.columns:
        continuous_features['Estimated Salary'] = raw_df[['estimatedsalary', 'exited']].copy()
    
    # Get feature names
    feature_names = list(continuous_features.keys())
    n_features = len(feature_names)
    
    # 1. VIOLIN PLOTS (swapped axes)
    print("  📊 Type 1: Violin plots")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, feat_name in enumerate(feature_names):
        ax = axes[idx]
        data = continuous_features[feat_name]
        
        # Get feature column name
        feat_col = 'creditscore' if feat_name == 'Credit Score' else \
                   'satisfaction_score' if feat_name == 'Satisfaction Score' else \
                   'point_earned' if feat_name == 'Points Earned' else 'estimatedsalary'
        
        # Create violin plot with swapped axes
        parts = ax.violinplot([data[data['exited'] == 0][feat_col].values,
                               data[data['exited'] == 1][feat_col].values],
                             positions=[0, 1],
                             showmeans=True, showmedians=True)
        
        # Set colors
        for pc in parts['bodies']:
            pc.set_facecolor('#95a5a6')
            pc.set_alpha(0.7)
        
        # Formatting
        ax.set_title(feat_name, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Churn Status', fontsize=10, fontweight='bold')
        ax.set_ylabel(feat_name, fontsize=10, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Retained', 'Churned'], fontsize=9)
        
        # Ensure all spines are visible and black
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
    
    # Hide unused
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Violin Plots: Distribution of Feature Values by Churn Status', 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_DIR / 'poc_violin_plots.png')
    plt.close()
    print("    ✓ Saved: poc_violin_plots.png")
    
    # 2. BOX PLOTS (swapped axes)
    print("  📊 Type 2: Box plots")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, feat_name in enumerate(feature_names):
        ax = axes[idx]
        data = continuous_features[feat_name]
        
        # Get feature column name
        feat_col = 'creditscore' if feat_name == 'Credit Score' else \
                   'satisfaction_score' if feat_name == 'Satisfaction Score' else \
                   'point_earned' if feat_name == 'Points Earned' else 'estimatedsalary'
        
        # Create box plot with swapped axes
        box_data = [data[data['exited'] == 0][feat_col].values,
                    data[data['exited'] == 1][feat_col].values]
        bp = ax.boxplot(box_data, patch_artist=True, labels=['Retained', 'Churned'])
        
        # Set colors
        for patch in bp['boxes']:
            patch.set_facecolor('#95a5a6')
            patch.set_alpha(0.7)
        
        # Formatting
        ax.set_title(feat_name, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Churn Status', fontsize=10, fontweight='bold')
        ax.set_ylabel(feat_name, fontsize=10, fontweight='bold')
        ax.set_xticklabels(['Retained', 'Churned'], fontsize=9)
        
        # Ensure all spines are visible and black
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
    
    # Hide unused
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Box Plots: Distribution of Feature Values by Churn Status', 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_DIR / 'poc_box_plots.png')
    plt.close()
    print("    ✓ Saved: poc_box_plots.png")
    
    # 3. STRIP PLOTS WITH JITTER (swapped axes)
    print("  📊 Type 3: Strip plots with jitter")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, feat_name in enumerate(feature_names):
        ax = axes[idx]
        data = continuous_features[feat_name]
        
        # Get feature column name
        feat_col = 'creditscore' if feat_name == 'Credit Score' else \
                   'satisfaction_score' if feat_name == 'Satisfaction Score' else \
                   'point_earned' if feat_name == 'Points Earned' else 'estimatedsalary'
        
        # Create strip plot with jitter and swapped axes
        # Retained customers
        retained_data = data[data['exited'] == 0][feat_col].values
        jitter_retained = np.random.normal(0, 0.05, len(retained_data))
        ax.scatter(0 + jitter_retained, retained_data, alpha=0.3, s=20, 
                  color='#55A868', edgecolors='black', linewidths=0.5, label='Retained')
        
        # Churned customers
        churned_data = data[data['exited'] == 1][feat_col].values
        jitter_churned = np.random.normal(0, 0.05, len(churned_data))
        ax.scatter(1 + jitter_churned, churned_data, alpha=0.3, s=20, 
                  color='#C44E52', edgecolors='black', linewidths=0.5, label='Churned')
        
        # Formatting
        ax.set_title(feat_name, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Churn Status', fontsize=10, fontweight='bold')
        ax.set_ylabel(feat_name, fontsize=10, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Retained', 'Churned'], fontsize=9)
        
        # Ensure all spines are visible and black
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
    
    # Hide unused
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Strip Plots with Jitter: Individual Feature Values by Churn Status', 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_DIR / 'poc_strip_plots.png')
    plt.close()
    print("    ✓ Saved: poc_strip_plots.png")
    
    # 4. PARTIAL DEPENDENCE PLOTS (PDPs) - Skipped due to model complexity
    print("  📊 Type 4: Partial Dependence Plots")
    print("    ⚠️  Skipping PDP plots (would require model refitting with specific features)")
    
    # 5. LOGISTIC REGRESSION ODDS RATIO PLOT
    print("  📊 Type 5: Logistic Regression Odds Ratio Plot")
    from sklearn.linear_model import LogisticRegression
    from scipy import stats
    
    # Prepare data for logistic regression - reload with correct column names
    logreg_raw = pd.read_csv('data/Customer-Churn-Records.csv')
    logreg_raw.columns = logreg_raw.columns.str.lower().str.replace(' ', '_')
    logreg_df = logreg_raw[['age', 'creditscore', 'satisfaction_score', 'point_earned', 'estimatedsalary', 'exited']].copy()
    
    # Fit individual logistic regressions for each feature
    odds_ratios = []
    ci_lowers = []
    ci_uppers = []
    feature_names_logreg = []
    feature_colors = []  # Track colors for each feature
    
    # Include strong predictor (age) first for comparison
    features_to_analyze = ['age', 'creditscore', 'satisfaction_score', 'point_earned', 'estimatedsalary']
    
    for feat in features_to_analyze:
        if feat not in logreg_df.columns:
            continue
        
        # Standardize feature
        X_feat = (logreg_df[feat] - logreg_df[feat].mean()) / logreg_df[feat].std()
        y = logreg_df['exited']
        
        # Fit logistic regression
        lr = LogisticRegression()
        lr.fit(X_feat.values.reshape(-1, 1), y)
        
        # Get coefficients and odds ratio
        coef = lr.coef_[0][0]
        or_val = np.exp(coef)
        
        # Calculate confidence interval (using Wald method)
        # Standard error
        se = np.sqrt(np.var(X_feat) / len(X_feat))
        z_score = 1.96  # 95% CI
        
        ci_lower = np.exp(coef - z_score * se)
        ci_upper = np.exp(coef + z_score * se)
        
        odds_ratios.append(or_val)
        ci_lowers.append(ci_lower)
        ci_uppers.append(ci_upper)
        feature_names_logreg.append(feat)
        
        # Assign color based on feature type
        if feat == 'age':
            feature_colors.append('#C44E52')  # Red for strong predictor
        else:
            feature_colors.append('#95a5a6')  # Gray for weak predictors
    
    # Create forest plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    y_pos = np.arange(len(feature_names_logreg))
    
    # Calculate appropriate x-axis range based on all CI bounds
    all_ci_bounds = ci_lowers + ci_uppers
    x_min = min(all_ci_bounds) * 0.995  # Add small margin
    x_max = max(all_ci_bounds) * 1.005
    
    # Extend range to ensure it includes 1.0 if needed
    if x_min > 1.0:
        x_min = 0.98
    if x_max < 1.0:
        x_max = 1.02
    
    # Round to nice numbers
    x_min = round(x_min - 0.01, 2)
    x_max = round(x_max + 0.01, 2)
    
    # Plot confidence intervals with thicker lines
    for i, (or_val, ci_low, ci_upp, color) in enumerate(zip(odds_ratios, ci_lowers, ci_uppers, feature_colors)):
        ax.plot([ci_low, ci_upp], [i, i], color=color, linewidth=3.5, alpha=0.8)
    
    # Plot odds ratios with larger, more visible points
    for i, (or_val, color) in enumerate(zip(odds_ratios, feature_colors)):
        ax.scatter(or_val, y_pos[i], s=150, color=color, edgecolors='black', 
                  linewidth=2.5, zorder=10, marker='D')
    
    # Add reference line at OR = 1
    ax.axvline(x=1, color='#C44E52', linestyle='--', linewidth=2.5, alpha=0.8, 
              label='OR = 1 (No Effect)', zorder=5)
    
    # Formatting
    ax.set_yticks(y_pos)
    # Create label mapping
    label_map = {
        'age': 'Age (Strong Predictor)',
        'creditscore': 'Credit Score',
        'satisfaction_score': 'Satisfaction Score',
        'point_earned': 'Points Earned',
        'estimatedsalary': 'Estimated Salary'
    }
    ax.set_yticklabels([label_map.get(feat, feat) for feat in feature_names_logreg], 
                       fontsize=11, fontweight='bold')
    ax.set_xlabel('Odds Ratio (95% Confidence Interval)', fontsize=13, fontweight='bold')
    ax.set_title('Logistic Regression: Odds Ratios Comparing Strong vs Weak Predictors', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(x_min, x_max)
    
    # Add grid for readability
    ax.grid(axis='x', alpha=0.3, linestyle=':', linewidth=1)
    ax.set_axisbelow(True)
    
    # Legend with better positioning and color explanation
    from matplotlib.patches import Patch
    legend_elements = [
        ax.axvline(0, 0, 0, color='#C44E52', linestyle='--', linewidth=2.5, alpha=0.8, label='OR = 1 (No Effect)'),
        Patch(facecolor='#C44E52', edgecolor='black', label='Strong Predictor (Age)'),
        Patch(facecolor='#95a5a6', edgecolor='black', label='Weak Predictors')
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='upper right', framealpha=0.95)
    
    # Add text annotations with better formatting
    for i, (or_val, ci_low, ci_upp) in enumerate(zip(odds_ratios, ci_lowers, ci_uppers)):
        ax.text(x_max - (x_max - x_min) * 0.05, i, 
               f'{or_val:.3f}  [{ci_low:.3f}, {ci_upp:.3f}]', 
               va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='black'))
    
    # Invert y-axis so top feature is at top
    ax.invert_yaxis()
    
    # Ensure all spines are visible and black
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'poc_odds_ratio_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved: poc_odds_ratio_plot.png")
    
    # 6. KDE PLOTS FOR WEAK FEATURES
    print("  📊 Type 6: KDE plots for weak features")
    
    # Prepare data for KDE plots
    kde_raw = pd.read_csv('data/Customer-Churn-Records.csv')
    kde_raw.columns = kde_raw.columns.str.lower().str.replace(' ', '_')
    
    # Define features and labels
    kde_features = ['creditscore', 'satisfaction_score', 'point_earned', 'estimatedsalary']
    kde_labels = ['Credit Score', 'Satisfaction Score', 'Points Earned', 'Estimated Salary']
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (feat, label) in enumerate(zip(kde_features, kde_labels)):
        ax = axes[idx]
        
        # Split data by churn status
        retained_data = kde_raw[kde_raw['exited'] == 0][feat].values
        churned_data = kde_raw[kde_raw['exited'] == 1][feat].values
        
        # Create KDE plots
        from scipy.stats import gaussian_kde
        
        # Compute KDE for retained customers
        kde_retained = gaussian_kde(retained_data)
        kde_churned = gaussian_kde(churned_data)
        
        # Create x-axis range
        x_min = min(kde_raw[feat].min(), retained_data.min(), churned_data.min())
        x_max = max(kde_raw[feat].max(), retained_data.max(), churned_data.max())
        x_range = np.linspace(x_min, x_max, 300)
        
        # Evaluate KDE
        kde_retained_vals = kde_retained(x_range)
        kde_churned_vals = kde_churned(x_range)
        
        # Plot KDE curves
        ax.plot(x_range, kde_retained_vals, color='#55A868', linewidth=2.5, 
               label='Retained', alpha=0.8)
        ax.fill_between(x_range, kde_retained_vals, alpha=0.3, color='#55A868')
        
        ax.plot(x_range, kde_churned_vals, color='#C44E52', linewidth=2.5, 
               label='Churned', alpha=0.8)
        ax.fill_between(x_range, kde_churned_vals, alpha=0.3, color='#C44E52')
        
        # Formatting
        ax.set_xlabel(label, fontsize=13, fontweight='bold')
        
        # Only show y-axis on left column (indices 0 and 2)
        if idx in [0, 2]:
            ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])
        
        # Ensure all spines are visible and black
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.5)
    
    # Create a single centered legend below the title
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#55A868', edgecolor='black', label='Retained'),
        Patch(facecolor='#C44E52', edgecolor='black', label='Churned')
    ]
    
    # Add legend at figure level, centered below title
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, 
              fontsize=12, framealpha=0.95, bbox_to_anchor=(0.5, 0.94))
    
    plt.suptitle('Kernel Density Estimates: Overlapping Distributions Show Weak Predictors', 
                 fontsize=15, fontweight='bold', y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(OUTPUT_DIR / 'poc_kde_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved: poc_kde_plots.png")


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
    # Ensure all spines are visible and black
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
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
    # Ensure all spines are visible and black for both subplots
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
    
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
    
    # Remove grid lines
    ax.grid(False)
    
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
    # Ensure all spines are visible and black
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '17_roc_curve.png')
    plt.close()
    print("✓ Saved: 17_roc_curve.png")
    
    # Plot 4: Partial Dependence - Age
    print("\n📊 Generating: 18_pdp_age.png")
    
    # Use same logic as quick_pdp function for better formatting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create grid for continuous feature
    xraw = X_test['age'].to_numpy()
    qs = np.linspace(0.01, 0.99, 30)
    grid = np.quantile(xraw, qs)
    
    pdp_vals = []
    ice_rows = []
    for v in grid:
        tmp = X_test.copy()
        tmp['age'] = v
        proba = model.predict_proba(tmp)[:, 1]
        ice_rows.append(proba)
        pdp_vals.append(proba.mean())
    
    pdp_vals = np.array(pdp_vals)
    ice = np.array(ice_rows)
    
    # Center the PDP
    y_main = pdp_vals - pdp_vals[0]
    ice_center = ice - ice[0, :]
    ci = ice_center.std(axis=1)
    
    # Plot with confidence intervals
    ax.fill_between(grid, y_main - ci, y_main + ci, alpha=0.25, label="±1 std")
    ax.plot(grid, y_main, lw=2.5, label="Mean PDP")
    ax.axhline(0, color="black", ls="--", lw=1, alpha=0.6)
    
    # Formatting
    ax.set_title('PDP: Customer Age (Lifecycle Pattern)', fontsize=13, fontweight='bold', pad=25)
    ax.text(0.5, 1.02, "Change in churn probability vs baseline. Shaded = ±1 SD", 
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="dimgray", style='italic')
    ax.set_ylabel("Δ Predicted Probability (centered)", fontweight='bold', fontsize=11)
    ax.set_xlabel('Age (years)', fontweight='bold', fontsize=11)
    ax.legend(loc="best", frameon=True, fontsize=10)
    ax.grid(alpha=0.3)
    
    # Ensure all spines are visible and black
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
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
    # Ensure all spines are visible and black
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
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
        # Ensure all spines are visible and black
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / '20_shap_summary.png')
        plt.close()
        print("✓ Saved: 20_shap_summary.png")
        
        # Plot 7: SHAP Waterfall with Customer Details
        print("\n📊 Generating: 26_shap_waterfall.png")
        
        # Select customer to explain (customer index 0)
        customer_idx = 0
        
        # Get customer data and actual outcome
        customer_actual = y_test.iloc[customer_idx]
        customer_pred_proba = model.predict_proba(X_test_sample.iloc[[customer_idx]])[0, 1]
        
        # Get SHAP explainer and base value
        shap_explainer_path = Path('03_churn_prediction/shap_explainer.bz2')
        if shap_explainer_path.exists():
            import joblib
            explainer = joblib.load(shap_explainer_path)
        else:
            # Create explainer if it doesn't exist
            explainer = shap.TreeExplainer(model)
        
        # Extract base value
        if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1:
            base_value = explainer.expected_value[1]  # Churn class
        else:
            base_value = explainer.expected_value if not isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value[0]
        
        # Get top 5 features
        customer_shap = shap_values_churn[customer_idx]
        customer_features = X_test_sample.iloc[customer_idx]
        top_shap_idx = np.argsort(np.abs(customer_shap))[-5:][::-1]
        
        # Create figure with waterfall plot spanning full width
        fig = plt.figure(figsize=(20, 5))
        
        # Main title and subtitle
        fig.suptitle('SHAP Waterfall Plot - Individual Customer Explanation', 
                    fontsize=16, fontweight='bold', y=0.97)
        fig.text(0.5, 0.93, 'Shows how each feature contributed to the model\'s prediction for this specific customer.\n'
                            'Red bars increase churn risk, blue bars decrease risk. Starts at average (base) and ends at final prediction.',
                ha='center', va='top', fontsize=11, style='italic', color='dimgray')
        
        # Main subplot: Waterfall plot spanning full width
        ax1 = fig.add_subplot(111)
        plt.sca(ax1)  # Set current axes
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_churn[customer_idx],
                base_values=base_value,
                data=X_test_sample.iloc[customer_idx].values,
                feature_names=feature_cols
            ),
            show=False
        )
        ax1.set_title('', fontsize=12, fontweight='bold', pad=30)
        # Add more padding around the plot area
        # ax1.margins(x=0.15, y=0.25)
        # Reduce font sizes for axis labels
        ax1.tick_params(axis='y', labelsize=8)
        ax1.tick_params(axis='x', labelsize=8)
        # Ensure all spines are visible and black
        for spine in ax1.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.0)
        
        # # Customer details as compact text below the plot
        # churn_status = 'WILL CHURN' if customer_pred_proba > 0.5 else 'WILL STAY'
        # churn_color = '#C44E52' if customer_pred_proba > 0.5 else '#55A868'
        # details_text = (f"Customer #{customer_idx} | "
        #                f"Actual: {'Churned' if customer_actual == 1 else 'Retained'} | "
        #                f"Predicted Probability: {customer_pred_proba:.2%} | "
        #                f"Prediction: ")
        
        # fig.text(0.5, 0.02, details_text, ha='center', va='bottom', 
        #         fontsize=10, fontweight='bold', color='black', transform=fig.transFigure)
        # fig.text(0.79, 0.02, churn_status, ha='left', va='bottom', 
        #         fontsize=10, fontweight='bold', color=churn_color, transform=fig.transFigure)
        from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

        details = TextArea(
            f"Customer #{customer_idx} | Actual: {'Churned' if customer_actual else 'Retained'} "
            f"| Predicted Probability: {customer_pred_proba:.2%} | Prediction: ",
            textprops=dict(color='black', fontsize=10)#, fontweight='bold')
        )
        status = TextArea(
            'WILL CHURN' if customer_pred_proba > 0.5 else 'WILL STAY',
            textprops=dict(color=('#C44E52' if customer_pred_proba > 0.5 else '#55A868'),
                        fontsize=10, fontweight='bold')
        )

        hbox = HPacker(children=[details, status], align="center", pad=0, sep=4)  # sep controls spacing
        anch = AnchoredOffsetbox(loc='lower center', child=hbox, pad=0, frameon=False,
                                bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure,
                                borderpad=0)
        fig.add_artist(anch)
        ax1.set_xlabel('Predicted probability of churn (f(x))', fontweight='bold')
        ax1.set_ylabel('Feature (value)', fontweight='bold')

        # Adjust layout manually to add more spacing around content
        plt.subplots_adjust(left=0.04, right=0.96, top=0.82, bottom=0.14)
        plt.savefig(OUTPUT_DIR / '26_shap_waterfall.png', dpi=300, facecolor='white')
        plt.close()
        print("✓ Saved: 26_shap_waterfall.png")
        
    else:
        print("  ⚠️  SHAP data not found, skipping SHAP plot")
    
    print(f"\n✅ Generated 7 churn prediction plots in {OUTPUT_DIR}/")


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

