"""
Generate simplified survival curve plots for LaTeX report.
These plots show only the curves without embedded tables.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from lifelines import KaplanMeierFitter
import sys
import os
import seaborn as sns

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

# Add parent directory to path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02_survival_analysis'))
from survival_utils import KM_COLORS, CI_ALPHA

# Load data
data_path = '../data/Customer-Churn-Records.csv'
df = pd.read_csv(data_path)

# Create age_group feature
def create_age_group(age):
    if age <= 30:
        return '18-30'
    elif age <= 40:
        return '31-40'
    elif age <= 50:
        return '41-50'
    elif age <= 60:
        return '51-60'
    elif age <= 70:
        return '61-70'
    else:
        return '70+'

df['age_group'] = df['Age'].apply(create_age_group)

# Setup
timevar = df['Tenure']
eventvar = df['Exited']

# Colors for multi-group plots
MULTI_COLORS = ["#3568D4", "#2CB386", "#E63946", "#F77F00", "#9B59B6", "#1ABC9C"]

def plot_survival_curves_only(timevar, eventvar, group_masks, group_labels, title, filename, colors=None):
    """Generate survival curves plot without tables."""
    if colors is None:
        colors = MULTI_COLORS
    
    fig, ax = plt.subplots(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    
    medians = []
    group_sizes = []
    
    # Plot curves
    for i, (mask, label) in enumerate(zip(group_masks, group_labels)):
        color = colors[i % len(colors)]
        kmf.fit(timevar[mask], event_observed=eventvar[mask], label=label)
        kmf.plot(ax=ax, ci_show=True, ci_alpha=CI_ALPHA, color=color)
        medians.append(kmf.median_survival_time_)
        group_sizes.append(int(mask.sum()))
    
    ax.set_xlabel("Tenure (years)", fontsize=12, fontweight="semibold")
    ax.set_ylabel("Survival Probability", fontsize=12, fontweight="semibold")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    
    ax.legend(loc="best", frameon=True, framealpha=0.9, facecolor="white", edgecolor="#D1D5DB")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    
    plt.tight_layout()
    plt.savefig(f'img/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return stats for caption
    return {
        'medians': medians,
        'group_sizes': group_sizes,
        'labels': group_labels
    }

# Generate survival by age groups
age_groups = ['18-30', '31-40', '41-50', '51-60', '61-70', '70+']
age_masks = [df['age_group'] == group for group in age_groups]
age_stats = plot_survival_curves_only(
    timevar, eventvar, age_masks, age_groups,
    "Survival Curves by Age Groups",
    "10_survival_age_groups_plot.png"
)

# Generate survival by products
product_masks = [df['NumOfProducts'] == i for i in [1, 2, 3, 4]]
product_labels = ['1 Product', '2 Products', '3 Products', '4 Products']
product_stats = plot_survival_curves_only(
    timevar, eventvar, product_masks, product_labels,
    "Survival Curves by Number of Products",
    "11_survival_products_plot.png"
)

# Generate survival by active status
active_mask = df['IsActiveMember'] == 1
inactive_mask = df['IsActiveMember'] == 0
active_stats = plot_survival_curves_only(
    timevar, eventvar, [active_mask, inactive_mask], ['Active Member', 'Inactive Member'],
    "Survival Curves by Activity Status",
    "12_survival_active_status_plot.png",
    colors=KM_COLORS
)

# Generate overall survival curve
kmf = KaplanMeierFitter()
kmf.fit(timevar, event_observed=eventvar)
fig, ax = plt.subplots(figsize=(10, 6))
kmf.plot(ax=ax, ci_show=True, ci_alpha=CI_ALPHA, color="#3568D4")
ax.set_xlabel("Tenure (years)", fontsize=12, fontweight="semibold")
ax.set_ylabel("Survival Probability", fontsize=12, fontweight="semibold")
ax.set_ylim(0, 1.0)
ax.yaxis.set_major_locator(MultipleLocator(0.1))

# Ensure all spines are visible and black
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('black')

ax.set_title("Overall Survival Curve", fontsize=14, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig('img/09_overall_survival_curve_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated simplified survival curve plots!")
print("\nAge Groups Stats:")
for label, size, median in zip(age_stats['labels'], age_stats['group_sizes'], age_stats['medians']):
    print(f"  {label}: n={size}, median={median:.1f} months" if np.isfinite(median) else f"  {label}: n={size}, median=NA")

