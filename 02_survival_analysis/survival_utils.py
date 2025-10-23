"""
Survival Analysis Utility Functions
====================================
Reusable functions for Kaplan-Meier curves, log-rank tests, and Cox PH modeling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

# Styling constants
RIGHT_PAD = 0.15
LEFT = 0.00
KM_COLORS = ["#3568D4", "#2CB386"]  # blue, green
CI_ALPHA = 0.18
HEADER_COLOR = "#6B7280"  # gray header


def _short(label, maxlen=22):
    """Truncate long labels with ellipsis"""
    return label if len(label) <= maxlen else label[:maxlen-1] + "…"


def _fmt_pct(x, decimals=0):
    """Format decimal as percentage"""
    return f"{x:.{decimals}%}"


def _fmt_p(p):
    """Format p-value with appropriate precision"""
    if p == 0 or p < 1e-6:
        return r"< 1×10⁻⁶"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.6f}"


def plot_survival_analysis_2groups(
    timevar, eventvar, group1_mask, group2_mask,
    group1_label, group2_label, title,
    kmf: KaplanMeierFitter,
    figsize=(16, 6),
    show_at_risk=False,
    show_log2p=True
):
    """
    Plot Kaplan-Meier survival curves for two groups with log-rank test statistics.
    
    Parameters
    ----------
    timevar : pd.Series
        Time variable (tenure/duration)
    eventvar : pd.Series
        Event variable (churn/exit as 0/1)
    group1_mask : pd.Series (boolean)
        Mask for group 1
    group2_mask : pd.Series (boolean)
        Mask for group 2
    group1_label : str
        Label for group 1
    group2_label : str
        Label for group 2
    title : str
        Plot title
    kmf : KaplanMeierFitter
        Kaplan-Meier fitter object
    figsize : tuple
        Figure size
    show_at_risk : bool
        Show at-risk counts table below plot
    show_log2p : bool
        Show -log2(p) in details table
        
    Returns
    -------
    LogRankTestResult
        Result object from log-rank test
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrows=1, ncols=2, width_ratios=[7, 6], wspace=0.1, figure=fig)
    ax_plot = fig.add_subplot(gs[0, 0])

    right = gs[0, 1]
    bbox = right.get_position(fig)
    gs_r = GridSpec(
        nrows=2, ncols=1, height_ratios=[0.95, 1.6],
        left=bbox.x0, right=bbox.x1, bottom=bbox.y0, top=bbox.y1, hspace=0.18, figure=fig
    )
    ax_summary = fig.add_subplot(gs_r[0, 0])
    ax_details = fig.add_subplot(gs_r[1, 0])

    # KM curves - store colors used
    kmf.fit(timevar[group1_mask], event_observed=eventvar[group1_mask], label=group1_label)
    kmf.plot(ax=ax_plot, ci_show=True, ci_alpha=CI_ALPHA, color=KM_COLORS[0])
    median1 = kmf.median_survival_time_
    n1 = int(group1_mask.sum())

    kmf.fit(timevar[group2_mask], event_observed=eventvar[group2_mask], label=group2_label)
    kmf.plot(ax=ax_plot, ci_show=True, ci_alpha=CI_ALPHA, color=KM_COLORS[1])
    median2 = kmf.median_survival_time_
    n2 = int(group2_mask.sum())

    total = n1 + n2
    p1 = n1 / total if total > 0 else 0
    p2 = 1 - p1

    ax_plot.set_xlabel("Tenure (years)", fontsize=12, fontweight="semibold")
    ax_plot.set_ylabel("Survival Probability", fontsize=12, fontweight="semibold")
    ax_plot.set_ylim(0, 1.0)
    ax_plot.yaxis.set_major_locator(MultipleLocator(0.1))
    ax_plot.grid(axis="y", linestyle="--", alpha=0.25)
    for s in ("top", "right"):
        ax_plot.spines[s].set_visible(False)

    ax_plot.legend(
        loc="upper right",
        frameon=True, framealpha=0.9,
        facecolor="white", edgecolor="#D1D5DB"
    )

    if show_at_risk:
        # Create custom styled at-risk table instead of using lifelines default
        km1 = KaplanMeierFitter(label=group1_label).fit(timevar[group1_mask], event_observed=eventvar[group1_mask])
        km2 = KaplanMeierFitter(label=group2_label).fit(timevar[group2_mask], event_observed=eventvar[group2_mask])
        
        # Get at-risk counts at even intervals to match plot x-axis
        max_time = int(max(timevar))
        # Create even intervals (e.g., 0, 2, 4, 6, 8, 10)
        interval = 2
        time_points = list(range(0, max_time + 1, interval))
        
        # Build at-risk data
        at_risk_data = []
        for km, label, color in [(km1, group1_label, KM_COLORS[0]), (km2, group2_label, KM_COLORS[1])]:
            row = [label]  # Don't truncate group labels
            for t in time_points:
                # Find closest time point in survival table
                time_diffs = np.abs(km.survival_function_.index.to_numpy() - t)
                closest_idx = km.survival_function_.index[np.argmin(time_diffs)]
                n_at_risk = int(km.event_table.loc[:closest_idx, 'at_risk'].iloc[-1]) if len(km.event_table) > 0 else 0
                row.append(str(n_at_risk))
            at_risk_data.append((row, color))
        
        # Create new axis for styled at-risk table
        # Position to align with plot width and leave minimal space for x-axis labels
        # Plot area is roughly 0.06 to 0.62 (width ~0.40) based on GridSpec [7,6] ratio
        at_risk_ax = fig.add_axes([0.06, -0.08, 0.40, 0.12])
        at_risk_ax.axis('off')
        
        # Add title and subtitle above the table
        at_risk_ax.text(0.0, 1.0, "At Risk Counts", fontsize=11, fontweight="bold", 
                        transform=at_risk_ax.transAxes, va='bottom')
        at_risk_ax.text(0.0, 0.88, "Number of customers still being observed (haven't churned yet)", 
                        fontsize=8, color='#6B7280', style='italic',
                        transform=at_risk_ax.transAxes, va='top')
        
        # Create styled table (positioned below title)
        header = ['Group'] + [f'{t} mo' for t in time_points]  # Add "mo" suffix
        table_data = [row for row, _ in at_risk_data]
        
        # Calculate column widths - narrower for count columns, wider for group names
        group_col_width = 0.35  # Wider for full group names (no truncation)
        count_col_width = (1.0 - group_col_width) / len(time_points)  # Distribute remaining width
        col_widths = [group_col_width] + [count_col_width] * len(time_points)
        
        at_risk_table = at_risk_ax.table(
            cellText=table_data,
            colLabels=header,
            colWidths=col_widths,
            colLoc='center', cellLoc='center',
            loc='center', bbox=[0, 0, 1, 0.68]  # Use 68% for table, 32% for title + subtitle
        )
        at_risk_table.auto_set_font_size(False)
        at_risk_table.set_fontsize(9)
        at_risk_table.scale(1, 1.8)
        
        # Style the table
        for (r, c), cell in at_risk_table.get_celld().items():
            if r == 0:  # Header row
                cell.set_text_props(weight="bold", color="white", fontsize=10)
                cell.set_facecolor(HEADER_COLOR)
                cell.set_edgecolor("#E5E7EB")
                # Left-align group column header
                if c == 0:
                    cell.set_text_props(weight="bold", color="white", fontsize=10, ha='left')
            else:  # Data rows
                cell.set_edgecolor("#E5E7EB")
                if c == 0:  # Group name column - color-code it and left-align
                    group_idx = r - 1
                    color = at_risk_data[group_idx][1]
                    cell.set_text_props(weight="bold", color=color, fontsize=9, ha='left')
                else:
                    cell.set_text_props(fontsize=9)

    res = logrank_test(
        timevar[group1_mask], timevar[group2_mask],
        event_observed_A=eventvar[group1_mask],
        event_observed_B=eventvar[group2_mask]
    )
    p_str = _fmt_p(res.p_value)
    decision = "Significant (α = 0.05)" if res.p_value < 0.05 else "Not Significant"

    # Summary table
    ax_summary.axis("off")
    summary_rows = [
        ["Test statistic", f"{res.test_statistic:.2f}"],
        ["df", f"{int(res.degrees_of_freedom)}"],
        ["p-value", p_str],
        ["Decision", decision],
    ]
    summary = ax_summary.table(
        cellText=summary_rows,
        colLabels=["Statistic", "Value"],
        colLoc="left", cellLoc="left",
        colWidths=[0.55, 0.45],
        loc="center", bbox=[LEFT, 0.12, 1 - LEFT - RIGHT_PAD, 0.78]
    )
    summary.auto_set_font_size(False)
    summary.set_fontsize(10)
    summary.scale(1, 1.4)

    for (r, c), cell in summary.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="white", fontsize=11)
            cell.set_facecolor(HEADER_COLOR)
        else:
            if c == 0:
                cell.set_text_props(weight="semibold")
            if summary_rows[r-1][0] == "Decision":
                cell.set_facecolor("#F0FDF4")
            cell.set_edgecolor("#E5E7EB")

    ax_summary.set_title("Log-Rank Summary", fontsize=13, fontweight="bold", pad=6, loc="left", x=0.02)

    # Details table with color-coded group rows
    ax_details.axis("off")
    detail_rows = [
        ["Method", "Log-Rank (2-sample)"],
        ["Distribution", "Chi-squared"],
        ["Groups compared", "2"],
        [f"n ({_short(group1_label)})", f"{n1:,} ({_fmt_pct(p1)})"],
        [f"n ({_short(group2_label)})", f"{n2:,} ({_fmt_pct(p2)})"],
        [f"Median survival ({_short(group1_label)})", f"{median1:.1f}" if np.isfinite(median1) else "NA"],
        [f"Median survival ({_short(group2_label)})", f"{median2:.1f}" if np.isfinite(median2) else "NA"],
    ]
    if show_log2p:
        log2p = -np.log2(res.p_value) if res.p_value > 0 else np.inf
        detail_rows.append(["−log₂(p)", f"{log2p:.2f}"])

    details = ax_details.table(
        cellText=detail_rows,
        colLabels=["Item", "Value"],
        colLoc="left", cellLoc="left",
        colWidths=[0.58, 0.42],
        loc="center", bbox=[LEFT, 0.02, 1 - LEFT - RIGHT_PAD, 0.92]
    )
    details.auto_set_font_size(False)
    details.set_fontsize(10)
    details.scale(1, 1.45)

    for (r, c), cell in details.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="white", fontsize=11)
            cell.set_facecolor(HEADER_COLOR)
        else:
            if c == 0:
                cell.set_text_props(weight="semibold")
            # Color-code group-specific rows
            if r == 4 or r == 6:  # Group 1 rows
                cell.set_text_props(color=KM_COLORS[0], weight="bold")
            elif r == 5 or r == 7:  # Group 2 rows
                cell.set_text_props(color=KM_COLORS[1], weight="bold")
            cell.set_edgecolor("#E5E7EB")

    ax_details.set_title("Details", fontsize=13, fontweight="bold", pad=6, loc="left", x=0.02)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.97)
    
    # Adjust bottom margin based on whether at-risk table is shown
    # Need extra space for at-risk table positioned below the plot
    bottom_margin = 0.18 if show_at_risk else 0.10
    fig.subplots_adjust(left=0.06, right=0.96, top=0.90, bottom=bottom_margin, wspace=0.24)
    plt.show()

    return res


def plot_survival_analysis_multigroup(
    timevar, eventvar, df, column_name,
    group_masks, group_labels, title,
    kmf: KaplanMeierFitter,
    figsize=(16, 6),
    show_at_risk=False,
    show_log2p=True
):
    """
    Plot Kaplan-Meier survival curves for multiple groups with multivariate log-rank test.
    
    Parameters
    ----------
    timevar : pd.Series
        Time variable (tenure/duration)
    eventvar : pd.Series
        Event variable (churn/exit as 0/1)
    df : pd.DataFrame
        Original dataframe (needed for multivariate test)
    column_name : str
        Column name for grouping variable in df
    group_masks : list of pd.Series (boolean)
        List of masks for each group
    group_labels : list of str
        List of labels for each group
    title : str
        Plot title
    kmf : KaplanMeierFitter
        Kaplan-Meier fitter object
    figsize : tuple
        Figure size
    show_at_risk : bool
        Show at-risk counts table below plot
    show_log2p : bool
        Show -log2(p) in details table
        
    Returns
    -------
    MultivariateTesting
        Result object from multivariate log-rank test
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrows=1, ncols=2, width_ratios=[7, 6], wspace=0.1, figure=fig)
    ax_plot = fig.add_subplot(gs[0, 0])

    right = gs[0, 1]
    bbox = right.get_position(fig)
    gs_r = GridSpec(
        nrows=2, ncols=1, height_ratios=[0.95, 1.6],
        left=bbox.x0, right=bbox.x1, bottom=bbox.y0, top=bbox.y1, hspace=0.18, figure=fig
    )
    ax_summary = fig.add_subplot(gs_r[0, 0])
    ax_details = fig.add_subplot(gs_r[1, 0])

    # Extended color palette
    colors = ["#3568D4", "#2CB386", "#E63946", "#F77F00", "#9B59B6", "#1ABC9C"]
    
    medians = []
    group_sizes = []
    
    # Plot KM curves and collect stats
    for i, (mask, label) in enumerate(zip(group_masks, group_labels)):
        color = colors[i % len(colors)]
        kmf.fit(timevar[mask], event_observed=eventvar[mask], label=label)
        kmf.plot(ax=ax_plot, ci_show=True, ci_alpha=CI_ALPHA, color=color)
        medians.append(kmf.median_survival_time_)
        group_sizes.append(int(mask.sum()))

    total = sum(group_sizes)
    proportions = [n / total if total > 0 else 0 for n in group_sizes]

    ax_plot.set_xlabel("Tenure (years)", fontsize=12, fontweight="semibold")
    ax_plot.set_ylabel("Survival Probability", fontsize=12, fontweight="semibold")
    ax_plot.set_ylim(0, 1.0)
    ax_plot.yaxis.set_major_locator(MultipleLocator(0.1))
    ax_plot.grid(axis="y", linestyle="--", alpha=0.25)
    for s in ("top", "right"):
        ax_plot.spines[s].set_visible(False)

    ax_plot.legend(
        loc="best",
        frameon=True, framealpha=0.9,
        facecolor="white", edgecolor="#D1D5DB"
    )

    # Add at-risk counts if requested
    if show_at_risk:
        # Create KM fitters for all groups
        km_fitters = []
        for mask, label in zip(group_masks, group_labels):
            km = KaplanMeierFitter(label=label).fit(timevar[mask], event_observed=eventvar[mask])
            km_fitters.append(km)
        
        # Get at-risk counts at even intervals to match plot x-axis
        max_time = int(max(timevar))
        # Create even intervals (e.g., 0, 2, 4, 6, 8, 10)
        interval = 2
        time_points = list(range(0, max_time + 1, interval))
        
        # Build at-risk data
        at_risk_data = []
        for i, (km, label) in enumerate(zip(km_fitters, group_labels)):
            row = [label]  # Don't truncate group labels
            for t in time_points:
                # Find closest time point in survival table
                time_diffs = np.abs(km.survival_function_.index.to_numpy() - t)
                closest_idx = km.survival_function_.index[np.argmin(time_diffs)]
                n_at_risk = int(km.event_table.loc[:closest_idx, 'at_risk'].iloc[-1]) if len(km.event_table) > 0 else 0
                row.append(str(n_at_risk))
            at_risk_data.append((row, colors[i % len(colors)]))
        
        # Create new axis for styled at-risk table (adjust height based on number of groups)
        # Position to align with plot width and leave minimal space for x-axis labels
        # Plot area is roughly 0.06 to 0.62 (width ~0.40) based on GridSpec [7,6] ratio
        table_height = min(0.10 + len(group_labels) * 0.018, 0.22)
        at_risk_ax = fig.add_axes([0.06, -0.09, 0.40, table_height])
        at_risk_ax.axis('off')
        
        # Add title and subtitle above the table
        at_risk_ax.text(0.0, 1.0, "At Risk Counts", fontsize=10, fontweight="bold", 
                        transform=at_risk_ax.transAxes, va='bottom')
        at_risk_ax.text(0.0, 0.88, "Number of customers still being observed (haven't churned yet)", 
                        fontsize=7, color='#6B7280', style='italic',
                        transform=at_risk_ax.transAxes, va='top')
        
        # Create styled table (positioned below title)
        header = ['Group'] + [f'{t} mo' for t in time_points]  # Add "mo" suffix
        table_data = [row for row, _ in at_risk_data]
        
        # Calculate column widths - narrower for count columns, wider for group names
        group_col_width = 0.35  # Wider for full group names (no truncation)
        count_col_width = (1.0 - group_col_width) / len(time_points)  # Distribute remaining width
        col_widths = [group_col_width] + [count_col_width] * len(time_points)
        
        at_risk_table = at_risk_ax.table(
            cellText=table_data,
            colLabels=header,
            colWidths=col_widths,
            colLoc='center', cellLoc='center',
            loc='center', bbox=[0, 0, 1, 0.68]  # Use 68% for table, 32% for title + subtitle
        )
        at_risk_table.auto_set_font_size(False)
        at_risk_table.set_fontsize(8)
        at_risk_table.scale(1, 1.6)
        
        # Style the table
        for (r, c), cell in at_risk_table.get_celld().items():
            if r == 0:  # Header row
                cell.set_text_props(weight="bold", color="white", fontsize=9)
                cell.set_facecolor(HEADER_COLOR)
                cell.set_edgecolor("#E5E7EB")
                # Left-align group column header
                if c == 0:
                    cell.set_text_props(weight="bold", color="white", fontsize=9, ha='left')
            else:  # Data rows
                cell.set_edgecolor("#E5E7EB")
                if c == 0:  # Group name column - color-code it and left-align
                    group_idx = r - 1
                    color = at_risk_data[group_idx][1]
                    cell.set_text_props(weight="bold", color=color, fontsize=8, ha='left')
                else:
                    cell.set_text_props(fontsize=8)

    # Multivariate log-rank test (requires original column name)
    res = multivariate_logrank_test(
        df['tenure'], 
        df[column_name], 
        df['exited'],  # Using 'exited' as the event column
        alpha=0.95
    )
    p_str = _fmt_p(res.p_value)
    decision = "Significant (α = 0.05)" if res.p_value < 0.05 else "Not Significant"

    # Summary table
    ax_summary.axis("off")
    summary_rows = [
        ["Test statistic", f"{res.test_statistic:.2f}"],
        ["df", f"{int(res.degrees_of_freedom)}"],
        ["p-value", p_str],
        ["Decision", decision],
    ]
    summary = ax_summary.table(
        cellText=summary_rows,
        colLabels=["Statistic", "Value"],
        colLoc="left", cellLoc="left",
        colWidths=[0.55, 0.45],
        loc="center", bbox=[LEFT, 0.12, 1 - LEFT - RIGHT_PAD, 0.78]
    )
    summary.auto_set_font_size(False)
    summary.set_fontsize(10)
    summary.scale(1, 1.4)

    for (r, c), cell in summary.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="white", fontsize=11)
            cell.set_facecolor(HEADER_COLOR)
        else:
            if c == 0:
                cell.set_text_props(weight="semibold")
            if summary_rows[r-1][0] == "Decision":
                cell.set_facecolor("#F0FDF4")
            cell.set_edgecolor("#E5E7EB")

    ax_summary.set_title("Multivariate Log-Rank Summary", fontsize=13, fontweight="bold", pad=6, loc="left", x=0.02)

    # Details table
    ax_details.axis("off")
    detail_rows = [
        ["Method", "Log-Rank (multi-group)"],
        ["Distribution", "Chi-squared"],
        ["Groups compared", str(len(group_labels))],
    ]
    
    # Add group sizes
    for i, (label, size, prop) in enumerate(zip(group_labels, group_sizes, proportions)):
        detail_rows.append([f"n ({_short(label)})", f"{size:,} ({_fmt_pct(prop)})"])
    
    # Add medians
    for i, (label, median) in enumerate(zip(group_labels, medians)):
        median_str = f"{median:.1f}" if np.isfinite(median) else "NA"
        detail_rows.append([f"Median ({_short(label)})", median_str])
    
    if show_log2p:
        log2p = -np.log2(res.p_value) if res.p_value > 0 else np.inf
        detail_rows.append(["−log₂(p)", f"{log2p:.2f}"])

    details = ax_details.table(
        cellText=detail_rows,
        colLabels=["Item", "Value"],
        colLoc="left", cellLoc="left",
        colWidths=[0.58, 0.42],
        loc="center", bbox=[LEFT, 0.02, 1 - LEFT - RIGHT_PAD, 0.92]
    )
    details.auto_set_font_size(False)
    details.set_fontsize(9)
    details.scale(1, 1.3)

    # Calculate which rows correspond to each group
    n_groups = len(group_labels)
    group_size_rows = list(range(4, 4 + n_groups))  # Rows with group sizes
    median_rows = list(range(4 + n_groups, 4 + 2 * n_groups))  # Rows with medians
    
    for (r, c), cell in details.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="white", fontsize=11)
            cell.set_facecolor(HEADER_COLOR)
        else:
            if c == 0:
                cell.set_text_props(weight="semibold")
            
            # Color-code group-specific rows to match plot colors
            if r in group_size_rows:
                group_idx = r - 4  # Which group this row represents
                if group_idx < len(colors):
                    cell.set_text_props(color=colors[group_idx], weight="bold")
            elif r in median_rows:
                group_idx = r - (4 + n_groups)  # Which group this row represents
                if group_idx < len(colors):
                    cell.set_text_props(color=colors[group_idx], weight="bold")
            
            cell.set_edgecolor("#E5E7EB")

    ax_details.set_title("Details", fontsize=13, fontweight="bold", pad=6, loc="left", x=0.02)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.97)
    
    # Adjust bottom margin based on whether at-risk table is shown
    # Need extra space for at-risk table positioned below the plot
    # Increase margin more for multigroup to accommodate more rows
    bottom_margin = 0.20 if show_at_risk else 0.10
    fig.subplots_adjust(left=0.06, right=0.96, top=0.90, bottom=bottom_margin, wspace=0.24)
    plt.show()

    return res


def prepare_survival_data(df, drop_cols=None):
    """
    Prepare data for Cox PH regression by creating dummy variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    drop_cols : list, optional
        Additional columns to drop (beyond identifiers)
        
    Returns
    -------
    pd.DataFrame
        Prepared dataframe with dummy variables
    """
    df = df.copy()
    
    # Default columns to drop
    if drop_cols is None:
        drop_cols = []
    
    # Identify categorical columns for dummy encoding
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove columns we don't want to encode
    categorical_cols = [c for c in categorical_cols if c not in drop_cols]
    
    # Create dummy variables
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    
    # Drop specified columns
    if drop_cols:
        df = df.drop([c for c in drop_cols if c in df.columns], axis=1)
    
    return df

