# %% [markdown]
# # Bank Customer Churn - Exploratory Data Analysis

# %% [markdown]
# ---
#
# ## 📋 EDA Structure Overview
#
# This notebook conducts a comprehensive Exploratory Data Analysis (EDA) on bank customer churn data. The analysis is structured into the following major sections:
#
# ### **1. Data Loading & Initial Exploration**
# - Load dataset and examine basic statistics
# - Check data types and missing values
# - Drop irrelevant identifier columns (RowNumber, CustomerId, Surname)
#
# ### **2. Demographic Features Analysis**
# Analyze how customer demographics relate to churn:
# - **Gender**: Distribution and churn rate by gender (innate characteristic)
# - **Geography**: Churn patterns across countries (France, Spain, Germany)
# - **Age**: Age distribution and segmentation (age groups)
#
# ### **3. Customer Engagement & Activity Analysis**
# Examine behavioral indicators:
# - **IsActiveMember**: Active vs inactive member churn comparison (behavioral)
# - **Tenure**: Length of relationship with bank (relationship duration)
# - **Tenure Groups**: Binned tenure for simplified modeling
# - **Engagement Cross-Analysis**: Activity patterns across demographics
#
# ### **4. Product & Service Usage**
# Analyze product adoption patterns:
# - **NumOfProducts**: Number of bank products held (1, 2, 3, 4)
# - **HasCrCard**: Credit card ownership (product ownership indicator)
# - **Relationship to tenure**: How product count varies with customer longevity
# - **Interaction with demographics**: Product adoption by age, geography, etc.
#
# ### **5. Financial Metrics Analysis**
# Explore monetary relationships:
# - **Balance**: Account balance distribution by churn status
# - **EstimatedSalary**: Salary distribution and churn correlation
# - **CreditScore**: Credit score patterns among churners vs non-churners
# - **Comparative distributions**: KDE plots comparing churned vs retained customers
#
# ### **6. Customer Experience Indicators**
# Investigate satisfaction and complaints:
# - **Satisfaction Score**: Rating distribution (1-5 scale)
# - **Complain**: Complaint history and its strong relationship to churn
# - **Card Type**: Premium card tiers (SILVER, GOLD, PLATINUM, DIAMOND)
# - **Point Earned**: Loyalty/rewards points analysis
#
# ### **7. Feature Interactions & Deep Dives**
# Explore interesting combinations:
# - **Geography × Credit Score**: Regional credit patterns
# - **Age × Balance**: Financial standing by age group
# - **Active Status × Products**: Engagement and product adoption
# - **Complaint × Satisfaction**: Experience quality indicators
#
# ### **8. Correlation Analysis**
# - Examine relationships between numerical features
# - Identify potential redundancies
# - Feature importance for modeling
#
# ### **9. Data Preparation Function**
# - Create preprocessing pipeline based on EDA insights
# - Handle encoding for categorical variables
# - Engineer features identified as important
# - Prepare data for machine learning models
#
# ---
#
# **Key Differences from Original Telco Dataset:**
# - This is **bank churn** (not telecom) - different industry dynamics
# - Focus on **financial metrics** (balance, credit score, salary) rather than service packages
# - **Product count** instead of individual service subscriptions
# - **Geography** as a major factor (international banking)
# - **Customer satisfaction & complaints** are explicitly tracked
# - **Loyalty program** (card type, points earned)
#
# Let's begin the detailed analysis! 👇

# %%
# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom EDA utility functions
from eda_utils import (
    stacked_plot,
    countplot_enhanced,
    density_plot_enhanced,
    kde_comparison_plot,
    print_churn_summary,
    correlation_heatmap,
    plot_demographic_grid,
    validate_data_consistency
)

# Set style
sns.set_style('darkgrid')
plt.rcParams['figure.dpi'] = 100

print("✓ Libraries and utility functions loaded successfully!")

# %% [markdown]
# ---
#
# ## 📂 Section 1: Data Loading & Initial Exploration
#
# Load the dataset and perform initial data inspection.

# %%
# Load the dataset
df = pd.read_csv("../data/Customer-Churn-Records.csv")
print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
df.head()

# %%
# Dataset shape and basic info
print("Dataset Shape:", df.shape)
print("\nColumn Names and Types:")
print(df.dtypes)

# %%
# Check for missing values
missing = df.isnull().sum()
print("Missing Values per Column:")
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("✓ No missing values!")

# %%
# Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")

# %% [markdown]
# ### Data Cleaning & Preprocessing
#
# We'll drop identifier columns that don't contribute to churn prediction and standardize column names.

# %%
# Drop identifier columns
print("Dropping identifier columns: RowNumber, CustomerId, Surname")
df.drop(['RowNumber', 'CustomerId', 'Surname'], inplace=True, axis=1)

# Convert binary columns to boolean
bin_cols = ['HasCrCard', 'IsActiveMember', 'Exited', 'Complain']
df[bin_cols] = df[bin_cols].astype(bool)

print(f"\n✓ Data cleaned. New shape: {df.shape}")

# %%
# Standardize column names: lowercase, underscores instead of spaces
df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
print("✓ Column names standardized")
print("\nFinal columns:", list(df.columns))

# %%
# Create age groups for analysis
df['age_group'] = pd.cut(
    df['age'],
    bins=[18, 30, 40, 50, 60, 70, 100],
    labels=['18–30', '31–40', '41–50', '51–60', '61–70', '70+']
)

print("✓ Age groups created")
print("\nAge Group Distribution:")
print(df['age_group'].value_counts().sort_index())

# %%
# Basic statistics summary
print("=" * 80)
print("DATASET SUMMARY")
print("=" * 80)
print(df.describe())
print("\n" + "=" * 80)

# %%
# Check target variable distribution
print("Target Variable (Exited) Distribution:")
print(df['exited'].value_counts())
print(f"\nChurn Rate: {df['exited'].sum() / len(df) * 100:.2f}%")

# %% [markdown]
# ---
#
# ## 📊 Section 2: Demographic Features Analysis
#
# ### Analysis Goals:
# In this section, we examine how **innate or relatively stable demographic characteristics** relate to customer churn:
#
# 1. **Gender**: Does gender play a role in churn? 
#    - Hypothesis: Minimal impact (similar to Telco dataset)
#    - Innate characteristic
#
# 2. **Geography (Country)**: Are there regional patterns in churn rates?
#    - France, Spain, and Germany may have different banking regulations, competition, or customer expectations
#    - Major difference from Telco dataset (no geographic segmentation)
#    - Relatively stable characteristic
#
# 3. **Age Distribution and Segmentation**: 
#    - Create age groups for better visualization and modeling
#    - Younger customers may have different needs than older customers
#    - Hypothesis: Older customers more stable/loyal
#    - Innate characteristic (changes slowly)
#
# **Note:** This section focuses on demographics ONLY. Product ownership (HasCrCard) and behavioral features (IsActiveMember) are analyzed in later sections.
#
# **Expected Insights:**
# - Geography likely to be significant (unlike gender in Telco)
# - Age may show strong patterns (older = more loyal)
# - Combined demographic effects
#
# ---

# %% [markdown]
# ### Demographic Overview Grid
#
# Create a 3-feature visualization showing churn rates across pure demographic features.

# %%
# Create individual demographic visualizations
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Churn Analysis by Demographic Features', fontsize=16, fontweight='bold', y=1.02)

# Gender
stacked_plot(df, 'gender', 'exited', ax=axes[0], title='Gender')

# Geography
stacked_plot(df, 'geography', 'exited', ax=axes[1], title='Geography', show_legend=True)

# Age Group
stacked_plot(df, 'age_group', 'exited', ax=axes[2], title='Age Group')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Detailed Demographic Analysis

# %%
# Gender analysis
print_churn_summary(df, 'gender', 'exited')

# %%
# Geography analysis
print_churn_summary(df, 'geography', 'exited')

# %%
# Age group analysis
print_churn_summary(df, 'age_group', 'exited', segment_label='Age Group')

# %%
# Age distribution by churn status
fig, ax = plt.subplots(figsize=(12, 6))
churned_temp = df[df['exited'] == True]
not_churned_temp = df[df['exited'] == False]

ax.hist([not_churned_temp['age'], churned_temp['age']], 
        bins=30, label=['Not Churned', 'Churned'],
        color=['#66BB6A', '#EF5350'], alpha=0.7, edgecolor='black')
ax.set_xlabel('Age', fontsize=12, fontweight='semibold')
ax.set_ylabel('Count', fontsize=12, fontweight='semibold')
ax.set_title('Age Distribution by Churn Status', fontsize=14, fontweight='bold')
ax.legend(title='Customer Status', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Insights from Demographic Analysis:**
#
# **Gender:** Modest effect (25% female vs 16% male churn = 1.5× difference). Weakest demographic predictor.
#
# **Geography:** Strong effect - **Germany at 32% churn is 2× higher than France/Spain** (~16%). Major market issue requiring investigation.
#
# **Age:** Strongest demographic predictor with clear lifecycle pattern:
# - Young (18-30): 7.5% churn - highly loyal
# - Mid-career (31-40): 12% churn - stable
# - **Mid-life (41-50): 34% churn - elevated risk**
# - **Pre-retirement (51-60): 56% churn - CRITICAL** (more than half churn, likely moving assets for retirement planning)
# - Senior (61-70): 31% churn - still elevated
# - Elderly (70+): 8% churn - return to loyalty
#
# **Key Takeaway:** Age (especially 51-60) and Geography (Germany) are the actionable demographic predictors. Focus retention efforts on German customers and those aged 40-65.

# %% [markdown]
# ---
#
# ## 🎯 Section 3: Customer Engagement & Activity Analysis
#
# ### Analysis Goals:
# This section focuses on **behavioral indicators** that show how engaged customers are with the bank:
#
# 1. **IsActiveMember Status**: 
#    - Compare active vs inactive members
#    - Hypothesis: Active members churn less (very strong indicator)
#    - **Behavioral feature** - reflects how customers use their account
#    - Similar to "service usage" patterns in Telco dataset
#
# 2. **Tenure Distribution**:
#    - How long have customers been with the bank?
#    - Hypothesis: Longer tenure = lower churn (loyalty builds over time)
#    - **Relationship duration** - not innate, but builds over time
#    - Direct parallel to Telco analysis
#
# 3. **Tenure Grouping**:
#    - Create tenure groups for simplified modeling
#    - Reduces complexity
#    - Makes patterns easier to visualize
#
# 4. **Engagement Cross-Analysis**:
#    - Activity patterns across age groups
#    - Activity patterns across geography
#
# **Key Questions:**
# - What percentage of inactive members churn?
# - Is there a "critical period" for new customers?
# - Do tenure patterns differ by age or geography?
# - Are younger customers less active?
#
# ---

# %% [markdown]
# ### IsActiveMember Analysis

# %%
# Active member churn analysis
stacked_plot(df, 'isactivemember', 'exited', 
             title='Churn Rate by Active Member Status', 
             show_legend=True)
plt.show()

print_churn_summary(df, 'isactivemember', 'exited', segment_label='Active Member Status')

# %% [markdown]
# ### Tenure Analysis

# %%
# Tenure distribution statistics
print("Tenure Statistics:")
print(df['tenure'].describe())
print("\nMost Common Tenure Values:")
print(df['tenure'].value_counts().head(10))

# %%
# Tenure distribution by churn status
countplot_enhanced(df, x='tenure', hue='exited',
                   title='Customer Tenure Distribution by Churn Status',
                   xlabel='Tenure (years)')
plt.show()

# %%
# Create tenure groups
def categorize_tenure(t):
    if t <= 2:
        return '0-2 years'
    elif t <= 4:
        return '3-4 years'
    elif t <= 6:
        return '5-6 years'
    elif t <= 8:
        return '7-8 years'
    else:
        return '9+ years'

df['tenure_group'] = df['tenure'].apply(categorize_tenure)

print("✓ Tenure groups created")
print("\nTenure Group Distribution:")
print(df['tenure_group'].value_counts())

# %%
# Tenure group churn analysis
stacked_plot(df, 'tenure_group', 'exited',
             title='Churn Rate by Tenure Group',
             show_legend=True)
plt.show()

print_churn_summary(df, 'tenure_group', 'exited', segment_label='Tenure Group')

# %% [markdown]
# ### Engagement Cross-Analysis

# %%
# Activity rate by age group
activity_by_age = df.groupby('age_group')['isactivemember'].agg(['sum', 'count'])
activity_by_age['active_rate'] = (activity_by_age['sum'] / activity_by_age['count'] * 100)
print("=" * 70)
print("ACTIVITY RATE BY AGE GROUP")
print("=" * 70)
print(activity_by_age[['active_rate']])
print("=" * 70)

# %%
# Activity rate by geography
activity_by_geo = df.groupby('geography')['isactivemember'].agg(['sum', 'count'])
activity_by_geo['active_rate'] = (activity_by_geo['sum'] / activity_by_geo['count'] * 100)
print("=" * 70)
print("ACTIVITY RATE BY GEOGRAPHY")
print("=" * 70)
print(activity_by_geo[['active_rate']])
print("=" * 70)

# %% [markdown]
# **Insights from Engagement Analysis:**
#
# **IsActiveMember:** Very strong predictor - **inactive members have 27% churn vs 14% for active** (1.9× difference). Nearly half of churn is driven by inactivity.
#
# **Tenure:** Surprisingly weak predictor - churn rates are flat across all tenure groups (18-21%), with only 1.2× difference between best and worst. Tenure does NOT indicate loyalty in this dataset.
#
# **Activity Patterns (Cross-Analysis):**
# - Activity rate INCREASES with age: Young adults ~50% active, seniors 70-90% active (counterintuitive - older customers more engaged!)
# - Geography has minimal effect on activity (~50-53% across France/Spain/Germany)
#
# **Key Takeaway:** IsActiveMember is a critical behavioral predictor (1.9× effect), but tenure provides little value. The finding that older customers are MORE active helps explain why they're less likely to churn (until the 51-60 retirement planning phase).
#
# ---

# %% [markdown]
# ---
#
# ## 🏦 Section 4: Product & Service Usage Analysis
#
# ### Analysis Goals:
# Unlike Telco's multiple individual services, bank customers have consolidated product metrics. This section explores:
#
# 1. **NumOfProducts**: Number of bank products held (1-4)
#    - How many products does each customer have?
#    - Do customers with more products churn less?
#
# 2. **HasCrCard**: Credit card ownership
#    - **Product ownership indicator** (not demographic)
#    - Do credit card holders churn less?
#    - Indicates product diversification
#
# 3. **Product Adoption Patterns**:
#    - By tenure: Do longer-tenured customers have more products?
#    - By demographics: Age and geographic differences
#    - By activity status: Do active members have more products?
#
# **Hypothesis:**
# - Customers with 2-3 products likely have optimal engagement (not too few, not overwhelmed)
# - Single-product customers at higher churn risk
# - Credit card holders more invested in bank relationship
#
# ---

# %% [markdown]
# ### Number of Products Distribution

# %%
# Product count distribution
print("Number of Products Distribution:")
print(df['numofproducts'].value_counts().sort_index())

# Churn rate by product count
stacked_plot(df, 'numofproducts', 'exited',
             title='Churn Rate by Number of Products',
             show_legend=True)
plt.show()

print_churn_summary(df, 'numofproducts', 'exited', segment_label='Number of Products')

# %%
# Product count by tenure
density_plot_enhanced(df, x_col='tenure', group_col='numofproducts',
                      title='Tenure Distribution by Number of Products')
plt.show()

# %%
# Product count by geography
countplot_enhanced(df, x='geography', hue='numofproducts',
                   title='Product Adoption by Geography',
                   figsize=(12, 6),
                   palette={1: "#E57373", 2: "#81C784", 3: "#64B5F6", 4: "#FFD54F"})
plt.show()

# %%
# Product count by age group
countplot_enhanced(df, x='age_group', hue='numofproducts',
                   title='Product Adoption by Age Group',
                   figsize=(14, 6),
                   palette={1: "#E57373", 2: "#81C784", 3: "#64B5F6", 4: "#FFD54F"})
plt.show()

# %% [markdown]
# ### Credit Card Ownership (HasCrCard)

# %%
# Credit card ownership analysis
stacked_plot(df, 'hascrcard', 'exited', 
             title='Churn Rate by Credit Card Ownership', 
             show_legend=True)
plt.show()

print_churn_summary(df, 'hascrcard', 'exited', segment_label='Credit Card Ownership')

# %%
# Credit card ownership by number of products
crosstab_card_products = pd.crosstab(df['numofproducts'], df['hascrcard'], normalize='index') * 100
print("=" * 70)
print("CREDIT CARD OWNERSHIP BY NUMBER OF PRODUCTS (%)")
print("=" * 70)
print(crosstab_card_products)
print("=" * 70)

# %% [markdown]
# **Insights from Product Usage Analysis:**
#
# **Number of Products:** EXTREMELY strong predictor with clear "Goldilocks zone":
# - 1 product: 28% churn (under-engaged)
# - **2 products: 7.6% churn - OPTIMAL!** (lowest churn, 46% of customers)
# - 3 products: 83% churn - DISASTER ZONE! (likely over-serviced/forced cross-sell)
# - 4 products: 100% churn - ALL customers leave (60 customers, all churned)
#
# **HasCrCard:** Essentially no effect (20.8% vs 20.2% churn = 1.0× difference). Not useful for prediction. ~70% have credit cards regardless of product count.
#
# **Product Adoption Patterns:** Most customers have 1-2 products across all demographics. No strong variation by geography, age, or tenure.
#
# **Key Takeaway:** Having exactly 2 products is the sweet spot (7.6% churn). **Critical issue: Customers with 3-4 products have catastrophic churn (83-100%)** - suggests aggressive cross-selling backfires. Review product bundling strategy immediately.
#
# ---

# %% [markdown]
# **Insights from Financial Metrics Analysis:**
#
# **Balance:** Strong bimodal pattern with **36% at $0 balance** vs ~64% at $100-150k. Churned customers show two peaks: one at $0 and another at ~$125k (mid-high balance). See Section 7 for churn rate analysis.
#
# **EstimatedSalary:** No predictive value - distributions for churned vs not churned are nearly identical (uniform ~$0-$200k). Income level does not affect churn.
#
# **CreditScore:** No predictive value - distributions for churned vs not churned almost perfectly overlap (centered at ~650). Credit worthiness is irrelevant to churn.
#
# **Key Takeaway:** Balance shows the only notable distributional difference among financial metrics (bimodal pattern), though salary and credit score provide no information. See zero-balance deep dive in Section 7 for churn implications.
#
# ---
#
# ## 💰 Section 5: Financial Metrics Analysis
#
# ### Analysis Goals:
# This section is **unique to bank churn** - no equivalent in Telco dataset. We analyze three key financial indicators:
#
# 1. **Account Balance**: Distribution and impact on churn
# 2. **Estimated Salary**: Income level correlation with churn
# 3. **Credit Score**: Credit worthiness patterns
#
# **Expected Insights:**
# - Zero-balance accounts likely high churn risk
# - Salary may have less impact than expected
# - Credit score could show bidirectional effect
#
# ---

# %% [markdown]
# ### Account Balance Analysis

# %%
# Prepare churned/not_churned groups for KDE plots
churned = df[df['exited'] == True]
not_churned = df[df['exited'] == False]

# %%
# Balance statistics
print("Account Balance Statistics:")
print(df['balance'].describe())

# Zero balance analysis
zero_balance = (df['balance'] == 0).sum()
print(f"\nCustomers with $0 balance: {zero_balance:,} ({zero_balance/len(df)*100:.1f}%)")

# %%
# Balance distribution by churn
kde_comparison_plot(churned, not_churned, 'balance',
                    title='Account Balance Distribution by Churn Status')
plt.show()

# %% [markdown]
# ### Estimated Salary Analysis

# %%
# Salary statistics
print("Estimated Salary Statistics:")
print(df['estimatedsalary'].describe())

# Salary distribution by churn
kde_comparison_plot(churned, not_churned, 'estimatedsalary',
                    title='Salary Distribution by Churn Status')
plt.show()

# %% [markdown]
# ### Credit Score Analysis

# %%
# Credit score statistics
print("Credit Score Statistics:")
print(df['creditscore'].describe())

# Credit score distribution by churn
kde_comparison_plot(churned, not_churned, 'creditscore',
                    title='Credit Score Distribution by Churn Status',
                    xlabel='Credit Score')
plt.show()

# %% [markdown]
# **Insights from Customer Experience Analysis:**
#
# **Complain:** THE STRONGEST PREDICTOR IN THE ENTIRE DATASET - **99.5% of complainers churn vs 0.05% of non-complainers** (1979× difference!). Filing a complaint virtually guarantees churn. 20% of customers (2,044) filed complaints.
#
# **Satisfaction Score:** No predictive value - churn rates flat across all scores (19.6% - 21.8%, only 1.1× difference). Surprisingly, satisfaction level doesn't matter.
#
# **Card Type:** No predictive value - all card tiers have similar churn (19-22%, only 1.1× difference). Premium card holders don't stay longer.
#
# **Points Earned:** No predictive value - distributions for churned vs not churned are nearly identical (uniform ~200-1000 points). Rewards program usage is irrelevant.
#
# **Key Takeaway:** Complaint status is BY FAR the most critical predictor discovered - it's a death sentence for retention (99.5% churn). All other experience metrics (satisfaction, card tier, points) are surprisingly useless. Focus: Prevent complaints from happening, and aggressively intervene if one is filed.
#
# ---

# %% [markdown]
# ---
#
# ## ⭐ Section 6: Customer Experience Indicators
#
# ### Analysis Goals:
# These features are **unique to this bank dataset** and provide direct feedback on customer satisfaction:
#
# 1. **Satisfaction Score** (1-5 scale): Rating distribution and churn patterns
# 2. **Complain** (Binary): Complaint history impact
# 3. **Card Type** (SILVER, GOLD, PLATINUM, DIAMOND): Premium tier analysis
# 4. **Point Earned** (Loyalty Program): Rewards effectiveness
#
# **Key Questions:**
# - Is "Complain" a stronger predictor than "Satisfaction Score"?
# - Do loyalty programs actually reduce churn?
# - Are there dissatisfied customers who haven't complained yet?
#
# ---

# %% [markdown]
# ### Satisfaction Score Analysis

# %%
# Satisfaction score distribution
print("Satisfaction Score Distribution:")
print(df['satisfaction_score'].value_counts().sort_index())

# Churn by satisfaction
stacked_plot(df, 'satisfaction_score', 'exited',
             title='Churn Rate by Satisfaction Score',
             show_legend=True)
plt.show()

print_churn_summary(df, 'satisfaction_score', 'exited', segment_label='Satisfaction Score')

# %% [markdown]
# ### Complain Analysis

# %%
# Complaint analysis
stacked_plot(df, 'complain', 'exited',
             title='Churn Rate by Complaint Status',
             show_legend=True)
plt.show()

print_churn_summary(df, 'complain', 'exited', segment_label='Complaint Filed')

# %%
# Cross-analysis: Complain × Satisfaction
print("=" * 80)
print("COMPLAINT VS SATISFACTION CROSS-ANALYSIS")
print("=" * 80)
crosstab = pd.crosstab(df['complain'], df['satisfaction_score'], margins=True)
print(crosstab)
print("=" * 80)

# %% [markdown]
# ### Card Type Analysis

# %%
# Card type distribution
print("Card Type Distribution:")
print(df['card_type'].value_counts())

# Churn by card type
stacked_plot(df, 'card_type', 'exited',
             title='Churn Rate by Card Type',
             show_legend=True)
plt.show()

print_churn_summary(df, 'card_type', 'exited', segment_label='Card Type')

# %% [markdown]
# ### Points Earned (Loyalty Program) Analysis

# %%
# Points statistics
print("Points Earned Statistics:")
print(df['point_earned'].describe())

# Points distribution by churn
kde_comparison_plot(churned, not_churned, 'point_earned',
                    title='Points Earned Distribution by Churn Status',
                    xlabel='Points Earned')
plt.show()

# %% [markdown]
# ---
#
# ## 🔄 Section 7: Feature Interactions & Deep Dives
#
# Explore interesting combinations of features that may reveal hidden patterns.

# %% [markdown]
# ### Geography × Credit Score

# %%
# Credit score by geography
fig, ax = plt.subplots(figsize=(10, 6))
df.boxplot(column='creditscore', by='geography', ax=ax)
ax.set_title('Credit Score Distribution by Geography', fontsize=14, fontweight='bold')
ax.set_xlabel('Geography', fontsize=12)
ax.set_ylabel('Credit Score', fontsize=12)
plt.suptitle('')  # Remove default title
plt.show()

# %% [markdown]
# ### Age × Balance

# %%
# Balance by age group
fig, ax = plt.subplots(figsize=(12, 6))
df.boxplot(column='balance', by='age_group', ax=ax)
ax.set_title('Account Balance by Age Group', fontsize=14, fontweight='bold')
ax.set_xlabel('Age Group', fontsize=12)
ax.set_ylabel('Balance ($)', fontsize=12)
plt.suptitle('')
plt.show()

# %% [markdown]
# ### Active Member × Products

# %%
# Product count by active status
countplot_enhanced(df, x='numofproducts', hue='isactivemember',
                   title='Product Count by Active Member Status',
                   figsize=(10, 6),
                   palette={False: "#EF5350", True: "#66BB6A"})
plt.show()

# %% [markdown]
# ### Zero Balance Deep Dive

# %%
# Zero balance analysis
zero_bal_df = df[df['balance'] == 0].copy()
non_zero_bal_df = df[df['balance'] > 0].copy()

print("=" * 80)
print("ZERO BALANCE CUSTOMER ANALYSIS")
print("=" * 80)
print(f"\nZero Balance Customers: {len(zero_bal_df):,} ({len(zero_bal_df)/len(df)*100:.1f}%)")
print(f"Non-Zero Balance Customers: {len(non_zero_bal_df):,} ({len(non_zero_bal_df)/len(df)*100:.1f}%)")

zero_churn = zero_bal_df['exited'].sum()
zero_churn_rate = (zero_churn / len(zero_bal_df) * 100) if len(zero_bal_df) > 0 else 0

non_zero_churn = non_zero_bal_df['exited'].sum()
non_zero_churn_rate = (non_zero_churn / len(non_zero_bal_df) * 100) if len(non_zero_bal_df) > 0 else 0

print(f"\nZero Balance Churn Rate: {zero_churn_rate:.2f}%")
print(f"Non-Zero Balance Churn Rate: {non_zero_churn_rate:.2f}%")

# Check if zero balance customers are inactive
zero_inactive_pct = (~zero_bal_df['isactivemember']).mean() * 100
print(f"\nZero Balance customers who are INACTIVE: {zero_inactive_pct:.1f}%")
print("=" * 80)

# %% [markdown]
# **Insights from Feature Interactions Analysis:**
#
# **Geography × Credit Score:** No interaction - credit scores uniformly distributed (~650 median) across all countries. Germany's high churn is NOT due to credit quality differences.
#
# **Age × Balance:** No interaction - account balances similar across all age groups (~$100k median). Age-based churn patterns are NOT driven by wealth differences.
#
# **Active Status × Product Count:** Clear pattern - inactive members skew toward 1 product, active members toward 2. This reinforces that 2-product customers are more engaged.
#
# **Zero Balance Deep Dive:** Counterintuitive finding - **zero balance customers actually churn LESS (13.8% vs 24.1%)**. Despite 36% of customers having $0 balance and 48% being inactive, they're more loyal than funded accounts. These may be "parking" accounts customers want to keep open.
#
# **Key Takeaway:** Most interactions show independence - demographic and financial factors operate separately. The zero-balance paradox suggests these aren't abandoned accounts but intentionally maintained low-activity accounts.
#
# ---

# %% [markdown]
# ---
#
# ## 📈 Section 8: Correlation Analysis
#
# Examine relationships between numerical features to identify correlations and redundancies.

# %%
# Select numerical columns for correlation
numerical_cols = [
    'creditscore', 'age', 'tenure', 'balance', 
    'numofproducts', 'estimatedsalary', 
    'satisfaction_score', 'point_earned'
]

# Add boolean columns as int
corr_df = df[numerical_cols].copy()
corr_df['hascrcard'] = df['hascrcard'].astype(int)
corr_df['isactivemember'] = df['isactivemember'].astype(int)
corr_df['exited'] = df['exited'].astype(int)
corr_df['complain'] = df['complain'].astype(int)

# Create correlation heatmap
correlation_heatmap(corr_df, figsize=(14, 12))
plt.show()

# %%
# Feature correlation with target (exited)
target_corr = corr_df.corr()['exited'].sort_values(ascending=False)
print("=" * 80)
print("FEATURE CORRELATION WITH CHURN (exited)")
print("=" * 80)
print(target_corr)
print("\n")
print("Top Positive Correlations (Higher value = More likely to churn):")
print(target_corr.head(5))
print("\nTop Negative Correlations (Higher value = Less likely to churn):")
print(target_corr.tail(5))
print("=" * 80)

# %% [markdown]
# **Insights from Correlation Analysis:**
#
# **Complain dominates:** Near-perfect correlation (0.996) with churn - filing a complaint and churning are virtually synonymous.
#
# **Age is the strongest demographic:** Moderate correlation (0.285) confirms the lifecycle pattern from Section 2.
#
# **IsActiveMember is the strongest behavior:** -0.156 correlation (highest negative) confirms inactive members are at elevated risk.
#
# **All other features are weak:** Correlations <0.12 (balance, salary, credit score, tenure, etc.) indicate minimal linear relationships. NumOfProducts shows weak -0.048 but we know from Section 4 it has strong non-linear effects.
#
# **Low multicollinearity:** Heatmap shows most features are independent (gray, near zero). Only notable inter-feature correlation is balance × numofproducts (-0.30), suggesting multi-product customers have lower balances.
#
# **Key Takeaway:** Linear correlation analysis confirms Complain (0.996), Age (0.285), and IsActiveMember (-0.156) as top predictors, but misses non-linear relationships like NumOfProducts' U-shape. Tree-based models will capture these better than linear models.
#
# ---

# %% [markdown]
# ---
#
# ## 🛠️ Section 9: Data Preparation Function
#
# Based on EDA insights, create a preprocessing function for model building.
#
# ### Feature Selection Strategy:
#
# **✅ KEEP (Strong/Moderate Predictors):**
# - `complain` - 0.996 correlation, 99.5% churn rate (strongest predictor)
# - `age` - 0.285 correlation, clear lifecycle pattern (7% → 56% churn)
# - `isactivemember` - -0.156 correlation, 1.9× difference (27% vs 14%)
# - `numofproducts` - Strong non-linear effect (7.6% → 100% churn)
# - `geography` - 2× difference (Germany 32% vs France/Spain 16%)
# - `balance` - Bimodal pattern, moderate effect (13.8% vs 24.1%)
# - `gender` - Weak but present (1.5× difference, 25% vs 16%)
# - `tenure` - Keep for completeness despite weak correlation
#
# **❌ DROP (No Predictive Value):**
# - `card_type` - Only 1.1× difference (19-22% churn across all tiers)
# - `hascrcard` - Only 1.0× difference (20.8% vs 20.2% churn)
# - `satisfaction_score` - Only 1.1× difference (flat 19-22% across scores)
# - `point_earned` - Nearly identical distributions (-0.005 correlation)
# - `estimatedsalary` - Uniform distributions (0.012 correlation)
# - `creditscore` - Overlapping distributions (-0.027 correlation)

# %%
def prepare_data_for_modeling(filepath):
    """
    Prepare bank churn data for machine learning models.
    
    Based on EDA insights, performs aggressive feature selection to keep
    only features with demonstrated predictive value.
    
    Actions:
    - Drops identifier columns (RowNumber, CustomerId, Surname)
    - Drops weak predictors (card_type, hascrcard, satisfaction_score, 
      point_earned, estimatedsalary, creditscore)
    - Creates age_group feature (lifecycle pattern discovered in EDA)
    - Encodes categorical variables (gender, geography)
    - Standardizes binary features to int (0/1)
    
    Parameters
    ----------
    filepath : str
        Path to the Customer-Churn-Records.csv file
        
    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe with 8 core features + engineered age_group
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Drop identifier columns
    df.drop(['RowNumber', 'CustomerId', 'Surname'], inplace=True, axis=1)
    
    # Standardize column names
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
    
    # Drop weak/useless features based on EDA
    weak_features = ['card_type', 'hascrcard', 'satisfaction_score', 
                     'point_earned', 'estimatedsalary', 'creditscore']
    df.drop(weak_features, axis=1, inplace=True)
    
    # Convert binary columns to int (0/1)
    bin_cols = ['isactivemember', 'exited', 'complain']
    for col in bin_cols:
        df[col] = df[col].astype(int)
    
    # Encode gender: Male=0, Female=1
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    
    # One-hot encode geography (drop first to avoid multicollinearity)
    df = pd.get_dummies(df, columns=['geography'], drop_first=True, dtype=int)
    
    # Create age_group feature (discovered strong lifecycle pattern)
    df['age_group'] = pd.cut(df['age'], 
                              bins=[0, 30, 40, 50, 60, 70, 100],
                              labels=['18-30', '31-40', '41-50', '51-60', '61-70', '70+'])
    
    return df

print("✓ Data preparation function defined")
print("✓ Feature selection: 8 core features + age_group engineering")
print("✓ Dropped 6 weak predictors based on EDA findings")

# %%
# Test the function
df_prepared = prepare_data_for_modeling("data/Customer-Churn-Records.csv")
print(f"Prepared data shape: {df_prepared.shape}")
print(f"\nColumns ({len(df_prepared.columns)}):")
print(list(df_prepared.columns))
print(f"\nSample of prepared data:")
print(df_prepared.head())

# %% [markdown]
# ---
#
# ## 📝 EDA Summary & Key Findings
#
# ### 🎯 The Critical Three Predictors:
#
# 1. **Complain (0.996 correlation)** - Filing a complaint = 99.5% churn. THE strongest predictor by far.
# 2. **Age (0.285 correlation)** - Lifecycle pattern: 51-60 age group has 56% churn (retirement planning).
# 3. **IsActiveMember (-0.156 correlation)** - Inactive members: 27% churn vs 14% for active (1.9× difference).
#
# ### 📊 Tier 2: Strong Non-Linear Predictors:
#
# 4. **NumOfProducts** - U-shaped: 1 product (28%) → 2 products (7.6% OPTIMAL) → 3 products (83%) → 4 products (100%).
# 5. **Geography** - Germany 32% churn = 2× higher than France/Spain (16%).
#
# ### 📊 Tier 3: Moderate Predictors:
#
# 6. **Balance** - Bimodal (36% have $0). Counterintuitively, zero balance = lower churn (13.8% vs 24.1%).
# 7. **Gender** - Females 25% vs Males 16% (1.5× difference).
#
# ### ❌ Surprisingly Useless Features:
#
# - **Tenure** - Flat churn across all tenure groups (18-21%). Loyalty ≠ retention.
# - **Satisfaction Score** - No effect (19-22% churn across all scores 1-5).
# - **Card Type** - Premium tiers don't retain better (19-22% across DIAMOND/GOLD/PLATINUM/SILVER).
# - **Credit Score, Salary, Points Earned, HasCrCard** - All show nearly identical distributions for churned vs retained.
#
# ### 💡 Key Business Insights:
#
# 1. **Complaint prevention is paramount** - 99.5% of complainers leave. Focus on preventing issues that cause complaints.
# 2. **Target 51-60 age group** - More than half churn (likely retirement planning). Need specialized retention.
# 3. **Re-activate inactive members** - Nearly 2× higher churn risk. Engagement campaigns critical.
# 4. **Stop aggressive cross-selling** - 3-4 product customers have catastrophic churn (83-100%). The "2 product sweet spot" is real.
# 5. **Investigate Germany operations** - 2× higher churn than other markets suggests systemic issues.
# 6. **Zero-balance accounts aren't dead** - They churn less, suggesting intentional "parking" accounts worth keeping.
#
# ### 🎯 Final Feature Set for Modeling:
#
# **KEEP (8 features):**
# - `complain`, `age`, `isactivemember`, `numofproducts`, `geography`, `balance`, `gender`, `tenure`
#
# **DROP (6 features):**
# - `card_type`, `hascrcard`, `satisfaction_score`, `point_earned`, `estimatedsalary`, `creditscore`
#
# ### 📈 Ready for Next Steps:
# - **01_NEW_CustomersSurvivalAnalysis.ipynb**: Time-to-churn analysis with Cox regression
# - **02_NEW_Churn Prediction Model.ipynb**: Build predictive models (Random Forest, XGBoost, etc.)
#
# ---
#
# **🎉 EDA Complete! Ready to build predictive models with evidence-based feature selection.**

# %%

