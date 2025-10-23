# %% [markdown]
# # Customer Survival Analysis - Bank Churn Dataset
#
# This notebook performs survival analysis to understand time-to-churn patterns
# and identify which customer segments have significantly different retention rates.
#
# **Key Questions:**
# - How long do customers typically stay with the bank before churning?
# - Which customer segments have significantly different survival curves?
# - What are the hazard ratios for different features?
# - Can we predict individual customer lifetime?

# %% [markdown]
# ## Theory: Survival Analysis Fundamentals
#
# If time to event has the probability density function $f(t)$ and cumulative distribution function $F(t)$, 
# then the probability of surviving at least to time $t$ is: 
#
# $$S(t) = Pr(T>t) = 1-F(t)$$
#
# **Cumulative hazard** at time $t$ is defined as:
#
# $$H(t)=-\ln(S(t))$$
#
# **Instantaneous hazard** at time $t$ is:
#
# $$h(t)=\frac{dH(t)}{dt} = \frac{f(t)}{S(t)}$$
#
# ### Likelihood Function for Survival Analysis
#
# $$ \mathcal{L}(\beta) = \prod_{i=1}^{n} h(t_{i})^{d_{i}} S(t_{i}) $$
#
# where:
# - $d_i$ = censoring variable (1 if event observed, 0 if censored)
# - $h(t_i)$ = hazard for individual $i$ at time $t$
# - $S(t_i)$ = survival probability for individual $i$ at time $t$
#
# The **log-likelihood** is:
#
# $$ \log\mathcal{L}(\beta) = \sum_{i=1}^n d_i \log(h(t_i)) - H(t_i) $$

# %% [markdown]
# ## Importing Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import statsmodels.api as sm

# Lifelines is a survival analysis package
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test, logrank_test

# Import custom plotting functions
from survival_utils import (
    plot_survival_analysis_2groups,
    plot_survival_analysis_multigroup,
    prepare_survival_data
)

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("✓ Libraries loaded successfully")

# %% [markdown]
# ## Data Preparation

# %%
# Load data
df = pd.read_csv("../data/Customer-Churn-Records.csv")
print(f"Data shape: {df.shape}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nFirst few rows:")
df.head()

# %% [markdown]
# ### Data Preprocessing

# %%
# Drop identifier columns
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Standardize column names
df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

# Convert binary columns to int (0/1)
bin_cols = ['hascrcard', 'isactivemember', 'exited', 'complain']
for col in bin_cols:
    df[col] = df[col].astype(int)

# Encode gender: Male=0, Female=1
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

# Create age groups (based on EDA findings)
df['age_group'] = pd.cut(df['age'], 
                          bins=[0, 30, 40, 50, 60, 70, 100],
                          labels=['18-30', '31-40', '41-50', '51-60', '61-70', '70+'])

# Create balance groups
df['balance_group'] = pd.cut(df['balance'],
                              bins=[-1, 0, 50000, 100000, 150000, 300000],
                              labels=['Zero', 'Low', 'Medium', 'High', 'Very High'])

# Create tenure groups
df['tenure_group'] = pd.cut(df['tenure'],
                             bins=[0, 2, 4, 6, 8, 11],
                             labels=['0-2', '3-4', '5-6', '7-8', '9-10'])

print(f"✓ Data preprocessed. Shape: {df.shape}")
print(f"\nFeatures: {list(df.columns)}")

# %%
# Define event and time variables for survival analysis
eventvar = df['exited']  # Churn indicator (0 = retained, 1 = churned)
timevar = df['tenure']   # Time customer remained with bank (in years)

print(f"Event variable (exited): {eventvar.value_counts().to_dict()}")
print(f"Time variable (tenure) - range: {timevar.min()} to {timevar.max()} years")
print(f"Time variable (tenure) - mean: {timevar.mean():.2f} years")

# %% [markdown]
# ---
#
# ## Overall Kaplan-Meier Survival Curve

# %% [markdown]
# The Kaplan-Meier estimator provides a non-parametric estimate of the survival function.
# This shows the probability that a customer will remain with the bank beyond time $t$.

# %%
# Create a Kaplan-Meier object
kmf = KaplanMeierFitter()

# Calculate the K-M curve for all customers
kmf.fit(timevar, event_observed=eventvar, label="All Customers")

# Plot the curve
fig, ax = plt.subplots(figsize=(12, 6))
kmf.plot(ax=ax, ci_show=True)
ax.set_ylabel('Probability of Customer Retention', fontsize=12, fontweight='bold')
ax.set_xlabel('Tenure (years)', fontsize=12, fontweight='bold')
ax.set_title('Kaplan-Meier Survival Curve: Bank Customer Retention', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Median survival time: {kmf.median_survival_time_:.2f} years")

# %% [markdown]
# ### Interpretation
#
# The K-M curve shows how customer retention probability changes over time. 
# A steeper decline indicates periods of higher churn risk. The median survival time 
# tells us when 50% of customers are expected to have churned.

# %% [markdown]
# ---
#
# ## Log-Rank Test Theory

# %% [markdown]
# ### Statistical Framework
#
# The **log-rank test** is a non-parametric method to compare survival curves between groups.
# It assumes proportional hazards and tests the null hypothesis that survival probabilities 
# are the same across groups at all time points.
#
# **Test statistic** for group $j$:
#
# $$k_{j} = \frac{(O_{j}-E_{j})^{2}}{var(O_{j}-E_{j})}$$
#
# where:
# - $O_{j}$ = observed events in group $j$
# - $E_{j}$ = expected events in group $j$ under null hypothesis
# - $var(O_{j}-E_{j})$ = variance of the difference
#
# The observed minus expected for group $j$:
#
# $$O_{j}-E_{j} = \sum_{i}(o_{ij}-e_{ij})$$
#
# The variance:
#
# $$var(O_{j}-E_{j}) = \sum_i o_{i}\frac{n_{ij}}{n_{i}}\Big(1-\frac{n_{ij}}{n_{i}}\Big)\frac{(n_{i}-o_{i})}{(n_{i}-1)}$$
#
# For **multiple groups**, the test statistic is:
#
# $$\chi^2 = \mathbf{Z} \Sigma^{-1} \mathbf{Z}'$$
#
# which follows a $\chi^2$ distribution with $k-1$ degrees of freedom, where $k$ is the number of groups.

# %% [markdown]
# ---
#
# ## Group Comparisons with Log-Rank Tests
#
# Based on our EDA findings, we'll compare survival curves for the features that showed 
# significant differences in churn rates:
#
# **Priority features from EDA:**
# 1. Complain (99.5% churn for complainers - strongest predictor)
# 2. Age Group (lifecycle pattern: 51-60 has 56% churn)
# 3. IsActiveMember (1.9× difference: 27% vs 14%)
# 4. NumOfProducts (U-shape: 7.6% for 2 products, 100% for 4)
# 5. Geography (Germany 32% vs France/Spain 16%)

# %% [markdown]
# ### 1. Complain Status

# %%
# From EDA: THE strongest predictor (0.996 correlation)
complain_mask = (df['complain'] == 1)
no_complain_mask = (df['complain'] == 0)

result_complain = plot_survival_analysis_2groups(
    timevar, eventvar, complain_mask, no_complain_mask,
    "Filed Complaint", "No Complaint",
    "Survival Analysis: Complaint Status (STRONGEST PREDICTOR)",
    kmf,
    show_at_risk=True
)

# %% [markdown]
# **Insight:** Customers who file complaints have catastrophically poor retention.
# This is the single most important red flag for churn prediction.

# %% [markdown]
# ### 2. Active Member Status

# %%
# From EDA: Strong behavioral predictor (-0.156 correlation)
active_mask = (df['isactivemember'] == 1)
inactive_mask = (df['isactivemember'] == 0)

result_active = plot_survival_analysis_2groups(
    timevar, eventvar, active_mask, inactive_mask,
    "Active Member", "Inactive Member",
    "Survival Analysis: Active Member Status",
    kmf,
    show_at_risk=True
)

# %% [markdown]
# **Insight:** Inactive members show significantly lower retention. 
# Re-activation campaigns should target this segment.

# %% [markdown]
# ### 3. Age Group

# %%
# From EDA: Strong demographic predictor with clear lifecycle pattern
# Create masks for each age group
age_18_30 = (df['age_group'] == '18-30')
age_31_40 = (df['age_group'] == '31-40')
age_41_50 = (df['age_group'] == '41-50')
age_51_60 = (df['age_group'] == '51-60')
age_61_70 = (df['age_group'] == '61-70')
age_70_plus = (df['age_group'] == '70+')

result_age = plot_survival_analysis_multigroup(
    timevar, eventvar, df, 'age_group',
    [age_18_30, age_31_40, age_41_50, age_51_60, age_61_70, age_70_plus],
    ["18-30 (Young)", "31-40 (Mid-Career)", "41-50 (Mid-Life)", 
     "51-60 (Pre-Retirement)", "61-70 (Senior)", "70+ (Elderly)"],
    "Survival Analysis: Age Groups (Lifecycle Pattern)",
    kmf,
    show_at_risk=True
)

# %% [markdown]
# **Insight:** The 51-60 age group (pre-retirement) shows dramatically lower retention.
# This segment likely moves assets for retirement planning and requires specialized retention strategies.

# %% [markdown]
# ### 4. Number of Products

# %%
# From EDA: Strong non-linear predictor (U-shape with 2 products optimal)
prod_1 = (df['numofproducts'] == 1)
prod_2 = (df['numofproducts'] == 2)
prod_3 = (df['numofproducts'] == 3)
prod_4 = (df['numofproducts'] == 4)

result_products = plot_survival_analysis_multigroup(
    timevar, eventvar, df, 'numofproducts',
    [prod_1, prod_2, prod_3, prod_4],
    ["1 Product", "2 Products (OPTIMAL)", "3 Products", "4 Products"],
    "Survival Analysis: Number of Products (U-Shape Effect)",
    kmf,
    show_at_risk=True
)

# %% [markdown]
# **Insight:** Customers with 2 products have the best retention (7.6% churn).
# Those with 3-4 products have catastrophic churn - likely over-selling backfires.

# %% [markdown]
# ### 5. Geography

# %%
# From EDA: Strong geographic effect (Germany 2× higher churn)
france_mask = (df['geography'] == 'France')
spain_mask = (df['geography'] == 'Spain')
germany_mask = (df['geography'] == 'Germany')

result_geo = plot_survival_analysis_multigroup(
    timevar, eventvar, df, 'geography',
    [france_mask, spain_mask, germany_mask],
    ["France", "Spain", "Germany"],
    "Survival Analysis: Geography (Market Differences)",
    kmf
)

# %% [markdown]
# **Insight:** Germany shows significantly lower retention than France/Spain.
# This suggests market-specific issues requiring investigation.

# %% [markdown]
# ### 6. Gender

# %%
# From EDA: Moderate demographic effect (1.5× difference)
female_mask = (df['gender'] == 1)
male_mask = (df['gender'] == 0)

result_gender = plot_survival_analysis_2groups(
    timevar, eventvar, female_mask, male_mask,
    "Female", "Male",
    "Survival Analysis: Gender",
    kmf
)

# %% [markdown]
# **Insight:** Females show moderately higher churn than males.

# %% [markdown]
# ### 7. Balance Group (Zero vs Non-Zero)

# %%
# From EDA: Counterintuitive finding - zero balance has LOWER churn
zero_balance = (df['balance'] == 0)
nonzero_balance = (df['balance'] > 0)

result_balance = plot_survival_analysis_2groups(
    timevar, eventvar, zero_balance, nonzero_balance,
    "Zero Balance", "Non-Zero Balance",
    "Survival Analysis: Account Balance (Counterintuitive Finding)",
    kmf,
    show_at_risk=True
)

# %% [markdown]
# **Insight:** Zero-balance accounts actually have BETTER retention (13.8% vs 24.1% churn).
# These may be "parking" accounts customers intentionally maintain.

# %% [markdown]
# ---
#
# ## Cox Proportional Hazards Model
#
# ### Theory
#
# The Cox PH model is a semi-parametric regression model that estimates the hazard ratio 
# for each predictor while leaving the baseline hazard unspecified:
#
# $$h(t|X) = h_0(t) \exp(\beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p)$$
#
# where:
# - $h(t|X)$ = hazard at time $t$ given covariates $X$
# - $h_0(t)$ = baseline hazard function
# - $\beta$ = coefficients (log hazard ratios)
# - $\exp(\beta_i)$ = hazard ratio for feature $i$
#
# **Hazard Ratio Interpretation:**
# - HR > 1: Feature increases churn risk
# - HR < 1: Feature decreases churn risk (protective)
# - HR = 1: No effect

# %% [markdown]
# ### Data Preparation for Cox PH Model
#
# **Feature Selection Strategy:**
#
# **Dropped 6 weak features** (from EDA Section 9):
# - ❌ `card_type` - No predictive value (1.1× difference)
# - ❌ `hascrcard` - No predictive value (1.0× difference)
# - ❌ `satisfaction_score` - No predictive value (1.1× difference)
# - ❌ `point_earned` - No predictive value (-0.005 correlation)
# - ❌ `estimatedsalary` - No predictive value (0.012 correlation)
# - ❌ `creditscore` - No predictive value (-0.027 correlation)
#
# **Dropped 2 problematic features** (for Cox PH model):
# - ❌ `age` - Multicollinearity with `age_group` (both measure the same thing)
# - ❌ `complain` - Too dominant (exp(coef) = 2,652×, 99.5% churn rate)
#   - Complaint status drowns out all other features
#   - We already know from log-rank tests that complain is THE strongest predictor
#   - Removing it lets us see the effects of other features
#
# **Final feature set:** 6 core features + age_group categories
# - ✅ Gender, tenure, balance, numofproducts, isactivemember, geography, age_group

# %%
# Prepare data for regression
regression_df = df.copy()

# Drop weak/useless features identified in EDA
weak_features = ['card_type', 'hascrcard', 'satisfaction_score', 
                 'point_earned', 'estimatedsalary', 'creditscore']
regression_df = regression_df.drop(weak_features, axis=1)
print(f"✓ Dropped {len(weak_features)} weak features: {weak_features}")

# Drop features that cause issues in Cox PH model
# 1. Drop 'age' to avoid multicollinearity with age_group
# 2. Drop 'complain' because it dominates all other effects (exp(coef) = 2,652×)
problematic_features = ['age', 'complain']
regression_df = regression_df.drop(problematic_features, axis=1)
print(f"✓ Dropped {len(problematic_features)} problematic features for Cox model: {problematic_features}")
print("  - age: Multicollinearity with age_group_*")
print("  - complain: Too dominant (99.5% churn rate), drowns out other features")

# Create dummy variables for categorical features (only the useful ones)
categorical_cols = ['geography', 'age_group']
regression_df = pd.get_dummies(regression_df, columns=categorical_cols, drop_first=True, dtype=int)

# Drop grouped versions (already have continuous versions)
drop_cols = ['balance_group', 'tenure_group']
regression_df = regression_df.drop(drop_cols, axis=1)

print(f"\nRegression data shape: {regression_df.shape}")
print(f"Columns ({len(regression_df.columns)}): {list(regression_df.columns)}")
print(f"\n✓ Cox PH Model will use 6 core features:")
print("  - isactivemember, numofproducts, geography, balance, gender, tenure, age_group")
print(f"\nNote: Run separate analysis if you want to see 'complain' effect (it's THE dominant predictor)")
regression_df.head()

# %% [markdown]
# ### Fit Cox PH Model

# %%
# Initialize Cox PH fitter
cph = CoxPHFitter()

# Fit model (using tenure as duration and exited as event)
cph.fit(regression_df, duration_col='tenure', event_col='exited')

# Print summary
cph.print_summary()

# %%
# Model performance: Concordance Index (C-index)
# C-index = 1.0: Perfect predictions
# C-index = 0.5: Random predictions
print(f"\n{'='*80}")
print(f"MODEL PERFORMANCE")
print(f"{'='*80}")
print(f"Concordance Index (C-index): {cph.concordance_index_:.4f}")
print(f"\nInterpretation:")
if cph.concordance_index_ >= 0.8:
    print("✓ Excellent predictive power (C-index ≥ 0.8)")
elif cph.concordance_index_ >= 0.7:
    print("✓ Good predictive power (0.7 ≤ C-index < 0.8)")
elif cph.concordance_index_ >= 0.6:
    print("○ Moderate predictive power (0.6 ≤ C-index < 0.7)")
else:
    print("✗ Weak predictive power (C-index < 0.6)")
print(f"{'='*80}")

# %% [markdown]
# ### Visualize Feature Coefficients

# %%
fig, ax = plt.subplots(figsize=(10, 8))
cph.plot(ax=ax)
ax.set_title('Cox Proportional Hazards Model - Feature Coefficients (Log Hazard Ratios)', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Coefficient (log HR)', fontsize=12)
ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Interpretation:**
# - **Positive coefficients** (right of 0): Increase churn risk (higher hazard)
# - **Negative coefficients** (left of 0): Decrease churn risk (protective)
# - **Further from 0**: Stronger effect
# - **Error bars crossing 0**: Not statistically significant

# %% [markdown]
# ### Key Findings from Cox PH Model
#
# **Model Performance:**
# - **C-index: 0.74** (Good predictive power)
# - Down from 0.92 when `complain` was included, but now we can see other features' effects
#
# **Top Risk Factors (Hazard Ratios > 1.5):**
#
# 1. **Age Group 51-60** (exp(coef) = 7.94, p < 0.005) ⚠️ HIGHEST RISK
#    - Pre-retirement customers have **7.94× higher churn risk** vs baseline (18-30)
#    - This lifecycle stage is the critical vulnerability window
#    - Likely moving assets for retirement planning
#
# 2. **Age Group 61-70** (exp(coef) = 5.12, p < 0.005)
#    - Early retirement customers have **5.12× higher churn risk** vs baseline
#    - Second highest risk group
#
# 3. **Age Group 41-50** (exp(coef) = 4.31, p < 0.005)
#    - Mid-life customers have **4.31× higher churn risk** vs baseline
#    - Clear lifecycle pattern emerges: risk increases with age until retirement
#
# 4. **Age Group 31-40** (exp(coef) = 1.61, p < 0.005)
#    - Mid-career customers have **1.61× higher churn risk** vs baseline
#    - Young adults (18-30) have the best retention
#
# 5. **Germany Market** (exp(coef) = 1.60, p < 0.005)
#    - German customers have **1.60× higher churn risk** vs France (baseline)
#    - Market-specific issues require investigation
#
# 6. **Female Gender** (exp(coef) = 1.47, p < 0.005)
#    - Female customers have **1.47× higher churn risk** vs males
#    - Moderate but significant gender effect
#
# **Protective Factors (Hazard Ratios < 1):**
#
# 1. **Active Membership** (exp(coef) = 0.54, p < 0.005) ✓ STRONGEST PROTECTIVE
#    - Active members have **46% lower churn risk** (0.54× the hazard)
#    - Re-activation campaigns are critical
#
# 2. **Number of Products** (exp(coef) = 0.92, p = 0.04)
#    - Each additional product reduces churn by **8%** (0.92× per product)
#    - BUT: EDA showed U-shape (2 products optimal, 3-4 catastrophic)
#    - This linear effect masks the non-linear pattern
#
# **Non-Significant Features:**
# - **Spain Geography** (p = 0.38): No difference vs France
# - **Age Group 70+** (p = 0.40): Small sample size, wide confidence intervals
# - **Balance** (p < 0.005 but HR ≈ 1.00): Statistically significant but practically negligible effect
#
# **Key Insight:**  
# Age lifecycle is the dominant pattern after removing complaints. The 51-60 age group is 
# the critical intervention point. Active membership is the strongest modifiable protective factor.

# %% [markdown]
# ---
#
# ## Individual Customer Predictions

# %% [markdown]
# The Cox PH model allows us to predict survival probabilities and cumulative hazard 
# for individual customers based on their characteristics.

# %%
# Select a random customer for demonstration
test_customer = regression_df.sample(1, random_state=42)
print("Selected customer profile:")
print(test_customer.T)

# %% [markdown]
# ### Cumulative Hazard Over Time

# %%
fig, ax = plt.subplots(figsize=(10, 6))
cph.predict_cumulative_hazard(test_customer).plot(ax=ax, color='red', linewidth=2)
plt.axvline(x=test_customer['tenure'].values[0], color='blue', linestyle='--', linewidth=2)
plt.legend(labels=['Cumulative Hazard', 'Current Tenure'], fontsize=11)
ax.set_xlabel('Tenure (years)', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Hazard', fontsize=12, fontweight='bold')
ax.set_title('Customer Cumulative Hazard Over Time', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Survival Probability Over Time

# %%
fig, ax = plt.subplots(figsize=(10, 6))
cph.predict_survival_function(test_customer).plot(ax=ax, color='red', linewidth=2)
plt.axvline(x=test_customer['tenure'].values[0], color='blue', linestyle='--', linewidth=2)
plt.legend(labels=['Survival Probability', 'Current Tenure'], fontsize=11)
ax.set_xlabel('Tenure (years)', fontsize=12, fontweight='bold')
ax.set_ylabel('Survival Probability', fontsize=12, fontweight='bold')
ax.set_title('Customer Survival Probability Over Time', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
#
# ## Saving the Model

# %%
import pickle

# Save the Cox PH model
with open('survivemodel.pkl', 'wb') as f:
    pickle.dump(cph, f)

print("✓ Model saved to 'survivemodel.pkl'")

# %% [markdown]
# ---
#
# ## Customer Lifetime Value (CLV) Estimation

# %% [markdown]
# ### Methodology
#
# Customer Lifetime Value can be estimated by multiplying the customer's account balance
# (or expected revenue stream) by their expected remaining lifetime.
#
# We'll use the survival function to calculate expected lifetime, defining "churn threshold"
# as when survival probability drops below 10% (conservative estimate).

# %%
def calculate_expected_lifetime(cph, customer_data, threshold=0.10):
    """
    Calculate expected customer lifetime based on survival function.
    
    Parameters
    ----------
    cph : CoxPHFitter
        Fitted Cox PH model
    customer_data : pd.DataFrame
        Customer feature vector
    threshold : float
        Survival probability threshold to consider as "churned" (default: 0.10)
        
    Returns
    -------
    float
        Expected lifetime in years
    """
    # Get survival function
    surv_func = cph.predict_survival_function(customer_data)
    
    # Find time when survival drops below threshold
    surv_curve = surv_func.iloc[:, 0]
    
    # If survival never drops below threshold, use max tenure
    churned_times = surv_curve[surv_curve < threshold]
    if len(churned_times) == 0:
        expected_lifetime = surv_curve.index.max()
    else:
        expected_lifetime = churned_times.index.min()
    
    return expected_lifetime

# %%
# Calculate CLV for sample customer
test_customer_lifetime = calculate_expected_lifetime(cph, test_customer, threshold=0.10)
test_customer_balance = test_customer['balance'].values[0]
test_customer_tenure = test_customer['tenure'].values[0]

# Determine age group from dummy variables (for display purposes)
age_group_cols = [col for col in test_customer.columns if col.startswith('age_group_')]
active_age_group = 'Baseline (18-30)'  # Default if no age_group column is 1
for col in age_group_cols:
    if test_customer[col].values[0] == 1:
        active_age_group = col.replace('age_group_', '')
        break

print(f"{'='*80}")
print(f"CUSTOMER LIFETIME VALUE ESTIMATION")
print(f"{'='*80}")
print(f"Customer Profile:")
print(f"  - Age Group: {active_age_group}")
print(f"  - Current Tenure: {test_customer_tenure:.1f} years")
print(f"  - Account Balance: ${test_customer_balance:,.2f}")
print(f"\nLifetime Prediction:")
print(f"  - Expected Remaining Lifetime: {test_customer_lifetime:.2f} years")
print(f"  - Total Expected Tenure: {test_customer_tenure + test_customer_lifetime:.2f} years")
print(f"\nSimplified CLV (Balance × Remaining Lifetime):")
print(f"  - Estimated CLV: ${test_customer_balance * test_customer_lifetime:,.2f}")
print(f"{'='*80}")

# %% [markdown]
# **Note:** This is a simplified CLV calculation. A more sophisticated model would incorporate:
# - Expected revenue streams (not just balance)
# - Discount rates for time value of money
# - Customer acquisition costs
# - Operating costs per customer
# - Cross-sell/upsell opportunities

# %% [markdown]
# ---
#
# ## Summary & Key Findings
#
# ### Survival Analysis Highlights:
#
# 1. **Complaint Status** - Most critical predictor (not in Cox model)
#    - Customers who complain have near-certain churn (99.5% churn rate)
#    - exp(coef) = 2,652× when included in model - too dominant to model with other features
#    - Survival curves are dramatically different (highly significant log-rank test)
#    - **Action:** Proactive complaint prevention is priority #1
#
# 2. **Age Lifecycle Pattern** - Dominant pattern in Cox PH model
#    - **51-60 age group**: 7.94× higher churn risk (HIGHEST RISK)
#    - **61-70 age group**: 5.12× higher churn risk
#    - **41-50 age group**: 4.31× higher churn risk
#    - **31-40 age group**: 1.61× higher churn risk
#    - **18-30 baseline**: Best retention (reference group)
#    - Clear lifecycle vulnerability: risk peaks during pre-retirement (51-60)
#    - **Action:** Target 51-60 age group with specialized retirement planning offerings
#
# 3. **Active Member Status** - Strongest protective factor (modifiable)
#    - Inactive members have 1.85× higher churn risk (1/0.54)
#    - Active membership reduces risk by 46%
#    - Clear separation in survival curves
#    - **Action:** Re-activation campaigns for inactive members are critical
#
# 4. **Geography** - Market-specific effect
#    - Germany: 1.60× higher churn risk vs France
#    - Spain: No significant difference vs France
#    - **Action:** Investigate and address systemic issues in German market
#
# 5. **Gender** - Moderate demographic effect
#    - Females: 1.47× higher churn risk vs males
#    - **Action:** Consider gender-specific retention strategies
#
# 6. **Number of Products** - Non-linear pattern (U-shape)
#    - Cox model shows 8% risk reduction per product (linear assumption)
#    - BUT: EDA/Log-rank tests reveal U-shape: 2 products optimal (7.6% churn), 3-4 catastrophic (100%)
#    - Cox model's linear effect masks this non-linearity
#    - **Action:** Avoid pushing 3+ products; optimize at 2 products per customer
#
# ### Cox PH Model Performance:
#
# - **C-index: 0.74** (Good predictive power)
# - Provides individual-level churn risk predictions
# - Quantifies hazard ratios for each feature
# - Enables targeted retention interventions
# - Supports customer lifetime value estimation
# - Model excludes `complain` (too dominant) and `age` (multicollinearity with age_group)
#
# ### Top 3 Actionable Recommendations:
#
# 1. **Prevent complaints** - Proactive issue resolution before escalation (99.5% churn if complaint filed)
# 2. **Target pre-retirement customers (51-60)** - Specialized offerings for highest-risk lifecycle stage
# 3. **Re-activate inactive members** - Engagement campaigns for strongest modifiable protective factor
# 4. **Investigate Germany market** - Address market-specific retention issues
# 5. **Product optimization** - Promote 2-product sweet spot, avoid over-selling
#
# ---
#
# **✓ Survival Analysis Complete! Ready for predictive modeling in the next notebook.**

# %%

