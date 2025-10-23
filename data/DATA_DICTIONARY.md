# Bank Customer Churn Dataset - Data Dictionary

## Dataset Overview
- **Source**: Bank customer records
- **Size**: 10,000 customers × 18 features
- **Purpose**: Predict customer churn (Exited)
- **File**: `Customer-Churn-Records.csv`

---

## Column Descriptions

### 🔢 Identifier Columns (Not used in modeling)

| Column | Type | Description | Modeling Use |
|--------|------|-------------|--------------|
| **RowNumber** | int64 | Record (row) number | ❌ Drop - No predictive value |
| **CustomerId** | int64 | Unique customer identifier | ❌ Drop - Random ID, no predictive value |
| **Surname** | object | Customer's last name | ❌ Drop - No impact on churn |

---

### 👥 Demographic Features (Innate or relatively stable characteristics)

| Column | Type | Description | Values | Modeling Impact |
|--------|------|-------------|--------|-----------------|
| **Gender** | object | Customer's gender | Male, Female | ⚠️ Typically **weak** predictor |
| **Age** | int64 | Customer's age in years | 18-100+ | ✅ **Strong** predictor - older customers more loyal |
| **Geography** | object | Customer's country | France, Spain, Germany | ✅ **Moderate-Strong** - regional differences in banking behavior |

---

### 💳 Financial Profile Features

| Column | Type | Description | Range | Modeling Impact |
|--------|------|-------------|-------|-----------------|
| **CreditScore** | int64 | Customer's credit score | ~350-850 | ✅ **Moderate** - higher scores = more stable, but bidirectional effect possible |
| **Balance** | float64 | Current account balance ($) | 0 - ~250,000 | ✅ **Strong** - zero balance = high churn risk |
| **EstimatedSalary** | float64 | Annual salary estimate ($) | ~0 - 200,000 | ⚠️ May be **weak** - all income levels can churn |

---

### 🏦 Banking Relationship Features

| Column | Type | Description | Range | Modeling Impact |
|--------|------|-------------|-------|-----------------|
| **Tenure** | int64 | Years as bank customer | 0-10 | ✅ **Strong** - longer tenure = more loyalty |
| **NumOfProducts** | int64 | Number of bank products held | 1-4 | ✅ **Strong** - more products = more invested |
| **HasCrCard** | int64 (binary) | Has credit card with bank | 0 (No), 1 (Yes) | ✅ **Moderate** - indicates relationship strength |
| **IsActiveMember** | int64 (binary) | Active account usage | 0 (Inactive), 1 (Active) | ✅✅ **Very Strong** - activity = engagement |

---

### ⭐ Customer Experience Features (Unique to this dataset)

| Column | Type | Description | Values | Modeling Impact |
|--------|------|-------------|--------|-----------------|
| **Complain** | int64 (binary) | Customer filed complaint | 0 (No), 1 (Yes) | ✅✅ **Very Strong** - complaints strongly indicate dissatisfaction |
| **Satisfaction Score** | int64 | Satisfaction rating | 1-5 (1=lowest, 5=highest) | ✅ **Moderate-Strong** - lower scores = higher churn |
| **Card Type** | object | Credit card tier | SILVER, GOLD, PLATINUM, DIAMOND | ✅ **Moderate** - premium tiers may indicate value/loyalty |
| **Point Earned** | int64 | Loyalty rewards points | Variable | ✅ **Moderate** - more points = more engagement |

---

### 🎯 Target Variable

| Column | Type | Description | Values | Purpose |
|--------|------|-------------|--------|---------|
| **Exited** | int64 (binary) | Customer left the bank | 0 (Retained), 1 (Churned) | **TARGET** - What we're predicting |

---

## Feature Categories for Analysis

### Section 1: Data Loading
- Load all 18 columns
- Drop: RowNumber, CustomerId, Surname
- Keep: 15 features

### Section 2: Demographics
- **Gender** (innate)
- **Geography** (location)
- **Age** → create **age_group** (binned)
- *(Note: HasCrCard is borderline - could be here or in "Banking Relationship")*

### Section 3: Customer Engagement & Activity
- **IsActiveMember** (behavioral)
- **Tenure** → create **tenure_group** (binned)
- Cross-analysis of engagement factors

### Section 4: Product & Service Usage
- **NumOfProducts** (product adoption)
- **HasCrCard** (could also go here as product ownership)

### Section 5: Financial Metrics
- **Balance** (especially zero-balance analysis)
- **EstimatedSalary**
- **CreditScore**

### Section 6: Customer Experience Indicators
- **Satisfaction Score**
- **Complain**
- **Card Type**
- **Point Earned**

---

## Data Quality Notes

### Missing Values
- ✅ **No missing values** in original dataset

### Data Types
- **Binary features**: Convert HasCrCard, IsActiveMember, Exited, Complain to boolean
- **Categorical features**: Gender, Geography, Card Type (strings)
- **Numerical features**: Age, Tenure, Balance, CreditScore, EstimatedSalary, etc.

### Feature Engineering Opportunities
1. **age_group**: Bin age into ranges (18-30, 31-40, 41-50, 51-60, 61-70, 70+)
2. **tenure_group**: Bin tenure into ranges (0-2, 3-4, 5-6, 7-8, 9+ years)
3. **balance_category**: Zero vs Non-zero balance (strong signal)
4. **high_value_customer**: Combination of high balance + high salary + premium card

---

## Key Insights from Data Dictionary

### Expected Strong Predictors 💪
1. **Complain** - Direct dissatisfaction indicator
2. **IsActiveMember** - Behavioral engagement
3. **Age** - Life stage and loyalty
4. **NumOfProducts** - Investment in bank
5. **Balance** - Especially zero balance

### Expected Moderate Predictors 📊
1. **Tenure** - Loyalty over time
2. **Geography** - Regional differences
3. **Satisfaction Score** - Sentiment indicator
4. **Card Type** - Value tier

### Expected Weak Predictors ⚠️
1. **Gender** - Historically minimal impact
2. **EstimatedSalary** - All income levels can churn
3. **CreditScore** - May have non-linear or bidirectional effects

---

## Comparison to Telco Churn Dataset

| Aspect | Telco Dataset | Bank Dataset |
|--------|--------------|--------------|
| **Industry** | Telecommunications | Banking |
| **Services** | Individual services (Internet, Streaming, etc.) | Product count (1-4) |
| **Contract** | Contract types | Not applicable |
| **Charges** | Monthly + Total charges | Balance + Salary |
| **Experience** | Implicit (usage patterns) | **Explicit** (Satisfaction, Complaints) |
| **Loyalty** | Tenure | Tenure + Points + Card Type |
| **Geography** | Not included | **Included** (major factor) |

---

## Usage in Machine Learning Pipeline

### 1. Data Preparation
- Drop identifiers: RowNumber, CustomerId, Surname
- Encode binary: HasCrCard, IsActiveMember, Exited, Complain → 0/1
- Encode categorical: Geography, Card Type → one-hot encoding
- Encode gender: Male=0, Female=1

### 2. Feature Selection
- Use correlation analysis to identify redundancies
- Consider PCA or feature importance from models
- May drop features with very low correlation to target

### 3. Model Building
- Target: **Exited** (binary classification)
- All other features (after encoding): Predictors
- Potential class imbalance: Check churn rate (~20%)

---

**Last Updated**: 2025-10-22  
**Created By**: Data Science Team

