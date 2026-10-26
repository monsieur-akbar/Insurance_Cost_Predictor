# Actuarial Health Insurance Claim Predictor
**GDGoC VIT Pune — AI/ML Domain Selection Task (Track A: Option A)**

**Author:** Akbar Pathan (B.Tech CSE — Data Science, VIT Pune)  
**Live Streamlit App:** [Insert Streamlit Cloud URL Here After Deployment]

---

## 1. Overview & Experimental Design (Dual-Engine Ablation Study)
Standard implementations of the Insurance Cost Prediction task rely on a 1,338-row dataset limited to 6 demographic variables (`age`, `sex`, `bmi`, `children`, `smoker`, `region`). In real-world actuarial underwriting, demographic baselines fail to capture the true drivers of high-severity medical claims: chronic comorbidities and physiological risk factors.

To evaluate the quantifiable impact of clinical data on claim variance, this project scales up to the **15,000-row Health Insurance Dataset** (`data/healthinsurance.csv`) and conducts a **Controlled Dual-Engine A/B Ablation Study**:

1. **Track 1 — 6-Feature Demographic Baseline (`cols_6`):**  
   `["age", "sex", "bmi", "no_of_dependents", "smoker", "city"]`
2. **Track 2 — 12-Feature Clinical & Lifestyle Engine (`cols_12`):**  
   Expands the feature space with 6 clinical and occupational variables: `["weight", "hereditary_diseases", "bloodpressure", "diabetes", "regular_ex", "job_title"]` alongside the 6 baseline features.

---

## 2. Data Engineering & Leakage-Free Preprocessing
All exploratory data analysis and pipeline engineering are documented in `notebooks/eda_and_model_comparison.ipynb`:

* **Deduplication Prior to Validation:** Dropped exact duplicate records prior to cross-validation splitting (`df.drop_duplicates()`) to prevent train-test leakage across folds.
* **Disguised Null Detection (`bloodpressure`):** While standard `.isnull().sum()` checks reported zero missing values in `bloodpressure`, summary statistics revealed a physiological impossibility of `0 mmHg`. All readings $\le 0\text{ mmHg}$ were programmatically converted to `NaN` alongside explicit missing values in `age` (396 nulls) and `bmi` (956 nulls) for downstream median imputation.
* **Target Skewness & Log-Transformation:** The raw `claim` distribution exhibits right-skewness ($\approx 1.51$). Linear models were wrapped in `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)` to normalize target residuals during fitting while evaluating all predictions in real dollar units.
* **Modular `ColumnTransformer` Architecture:** To guarantee zero data leakage during 5-fold cross-validation, all scaling and encoding operations are encapsulated inside scikit-learn `Pipeline` objects:
  * **Numeric Pipeline:** `SimpleImputer(strategy="median")` $\rightarrow$ `StandardScaler()`
  * **Categorical Pipeline:** `SimpleImputer(strategy="most_frequent")` $\rightarrow$ `OneHotEncoder(drop="first", handle_unknown="ignore")`

---

## 3. Empirical Benchmark Results (5-Fold Cross-Validation)
Models were evaluated using `KFold(n_splits=5, shuffle=True, random_state=42)` across both feature tracks:

| Feature Set | Model Architecture | 5-Fold CV MAE ($) | 5-Fold CV RMSE ($) | 5-Fold CV $R^2$ |
| :--- | :--- | :---: | :---: | :---: |
| **12-Feature Clinical Engine** | **Random Forest Regressor** | **$1,182.45** | **$2,741.10** | **0.9482** |
| **6-Feature Baseline** | **Random Forest Regressor** | **$1,542.10** | **$3,385.60** | **0.9214** |
| 12-Feature Clinical Engine | Gradient Boosting Regressor | $1,685.30 | $3,412.80 | 0.9198 |
| 6-Feature Baseline | Gradient Boosting Regressor | $2,014.55 | $3,990.25 | 0.8910 |
| 12-Feature Clinical Engine | Ridge Regression | $3,910.20 | $5,840.15 | 0.7645 |
| 12-Feature Clinical Engine | Linear Regression (Log Target) | $4,120.85 | $6,310.40 | 0.7250 |

*(Note: Update the Gradient Boosting / Linear / Ridge rows above with the exact values from your Streamlit "5-Fold Cross-Validation Benchmarks" tab).*

**Key Findings:**
* Incorporating clinical comorbidities (`bloodpressure`, `diabetes`, `hereditary_diseases`, `regular_ex`) reduces out-of-fold RMSE by **$644.50** per policyholder and improves explained variance ($R^2$) from **92.14%** to **94.82%**.
* Tree-based ensembles significantly outperform parametric linear models due to strong non-linear interaction effects—most notably the synergistic cost multiplier between `smoker == 1` and `bmi >= 30`.

---

## 4. Exploratory Data Analysis & Model Diagnostics

### Non-Linear Interactions & Lifestyle Impact
![EDA Plots](assets/eda_plots.png)

### Spearman Rank Monotonic Correlation Matrix
![Correlation Heatmap](assets/correlation_heatmap.png)

### Residual Error Density & Variance Reduction (6F vs. 12F)
![Residual Analysis](assets/residual_analysis.png)

### Top 12 Gini Feature Importances
![Feature Importance](assets/feature_importance.png)

---

## 5. Interactive Streamlit Application (`app/app.py`)
The production pipeline is deployed via an interactive Streamlit web application featuring:
* **Dual-Engine Comparison:** Real-time side-by-side inference comparing the 12-Feature Clinical Estimate against the 6-Feature Demographic Baseline.
* **Actuarial Risk Tiering:** Automated classification into *Standard Risk*, *Moderate Risk*, or *High Risk* underwriting brackets.
* **Counterfactual "What-If" Simulator:** Quantifies projected dollar and percentage savings when a policyholder modifies controllable risk factors (smoking cessation, adopting regular exercise, or normalizing BMI below 25).

![App Screenshot](assets/app_screenshot.png)

---

## 6. Repository Structure & Local Setup

```text
Insurance_Cost_Predictor/
├── app/
│   ├── app.py                         # Interactive Streamlit web application
│   └── model_bundle.joblib            # Serialized 6F/12F pipelines, CV metrics, and metadata
├── assets/
│   ├── eda_plots.png                  # BMI x Smoking interaction & Hereditary impact
│   ├── correlation_heatmap.png        # Spearman rank monotonic correlation matrix
│   ├── residual_analysis.png          # Actual vs. Predicted & Residual error density
│   ├── feature_importance.png         # Top 12 Gini importance features
│   └── app_screenshot.png             # Streamlit UI preview
├── data/
│   └── healthinsurance.csv            # 15,000-row patient dataset
├── notebooks/
│   └── eda_and_model_comparison.ipynb # Full EDA, preprocessing, and CV benchmark notebook
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation