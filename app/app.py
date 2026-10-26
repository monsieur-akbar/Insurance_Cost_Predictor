from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Actuarial Health Insurance Claim Estimator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- ELITE CUSTOM STYLING ---
st.markdown(
    """
    <style>
    .main { background-color: #f8fafc; }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #e2e8f0;
    }
    h1, h2, h3 { color: #0f172a; }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = BASE_DIR / "model_bundle.joblib"
ASSETS_DIR = BASE_DIR.parent / "assets"


@st.cache_resource
def load_bundle(path: Path):
    if not path.exists():
        st.error(f"Model bundle not found at: {path}")
        st.stop()
    return joblib.load(path)


bundle = load_bundle(BUNDLE_PATH)
model_6 = bundle["model_6"]
model_12 = bundle["model_12"]
cv_results = bundle["cv_results"]
cities = bundle["cities"]
diseases = bundle["diseases"]
jobs = bundle["jobs"]

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Policyholder Parameters")
st.sidebar.caption("Demographic & Baseline Inputs")

age = st.sidebar.slider("Age (Years)", 18, 65, 34, 1)
sex = st.sidebar.selectbox("Biological Sex", ["male", "female"])
bmi = st.sidebar.number_input("BMI (kg/m²)", 15.0, 55.0, 27.5, 0.5)
dependents = st.sidebar.slider("Number of Dependents", 0, 5, 1, 1)
smoker_label = st.sidebar.selectbox("Smoking Status", ["Non-Smoker", "Smoker"])
smoker = 1 if smoker_label == "Smoker" else 0
city = st.sidebar.selectbox("Residential City", cities, 0)

st.sidebar.markdown("---")
st.sidebar.caption("Clinical & Lifestyle Inputs")

weight = st.sidebar.number_input("Weight (kg)", 35.0, 130.0, 72.0, 1.0)
bp = st.sidebar.slider("Diastolic Blood Pressure (mmHg)", 50, 130, 78, 1)
disease = st.sidebar.selectbox(
    "Hereditary Condition",
    diseases,
    diseases.index("NoDisease") if "NoDisease" in diseases else 0,
)
diabetes = 1 if st.sidebar.selectbox("Diabetes Diagnosis", ["No", "Yes"]) == "Yes" else 0
regular_ex = 1 if st.sidebar.selectbox("Regular Physical Exercise", ["Yes", "No"]) == "Yes" else 0
job = st.sidebar.selectbox("Occupation / Job Title", jobs, 0)

# --- DATAFRAMES ---
input_6 = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "no_of_dependents": dependents,
    "smoker": smoker,
    "city": city,
}])
input_12 = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "weight": weight,
    "bmi": bmi,
    "hereditary_diseases": disease,
    "no_of_dependents": dependents,
    "smoker": smoker,
    "city": city,
    "bloodpressure": bp,
    "diabetes": diabetes,
    "regular_ex": regular_ex,
    "job_title": job,
}])

pred_6 = float(model_6.predict(input_6)[0])
pred_12 = float(model_12.predict(input_12)[0])
delta_val = pred_12 - pred_6

# --- MAIN PAGE HEADER ---
st.title("Actuarial Health Insurance Claim Predictor")
st.markdown(
    "**GDGoC AI/ML Domain Task** — Controlled ablation comparison between a "
    "**6-feature demographic baseline** and a **12-feature clinical Random Forest pipeline**."
)
st.markdown("")

# --- METRIC CARDS ---
c1, c2, c3 = st.columns(3)
with c1:
    st.metric(
        "12-Feature Clinical Estimate",
        f"${pred_12:,.2f}",
        f"{delta_val:+,.2f} vs. Baseline",
        delta_color="inverse",
    )
with c2:
    st.metric("6-Feature Demographic Baseline", f"${pred_6:,.2f}")
with c3:
    risk = "High Risk" if pred_12 > 18000 else ("Moderate Risk" if pred_12 > 9000 else "Standard Risk")
    st.metric("Actuarial Risk Tier", risk)

st.markdown("---")

# --- TABS ---
tab_sim, tab_cv, tab_eda = st.tabs([
    "Counterfactual Risk Simulator",
    "5-Fold Cross-Validation Benchmarks",
    "Model Diagnostics & EDA",
])

with tab_sim:
    st.subheader("Modifiable Lifestyle & Clinical Counterfactual Analysis")
    st.write("Simulates claim cost changes when policyholders adjust modifiable health metrics.")

    scenarios = []
    if smoker == 1:
        d_ns = input_12.copy()
        d_ns["smoker"] = 0
        p_ns = float(model_12.predict(d_ns)[0])
        scenarios.append({
            "Intervention Scenario": "Smoking Cessation",
            "Projected Claim ($)": f"${p_ns:,.2f}",
            "Net Change ($)": f"${p_ns - pred_12:+,.2f}",
            "Impact (%)": f"{((p_ns - pred_12) / pred_12) * 100:+.1f}%",
        })
    if regular_ex == 0:
        d_ex = input_12.copy()
        d_ex["regular_ex"] = 1
        p_ex = float(model_12.predict(d_ex)[0])
        scenarios.append({
            "Intervention Scenario": "Adopt Regular Exercise",
            "Projected Claim ($)": f"${p_ex:,.2f}",
            "Net Change ($)": f"${p_ex - pred_12:+,.2f}",
            "Impact (%)": f"{((p_ex - pred_12) / pred_12) * 100:+.1f}%",
        })
    if bmi >= 25.0:
        d_bmi = input_12.copy()
        d_bmi["bmi"] = 23.5
        p_bmi = float(model_12.predict(d_bmi)[0])
        scenarios.append({
            "Intervention Scenario": "Normalize BMI to 23.5",
            "Projected Claim ($)": f"${p_bmi:,.2f}",
            "Net Change ($)": f"${p_bmi - pred_12:+,.2f}",
            "Impact (%)": f"{((p_bmi - pred_12) / pred_12) * 100:+.1f}%",
        })

    if scenarios:
        st.dataframe(pd.DataFrame(scenarios), hide_index=True, width="stretch")
    else:
        st.info("Policyholder is already at optimal baseline for tested lifestyle interventions.")

with tab_cv:
    st.subheader("5-Fold Cross-Validation Results (k=5, shuffle=True, random_state=42)")
    st.write(
        "Leakage-free evaluation across 15,000 records using scikit-learn `ColumnTransformer` pipelines."
    )
    st.dataframe(
        cv_results.style.format({"CV MAE": "${:,.2f}", "CV RMSE": "${:,.2f}", "CV R2": "{:.4f}"}),
        hide_index=True,
        width="stretch",
    )

with tab_eda:
    st.subheader("Exploratory Data Analysis & Residual Diagnostics")
    col_a, col_b = st.columns(2)
    with col_a:
        if (ASSETS_DIR / "eda_plots.png").exists():
            st.image(str(ASSETS_DIR / "eda_plots.png"), caption="BMI x Smoking Interaction")
        if (ASSETS_DIR / "residual_analysis.png").exists():
            st.image(str(ASSETS_DIR / "residual_analysis.png"), caption="Residual Error Distribution")
    with col_b:
        if (ASSETS_DIR / "correlation_heatmap.png").exists():
            st.image(str(ASSETS_DIR / "correlation_heatmap.png"), caption="Spearman Rank Correlation")
        if (ASSETS_DIR / "feature_importance.png").exists():
            st.image(str(ASSETS_DIR / "feature_importance.png"), caption="Random Forest Feature Importances")