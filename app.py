import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Model and Scaler
@st.cache_resource
def load_artifacts():
    model_path = 'credit_card_fraud_xgb_model.pkl'
    scaler_path = 'robust_scaler.pkl'

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_artifacts()

# App Header
st.title("🛡️ Credit Card Fraud Detection Dashboard")
st.markdown("Enterprise-grade real-time transaction monitoring powered by **XGBoost**, **RobustScaler**, and **SMOTE**.")

if model is None or scaler is None:
    st.error("⚠️ Model or Scaler files not found! Please ensure 'credit_card_fraud_xgb_model.pkl' and 'robust_scaler.pkl' are in the directory.")
else:
    # Sidebar Configuration
    st.sidebar.header("⚙️ Model Parameters")
    threshold = st.sidebar.slider(
        "Decision Threshold", 
        min_value=0.1, 
        max_value=0.99, 
        value=0.80, 
        step=0.01,
        help="Adjusting threshold higher reduces false positives (false alarms)."
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Pipeline Info:**\n"
        "- **Model:** XGBoost Classifier\n"
        "- **Scaling:** RobustScaler\n"
        "- **Imbalance Handling:** SMOTE\n"
        "- **Trained ROC-AUC:** 0.9786"
    )

    # Input Form
    st.subheader("💳 Transaction Details Input")

    col1, col2 = st.columns(2)

    with col1:
        time_val = st.number_input("Transaction Time (seconds from start)", min_value=0.0, max_value=172800.0, value=406.0, step=1.0)
        amount_val = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=50000.0, value=149.62, step=0.01)

    with col2:
        st.markdown("**PCA Transformed Features (V1 - V28):**")
        st.caption("Default values represent a typical transaction profile. Adjust as needed.")

    # V features in collapsible expander
    with st.expander("🔬 Advanced PCA Features (V1 to V28)", expanded=False):
        v_cols = {}
        cols = st.columns(4)
        for i in range(1, 29):
            col_idx = (i - 1) % 4
            with cols[col_idx]:
                v_cols[f'V{i}'] = st.number_input(f"V{i}", value=0.0, format="%.4f")

    # Predict Button
    if st.button("🔍 Analyze Transaction for Fraud", type="primary", use_container_width=True):
        # Feature Engineering
        hour_val = (time_val / 3600) % 24
        log_amount_val = np.log1p(amount_val)

        # Build feature dictionary matching training columns order
        input_data = {'Time': time_val}
        for i in range(1, 29):
            input_data[f'V{i}'] = v_cols[f'V{i}']
        input_data['Amount'] = amount_val
        input_data['Hour'] = hour_val
        input_data['Log_Amount'] = log_amount_val

        input_df = pd.DataFrame([input_data])

        # Scale features
        input_scaled = scaler.transform(input_df)

        # Predict probability
        fraud_prob = model.predict_proba(input_scaled)[0][1]
        is_fraud = fraud_prob >= threshold

        st.markdown("---")
        st.subheader("📊 Analysis Results")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.metric(label="Fraud Probability", value=f"{fraud_prob*100:.2f}%")

        with res_col2:
            if is_fraud:
                st.error(f"🚨 **ALERT: Potential Fraudulent Transaction Detected!** (Confidence: {fraud_prob*100:.2f}%)")
                st.warning("Action Recommended: Flag transaction for immediate manual review or block card temporarily.")
            else:
                st.success(f"✅ **Transaction is Legitimate.** (Confidence: {(1-fraud_prob)*100:.2f}%)")
                st.info("Status: Approved for processing.")

        # Probability progress bar
        st.markdown("### Risk Level Meter")
        st.progress(float(fraud_prob))
