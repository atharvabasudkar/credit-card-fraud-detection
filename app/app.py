import streamlit as st
import joblib
import os
import pandas as pd
import plotly.express as px

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# =========================================
# LOAD MODEL & SCALER
# =========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(
    BASE_DIR,
    "models",
    "xgboost_fraud_model.pkl"
)

scaler_path = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl"
)

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# =========================================
# HEADER
# =========================================

st.title("💳 Credit Card Fraud Detection System")

st.markdown("""
### Business Problem

Credit card fraud causes billions of dollars in losses every year.

The objective of this project is to identify fraudulent transactions
while minimizing false positives and maximizing fraud detection.

This solution uses an XGBoost Machine Learning model trained on highly
imbalanced transaction data.
""")

# =========================================
# MODEL COMPARISON
# =========================================

st.subheader("📊 Model Comparison")

results_df = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],
    "Precision": [
        0.06,
        0.42,
        0.78
    ],
    "Recall": [
        0.92,
        0.85,
        0.84
    ],
    "F1 Score": [
        0.11,
        0.56,
        0.81
    ]
})

st.dataframe(
    results_df,
    use_container_width=True
)

# =========================================
# BEST MODEL
# =========================================

st.success("""
🏆 Selected Model: XGBoost

Reason:
• Highest Precision
• Strong Recall
• Best F1 Score
• Best balance between fraud detection and false alarms
""")

# =========================================
# CSV UPLOAD
# =========================================

st.markdown("---")

st.subheader("📂 Upload Transaction Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing transactions",
    type=["csv"]
)

# =========================================
# FILE PREVIEW
# =========================================

if uploaded_file is not None:

    uploaded_df = pd.read_csv(uploaded_file)

    st.success("✅ File Uploaded Successfully")

    st.write("### Dataset Preview")

    st.dataframe(
        uploaded_df.head(),
        use_container_width=True
    )

    st.write(
        f"Rows: {uploaded_df.shape[0]:,}"
    )

    st.write(
        f"Columns: {uploaded_df.shape[1]}"
    )

    # =========================================
    # FRAUD DETECTION
    # =========================================

    if st.button("🚀 Run Fraud Detection"):

        prediction_df = uploaded_df.copy()

        # Remove target column if present
        if "Class" in prediction_df.columns:
            prediction_df = prediction_df.drop(
                "Class",
                axis=1
            )

        with st.spinner("Running Fraud Detection..."):

            scaled_data = scaler.transform(
                prediction_df
            )

            predictions = model.predict(
                scaled_data
            )

            probabilities = model.predict_proba(
                scaled_data
            )[:, 1]

        uploaded_df["Prediction"] = predictions

        uploaded_df["Fraud_Probability"] = probabilities

        st.success(
            "✅ Fraud Detection Completed"
        )

        # =========================================
        # KPI METRICS
        # =========================================

        total_transactions = len(uploaded_df)

        fraud_transactions = uploaded_df[
            uploaded_df["Prediction"] == 1
        ].shape[0]

        fraud_rate = (
            fraud_transactions /
            total_transactions
        ) * 100

        avg_probability = uploaded_df[
            "Fraud_Probability"
        ].mean()

        st.markdown("---")

        st.subheader("📈 Fraud Detection Summary")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Transactions",
            f"{total_transactions:,}"
        )

        col2.metric(
            "Fraud Detected",
            fraud_transactions
        )

        col3.metric(
            "Fraud Rate (%)",
            f"{fraud_rate:.2f}%"
        )

        col4.metric(
            "Avg Fraud Probability",
            f"{avg_probability:.4f}"
        )

        # =========================================
        # FRAUD TABLE
        # =========================================

        st.markdown("---")

        st.subheader(
            "🚨 High Risk Transactions"
        )

        fraud_cases = uploaded_df[
            uploaded_df["Prediction"] == 1
        ]

        st.dataframe(
            fraud_cases.head(20),
            use_container_width=True
        )

        st.write(
            f"Total Fraudulent Transactions Detected: {fraud_transactions}"
        )

        # =========================================
        # DOWNLOAD RESULTS
        # =========================================

        st.markdown("---")

        csv = uploaded_df.to_csv(
            index=False
        )

        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="fraud_detection_results.csv",
            mime="text/csv"
        )
        # =========================================
        # FEATURE IMPORTANCE DASHBOARD
        # =========================================

        st.markdown("---")

        st.subheader("📊 Feature Importance Analysis")

        feature_path = os.path.join(
            BASE_DIR,
            "outputs",
            "feature_importance.csv"
        )

        feature_df = pd.read_csv(feature_path)

        top_features = feature_df.head(10)

        fig = px.bar(
            top_features,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 10 Features Driving Fraud Detection"
        )

        fig.update_layout(
            height=500
        )

        fig.update_yaxes(
            categoryorder="total ascending"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =========================================
        # FEATURE INSIGHTS
        # =========================================

        st.info(f"""
        🔍 Key Insights

        • Most Important Feature: {feature_df.iloc[0]['Feature']}

        • Importance Score: {feature_df.iloc[0]['Importance']:.4f}

        • Top 3 Fraud Indicators:
          1. {feature_df.iloc[0]['Feature']}
          2. {feature_df.iloc[1]['Feature']}
          3. {feature_df.iloc[2]['Feature']}

        These features contribute the most toward identifying fraudulent transactions.
        """)

        # =========================================
        # EXECUTIVE SUMMARY
        # =========================================

        st.markdown("---")

        st.subheader("💼 Executive Summary")

        st.success(f"""
        Dataset Size: {total_transactions:,} Transactions

        Fraud Transactions Detected: {fraud_transactions}

        Fraud Detection Rate: {fraud_rate:.2f}%

        Selected Model: XGBoost

        Precision: 78%

        Recall: 84%

        F1 Score: 81%

        Business Impact:
        This model helps financial institutions identify suspicious transactions
        before financial loss occurs while maintaining a strong balance between
        fraud detection and false alarms.
        """)

        # =========================================
        # PROJECT HIGHLIGHTS
        # =========================================

        st.markdown("---")

        st.subheader("🚀 Project Highlights")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Models Evaluated",
            "3"
        )

        col2.metric(
            "Best Model",
            "XGBoost"
        )

        col3.metric(
            "Top Feature",
            feature_df.iloc[0]["Feature"]
        )