
import streamlit as st
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

model = tf.keras.models.load_model("customer_churn_ann.keras")
preprocessor = joblib.load("customer_churn_preprocessor.pkl")

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

st.title("📊 Telecom Customer Churn Analytics")

st.markdown(
    """
    Predict customer churn using an Artificial Neural Network.

    This application analyzes customer demographics, subscribed services,
    contract details and billing information to estimate churn probability.
    """
)

st.divider()

col1, col2, col3 = st.columns(3)

col1.metric("Model", "Artificial Neural Network")
col2.metric("Input Features", "19")
col3.metric("Model AUC", "84.12%")

st.divider()

st.success("ANN Model and preprocessing pipeline loaded successfully!")

st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with col2:
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0
    )

st.divider()

if st.button("🔍 Predict Customer Churn", use_container_width=True):

    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    input_df = pd.DataFrame([customer_data])

    processed_data = preprocessor.transform(input_df)

    prediction_probability = float(
        model.predict(processed_data, verbose=0)[0][0]
    )

    prediction = 1 if prediction_probability >= 0.5 else 0

    if prediction_probability < 0.3:
        risk_level = "Low"
    elif prediction_probability <= 0.6:
        risk_level = "Medium"
    else:
        risk_level = "High"

    result = "Churn" if prediction == 1 else "Stay"

    st.session_state.prediction_history.append({
        "Prediction": len(st.session_state.prediction_history) + 1,
        "Churn Probability": round(
            prediction_probability * 100, 2
        ),
        "Risk Level": risk_level,
        "Result": result
    })

    st.subheader("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{prediction_probability * 100:.2f}%"
    )

    st.progress(prediction_probability)

    if risk_level == "Low":
        st.success("LOW RISK - Customer is likely to stay.")
        st.info("Recommendation: Maintain customer satisfaction.")

    elif risk_level == "Medium":
        st.warning("MEDIUM RISK - Customer may churn.")
        st.info(
            "Recommendation: Monitor customer engagement and offer support."
        )

    else:
        st.error("HIGH RISK - Customer is likely to churn.")
        st.info(
            "Recommendation: Consider a personalized retention offer."
        )

st.divider()

st.header("Prediction History")

if st.session_state.prediction_history:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    st.metric(
        "Total Predictions",
        len(history_df)
    )

    st.divider()

    st.header("Prediction Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Risk Distribution")

        risk_counts = (
            history_df["Risk Level"]
            .value_counts()
            .reindex(["Low", "Medium", "High"], fill_value=0)
        )

        st.bar_chart(risk_counts)

    with col2:
        st.subheader("Churn Probability by Prediction")

        probability_data = history_df.set_index("Prediction")[
            ["Churn Probability"]
        ]

        st.line_chart(probability_data)

    st.subheader("Prediction Summary")

    total = len(history_df)

    churn_count = (
        history_df["Result"] == "Churn"
    ).sum()

    stay_count = (
        history_df["Result"] == "Stay"
    ).sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", total)
    col2.metric("Predicted Churn", int(churn_count))
    col3.metric("Predicted Stay", int(stay_count))

    st.divider()

    st.header("Export Prediction Data")

    csv_data = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction History",
        data=csv_data,
        file_name="customer_churn_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    st.info("No predictions made yet. Enter customer details and click Predict.")
