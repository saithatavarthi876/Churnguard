
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "models" / "best_model.joblib"



st.set_page_config(
    page_title="ChurnGuard | Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
#  LOAD THE TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}"
        )

    return joblib.load(MODEL_FILE)


try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()




st.title("📊 ChurnGuard")
st.subheader("Customer Churn Prediction System")

st.write(
    "Enter customer details below to predict whether "
    "the customer is likely to leave the company."
)

st.divider()


# --------------------------------------------------
# 5. CUSTOMER INPUT FORM
# --------------------------------------------------

with st.form("customer_form"):

    st.header("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        partner = st.selectbox(
            "Has Partner?",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Has Dependents?",
            ["Yes", "No"]
        )

    with col2:
        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

    with col3:
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

    st.divider()

    st.header("Subscribed Services")

    col1, col2, col3 = st.columns(3)

    with col1:
        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

    with col2:
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

    with col3:
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=70.0,
            step=1.0
        )

        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=840.0,
            step=1.0
        )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Predict Churn",
        use_container_width=True
    )


# --------------------------------------------------
# 6. MAKE PREDICTION
# --------------------------------------------------

if submitted:

    # Keep the input columns consistent with training.
    customer_data = pd.DataFrame([{
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
    }])

    try:
        prediction = model.predict(customer_data)[0]

        probabilities = model.predict_proba(customer_data)[0]

        # Find the probability corresponding to class 1 (Churn).
        classes = list(model.classes_)
        churn_index = classes.index(1)

        churn_probability = probabilities[churn_index]
        stay_probability = 1 - churn_probability

        st.divider()
        st.header("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Churn Probability",
                f"{churn_probability * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Stay Probability",
                f"{stay_probability * 100:.2f}%"
            )

        st.progress(float(churn_probability))

        if prediction == 1:
            st.error(
                "⚠️ The model predicts that this customer "
                "is likely to churn."
            )
        else:
            st.success(
                "✅ The model predicts that this customer "
                "is likely to stay."
            )

        st.caption(
            "These are model predictions, not guaranteed "
            "customer outcomes."
        )

        with st.expander("View submitted customer details"):
            st.dataframe(
                customer_data,
                use_container_width=True
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
