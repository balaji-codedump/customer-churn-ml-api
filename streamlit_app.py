import streamlit as st
import requests

st.set_page_config(page_title="Customer Churn Prediction")

st.title("Customer Churn Prediction")
st.write("ML Application using FastAPI + XGBoost + AWS")

st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
    multiplelines = st.selectbox("Multiple Lines", ["Yes", "No"])
    internetservice = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    onlinesecurity = st.selectbox(
        "Online Security",
        ["Yes", "No"]
    )

    onlinebackup = st.selectbox(
        "Online Backup",
        ["Yes", "No"]
    )

with col2:
    deviceprotection = st.selectbox(
        "Device Protection",
        ["Yes", "No"]
    )

    techsupport = st.selectbox(
        "Tech Support",
        ["Yes", "No"]
    )

    streamingtv = st.selectbox(
        "Streaming TV",
        ["Yes", "No"]
    )

    streamingmovies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperlessbilling = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    paymentmethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

st.subheader("Customer Charges")

tenure = st.number_input("Tenure", min_value=0)

monthlycharges = st.number_input(
    "Monthly Charges",
    min_value=0.0
)

totalcharges = st.number_input(
    "Total Charges",
    min_value=0.0
)

if st.button("Predict Churn"):

    data = {
        "gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phoneservice,
        "MultipleLines": multiplelines,
        "InternetService": internetservice,
        "OnlineSecurity": onlinesecurity,
        "OnlineBackup": onlinebackup,
        "DeviceProtection": deviceprotection,
        "TechSupport": techsupport,
        "StreamingTV": streamingtv,
        "StreamingMovies": streamingmovies,
        "Contract": contract,
        "PaperlessBilling": paperlessbilling,
        "PaymentMethod": paymentmethod,
        "tenure": tenure,
        "MonthlyCharges": monthlycharges,
        "TotalCharges": totalcharges
    }

    response = requests.post(
        "http://51.20.193.189:8000/predict",
        json=data
    )

    result = response.json()

    probability = result["churn_probability"]

    st.subheader("Prediction Result")

    st.metric(
        label="Churn Probability",
        value=f"{probability:.2%}"
    )

    if result["churn_prediction"] == 1:
        st.error("Customer likely to churn")
    else:
        st.success("Customer not likely to churn")