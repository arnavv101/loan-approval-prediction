import streamlit as st
import joblib
import pandas as pd
import shap

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide"
)

# -----------------------------
# Load Saved Files
# -----------------------------
model = joblib.load("ngboost_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")
shap_background = joblib.load("shap_background.pkl")

explainer = shap.Explainer(
    model.predict_proba,
    shap_background
) 
# -----------------------------
# Title
# -----------------------------
st.title("🏦 Loan Approval Prediction System")

st.write(
    "This application predicts whether a loan application is likely to be approved using the trained NGBoost model."
)
st.subheader("📋 Applicant Information")

col1, col2 = st.columns(2)

with col1:
    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0.0,
        value=5000.0,
        step=500.0
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0.0,
        value=0.0,
        step=500.0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=1.0,
        value=150.0,
        step=10.0
    )

with col2:
    loan_term = st.number_input(
        "Loan Amount Term",
        min_value=12.0,
        value=360.0,
        step=12.0
    )

    credit_history = st.selectbox(
        "Credit History",
        [1, 0],
        format_func=lambda x: "Good" if x == 1 else "Bad"
    )
    
    
st.subheader("👤 Personal & Property Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "Married",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["0", "1", "2", "3+"]
    )

with col2:
    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )

with col3:
    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )
st.subheader("🔍 Loan Prediction")

if st.button("Predict Loan Approval", type="primary"):

    # Encode categorical variables
    gender_value = 1 if gender == "Male" else 0
    married_value = 1 if married == "Yes" else 0
    education_value = 1 if education == "Graduate" else 0
    self_employed_value = 1 if self_employed == "Yes" else 0
    dependents_value = int(dependents)

    # Feature engineering
    total_income = applicant_income + coapplicant_income

    income_per_loan = total_income / loan_amount

    # Create input DataFrame
    input_data = pd.DataFrame([{
        "Gender": gender_value,
        "Married": married_value,
        "Dependents": dependents_value,
        "Education": education_value,
        "Self_Employed": self_employed_value,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "TotalIncome": total_income,
        "IncomePerLoan": income_per_loan,
        "Property_Area_Rural": 1 if property_area == "Rural" else 0,
        "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
        "Property_Area_Urban": 1 if property_area == "Urban" else 0
    }])

    # Ensure exact feature order
    input_data = input_data[features]

    # Scale the input exactly as during training
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Get probabilities
    probabilities = model.predict_proba(input_scaled)[0]

    approval_probability = probabilities[1]
    # -----------------------------
    # Prediction Result
    # -----------------------------
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.metric(
        label="Approval Probability",
        value=f"{approval_probability * 100:.2f}%"
    )

    st.progress(
        float(approval_probability),
        text=f"Approval Confidence: {approval_probability * 100:.2f}%"
    )
        # Generate SHAP explanation
    shap_result = explainer(input_scaled)

    # Select SHAP values for the Approved class
    shap_exp = shap.Explanation(
        values=shap_result.values[0, :, 1],
        base_values=shap_result.base_values[0, 1],
        data=input_scaled[0],
        feature_names=features
    )
    
    st.subheader("🧠 Why did the model make this decision?")

    st.pyplot(
        shap.plots.waterfall(
            shap_exp,
            show=False
        ).figure
    )