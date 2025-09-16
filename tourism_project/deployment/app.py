import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# ---------------------------
# Load Model from Hugging Face
# ---------------------------
model_path = hf_hub_download(
    repo_id = "Bhanu15/Tourism-Package-Prediction-MLOps",
    filename="best_tourism_xgb_model_v1.joblib",
    repo_type="model"
)
model = joblib.load(model_path)

st.set_page_config(page_title="Tourism Package Prediction", layout="centered")

# ---------------------------
# App Header
# ---------------------------
st.title("🏖️ Tourism Package Purchase Prediction")
st.write("""
This app predicts whether a customer is likely to purchase a **Wellness Tourism Package**.
Please fill in the customer details below to get a prediction.
""")

# ---------------------------
# User Inputs
# ---------------------------

# Demographics
st.subheader("👤 Customer Information")
age = st.slider("Age", 18, 80, 30)
gender = st.radio("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Other"])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Student", "Unemployed", "Retired", "Other"])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP", "Other"])
monthly_income = st.number_input("Monthly Income (₹)", min_value=5000, max_value=500000, value=50000, step=500)

# Contact & Travel Behavior
st.subheader("📞 Contact & Travel Behavior")
typeof_contact = st.radio("Type of Contact", ["Company Invited", "Self Enquiry"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.slider("Duration of Pitch (minutes)", 1, 60, 15)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Super Deluxe", "King", "Queen", "Other"])
pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)  

# Family & Lifestyle
st.subheader("🏠 Family & Lifestyle")
family_members = st.slider("Number of Family Members", 1, 10, 3)
children_visiting = st.slider("Number of Children Visiting", 0, 5, 0)
own_car = st.radio("Owns a Car?", ["Yes", "No"])
passport = st.radio("Passport Available?", ["Yes", "No"])

# Travel History
st.subheader("🛫 Travel History")
number_of_trips = st.slider("Number of Trips Last Year", 0, 20, 2)
num_followups = st.slider("Number of Follow-ups", 0, 10, 2)
num_visitors = st.slider("Number of Persons Visiting", 1, 10, 2)
preferred_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])

# ---------------------------
# Assemble input into DataFrame
# ---------------------------
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeof_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_visitors,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": pitch_satisfaction
}])

# ---------------------------
# Prediction Button
# ---------------------------
if st.button("🔮 Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    result = "✅ Likely to Purchase Package" if prediction == 1 else "❌ Not Likely to Purchase Package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
    st.info(f"Probability of purchasing: {probability:.2%}")
