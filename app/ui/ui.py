import os
import streamlit as st
import requests

# DYNAMIC NETWORK ENGINE: Uses Docker container routing inside the mesh, falls back to local port if standalone
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0")

st.set_page_config(
    page_title="HR Employee Attrition Predictor",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Employee Attrition Prediction Engine")
st.markdown("Enter employee demographic and performance metrics to evaluate turnover and retention probability risk metrics.")

st.markdown("---")

# PRESENTATION FORMS SYSTEM
with st.form("attrition_form"):
    st.subheader("📋 Employee Profile Attributes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        employee_id = st.text_input("Employee ID", value="EMP-001")
        age = st.slider("Age", min_value=18, max_value=65, value=30)
        gender = st.selectbox("Gender", options=["Male", "Female", "Non-binary", "Other"])
        marital_status = st.selectbox("Marital Status", options=["Single", "Married", "Divorced"])
        education_level = st.selectbox("Education Level", options=["High School", "Associate Degree", "Bachelor’s Degree", "Master’s Degree", "PhD"])
        distance_from_home = st.number_input("Distance from Home (Miles)", min_value=0.0, max_value=100.0, value=5.0)
        
    with col2:
        monthly_income = st.number_input("Monthly Income ($)", min_value=0.0, value=5000.0)
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=45, value=5)
        company_tenure = st.number_input("Company Tenure (Total Months)", min_value=0, value=8)
        num_promotions = st.number_input("Number of Promotions", min_value=0, value=2)
        job_role = st.selectbox("Job Role", options=["Finance", "Healthcare", "Technology", "Education", "Media"])
        job_level = st.selectbox("Job Level", options=["Entry", "Mid", "Senior"])

    st.markdown("---")
    st.subheader("📊 Performance & Cultural Workplace Settings")
    
    col3, col4 = st.columns(2)
    
    with col3:
        work_life_balance = st.selectbox("Work-Life Balance", options=["Poor", "Below Average", "Good", "Excellent"], index=2)
        job_satisfaction = st.selectbox("Job Satisfaction", options=["Very Low", "Low", "Medium", "High"])
        performance_rating = st.selectbox("Performance Rating", options=["Low", "Below Average", "Average", "High"], index=3)
        overtime = st.selectbox("Overtime", options=["Yes", "No"], index=1)
        
    with col4:
        company_size = st.selectbox("Company Size", options=["Small", "Medium", "Large"], index=0)
        remote_work = st.selectbox("Remote Work", options=["Yes", "No"], index=1)
        leadership_opportunities = st.selectbox("Leadership Opportunities", options=["Yes", "No"], index=1)
        innovation_opportunities = st.selectbox("Innovation Opportunities", options=["Yes", "No"], index=1)
        company_reputation = st.selectbox("Company Reputation", options=["Very Poor", "Poor", "Good", "Excellent"], index=2)
        employee_recognition = st.selectbox("Employee Recognition", options=["Very Low", "Low", "Medium", "High"], index=2)

    # Submission anchor
    submit_button = st.form_submit_button(label="Predict Attrition Risk")

# TRANSACTION DATA ROUTING ENGINE
if submit_button:
    # --- 🛡️ FIXED SEQUENCE DICTIONARY MAPPING FOREGROUND GUARDRAIL ---
    # The fields are explicitly ordered to ensure the parser presents numerical items
    # preceding categorical arrays exactly matching your training matrix footprint.
    payload = {
        "Employee ID": str(employee_id),
        "Age": int(age),
        "Years at Company": int(years_at_company),
        "Monthly Income": float(monthly_income),
        "Company Tenure": int(company_tenure),
        "Number of Promotions": int(num_promotions),
        "Distance from Home": float(distance_from_home),
        "Gender": str(gender),
        "Job Role": str(job_role),
        "Job Level": str(job_level),
        "Work-Life Balance": str(work_life_balance),
        "Job Satisfaction": str(job_satisfaction),
        "Performance Rating": str(performance_rating),
        "Education Level": str(education_level),
        "Marital Status": str(marital_status),
        "Company Size": str(company_size),
        "Remote Work": str(remote_work),
        "Leadership Opportunities": str(leadership_opportunities),
        "Innovation Opportunities": str(innovation_opportunities),
        "Company Reputation": str(company_reputation),
        "Employee Recognition": str(employee_recognition),
        "Overtime": str(overtime)
    }
    
    with st.spinner("Transmitting request metrics payload to inference processor container..."):
        try:
            response = requests.post(BACKEND_URL, json=payload, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                st.markdown("### 📈 Evaluation Risk Assessment Results")
                
                status = result["attrition_status"]
                prob = result["probability_of_leaving"] * 100
                
                # Visual output formatting cards
                if status == "Left" or status == "Yes":
                    st.error(f"🚨 **High Turnover Risk Detected**\n\nPrediction Status: **{status}**\n\nProbability of leaving the enterprise: **{prob:.2f}%**")
                else:
                    st.success(f"✅ **Stable Retention Signature Verified**\n\nPrediction Status: **{status}**\n\nProbability of leaving the enterprise: **{prob:.2f}%**")
            else:
                st.error(f"❌ Server Error (Code {response.status_code}): {response.text}")
                
        except Exception as e:
            st.error(f"🔌 Connection Interrupted. Unable to locate the API backend node.")
            st.info(f"Target Endpoint attempted: **{BACKEND_URL}**\n\nError Context: {str(e)}")
