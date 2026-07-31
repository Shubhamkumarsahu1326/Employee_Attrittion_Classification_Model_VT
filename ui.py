import streamlit as st
import requests

st.set_page_config(page_title="HR Attrition Predictor", layout="centered")
st.title("🧑‍💼 Employee Attrition Prediction UI")
st.write("Fill out the employee profile below to calculate retention risk.")

# Fixed backend destination URL string
FASTAPI_URL = "http://127.0.0.1:8000/predict"

with st.form("employee_form"):
    st.subheader("Demographics & Basic Info")
    col1, col2, col3 = st.columns(3)
    with col1:
        emp_id = st.text_input("Employee ID", value="EMP-001")
    with col2:
        age = st.number_input("Age", min_value=18, max_value=60, value=30)
    with col3:
        gender = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Other"])

    st.subheader("Employment Details")
    col4, col5, col6 = st.columns(3)
    with col4:
        job_role = st.selectbox("Job Role", ["Finance", "Healthcare", "Technology", "Education", "Media"])
    with col5:
        job_level = st.selectbox("Job Level", ["Entry", "Mid", "Senior"])
    with col6:
        monthly_income = st.number_input("Monthly Income ($)", min_value=0.0, value=5000.0, step=100.0)

    col7, col8, col9 = st.columns(3)
    with col7:
        years_at_company = st.number_input("Years at Company", min_value=0, value=3)
    with col8:
        company_tenure = st.number_input("Total Industry Tenure (Years)", min_value=0, value=5)
    with col9:
        num_promotions = st.number_input("Number of Promotions", min_value=0, value=0)

    st.subheader("Workplace & Cultural Environment")
    col10, col11, col12 = st.columns(3)
    with col10:
        wlb = st.selectbox("Work-Life Balance", ["Poor", "Below Average", "Good", "Excellent"], index=2)
    with col11:
        satisfaction = st.selectbox("Job Satisfaction", ["Very Low", "Low", "Medium", "High"], index=2)
    with col12:
        perf_rating = st.selectbox("Performance Rating", ["Low", "Below Average", "Average", "High"], index=2)

    col13, col14, col15 = st.columns(3)
    with col13:
        education = st.selectbox("Education Level", ["High School", "Associate Degree", "Bachelor’s Degree", "Master’s Degree", "PhD"], index=2)
    with col14:
        marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    with col15:
        distance = st.number_input("Distance from Home (Miles)", min_value=0.0, value=5.0)

    col16, col17, col18 = st.columns(3)
    with col16:
        comp_size = st.selectbox("Company Size", ["Small", "Medium", "Large"], index=1)
    with col17:
        remote = st.selectbox("Remote Work", ["Yes", "No"], index=1)
    with col18:
        reputation = st.selectbox("Company Reputation", ["Very Poor", "Poor", "Good", "Excellent"], index=2)

    col19, col20 = st.columns(2)
    with col19:
        leadership = st.selectbox("Leadership Opportunities", ["Yes", "No"], index=1)
    with col20:
        innovation = st.selectbox("Innovation Opportunities", ["Yes", "No"], index=1)
        
    recognition = st.selectbox("Employee Recognition", ["Very Low", "Low", "Medium", "High"], index=2)
    overtime = st.selectbox("Overtime", ["Yes", "No"], index=1)
    submit = st.form_submit_button("Predict Attrition Risk")

if submit:
    payload = {
        "Employee ID": emp_id,
        "Age": age,
        "Gender": gender,
        "Years at Company": years_at_company,
        "Monthly Income": monthly_income,
        "Job Role": job_role,
        "Work-Life Balance": wlb,
        "Job Satisfaction": satisfaction,
        "Performance Rating": perf_rating,
        "Number of Promotions": num_promotions,
        "Distance from Home": distance,
        "Education Level": education,
        "Marital Status": marital,
        "Job Level": job_level,
        "Company Size": comp_size,
        "Company Tenure": company_tenure,
        "Remote Work": remote,
        "Leadership Opportunities": leadership,
        "Innovation Opportunities": innovation,
        "Company Reputation": reputation,
        "Employee Recognition": recognition,
        "Overtime": overtime
    }

    try:
        response = requests.post(FASTAPI_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            
            st.write("---")
            st.subheader("Inference Result")
            
            prob = result["probability_of_leaving"]
            status = result["attrition_status"]
            
            if result["attrition_prediction"] == 1:
                st.error(f"⚠️ High Risk: Employee is predicted to **{status}**.")
            else:
                st.success(f"✅ Low Risk: Employee is predicted to **{status}**.")
                
            st.metric(label="Probability of Leaving", value=f"{prob * 100:.2f}%")
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to FastAPI backend. Ensure it is running on port 8000.")

