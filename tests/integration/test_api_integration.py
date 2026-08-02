import os
import sys
import pytest
from fastapi.testclient import TestClient

# Step out of tests/integration/ to find root app modules cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from app.api.app import app

client = TestClient(app)

def test_api_root_endpoint():
    """Verifies that the base API endpoint returns a valid connection message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "employee attrition prediction project" in response.json()["message"].lower()

def test_prediction_endpoint_with_valid_payload():
    """Verifies that a perfectly structured profile processes through the model matrix successfully."""
    valid_payload = {
        "Employee ID": "TST-999",
        "Age": 35,
        "Years at Company": 4,
        "Monthly Income": 6200.0,
        "Company Tenure": 12,
        "Number of Promotions": 1,
        "Distance from Home": 3.5,
        "Gender": "Male",
        "Job Role": "Technology",
        "Job Level": "Mid",
        "Work-Life Balance": "Good",
        "Job Satisfaction": "High",
        "Performance Rating": "Average",
        "Education Level": "Bachelor’s Degree",
        "Marital Status": "Married",
        "Company Size": "Medium",
        "Remote Work": "Yes",
        "Leadership Opportunities": "No",
        "Innovation Opportunities": "Yes",
        "Company Reputation": "Good",
        "Employee Recognition": "Medium",
        "Overtime": "No"
    }
    
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    
    json_data = response.json()
    assert "employee_id" in json_data
    assert "attrition_status" in json_data
    assert "probability_of_leaving" in json_data
    assert json_data["employee_id"] == "TST-999"

def test_prediction_endpoint_with_invalid_payload_fails():
    """Verifies that bad records fail structural Pydantic validation checks."""
    invalid_payload = {
        "Employee ID": "ERR-111",
        "Age": 12,  # Invalid: Schema constraints require age >= 18
        "Gender": "Male",
        "Years at Company": -5
    }
    
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
