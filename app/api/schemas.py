from pydantic import BaseModel, Field
from typing import Literal

class EmployeeDataInput(BaseModel):
    # Match the exact casing and spacing expected by your pre-trained model artifacts
    Employee_ID: str = Field(..., alias="Employee ID")
    Age: int = Field(..., ge=18, le=65)
    Gender: Literal["Male", "Female", "Non-binary", "Other"]
    
    # Employment Details
    Years_at_Company: int = Field(..., alias="Years at Company", ge=0)
    Monthly_Income: float = Field(..., alias="Monthly Income", ge=0.0)
    Job_Role: Literal["Finance", "Healthcare", "Technology", "Education", "Media"] = Field(..., alias="Job Role")
    Job_Level: Literal["Entry", "Mid", "Senior"] = Field(..., alias="Job Level")
    Company_Tenure: int = Field(..., alias="Company Tenure", ge=0)
    Number_of_Promotions: int = Field(..., alias="Number of Promotions", ge=0)
    
    # Workplace & Cultural Environment
    Work_Life_Balance: Literal["Poor", "Below Average", "Good", "Excellent"] = Field(..., alias="Work-Life Balance")
    Job_Satisfaction: Literal["Very Low", "Low", "Medium", "High"] = Field(..., alias="Job Satisfaction")
    Performance_Rating: Literal["Low", "Below Average", "Average", "High"] = Field(..., alias="Performance Rating")
    Distance_from_Home: float = Field(..., alias="Distance from Home", ge=0.0)
    Education_Level: Literal["High School", "Associate Degree", "Bachelor’s Degree", "Master’s Degree", "PhD"] = Field(..., alias="Education Level")
    Marital_Status: Literal["Single", "Married", "Divorced"] = Field(..., alias="Marital Status")
    Company_Size: Literal["Small", "Medium", "Large"] = Field(..., alias="Company Size")
    Remote_Work: Literal["Yes", "No"] = Field(..., alias="Remote Work")
    Leadership_Opportunities: Literal["Yes", "No"] = Field(..., alias="Leadership Opportunities")
    Innovation_Opportunities: Literal["Yes", "No"] = Field(..., alias="Innovation Opportunities")
    Company_Reputation: Literal["Very Poor", "Poor", "Good", "Excellent"] = Field(..., alias="Company Reputation")
    Employee_Recognition: Literal["Very Low", "Low", "Medium", "High"] = Field(..., alias="Employee Recognition")
    Overtime: Literal["Yes", "No"]

    # Native configuration to accept human-readable inputs from your UI seamlessly
    model_config = {
        "populate_by_name": True,
        "extra": "ignore"
    }

class PredictionResponse(BaseModel):
    employee_id: str
    attrition_prediction: int
    probability_of_leaving: float
    attrition_status: str

