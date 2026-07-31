from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

class EmployeeDataInput(BaseModel):
    # Enable alias population using Pydantic V2 model_config
    model_config = ConfigDict(populate_by_name=True)

    employee_id: str = Field(..., alias="Employee ID", description="Unique identifier")
    age: int = Field(..., alias="Age", ge=18, le=60)
    gender: Literal["Male", "Female", "Non-binary", "Other"] = Field(..., alias="Gender")
    years_at_company: int = Field(..., alias="Years at Company", ge=0)
    monthly_income: float = Field(..., alias="Monthly Income", ge=0)
    job_role: Literal["Finance", "Healthcare", "Technology", "Education", "Media"] = Field(..., alias="Job Role")
    work_life_balance: Literal["Poor", "Below Average", "Good", "Excellent"] = Field(..., alias="Work-Life Balance")
    job_satisfaction: Literal["Very Low", "Low", "Medium", "High"] = Field(..., alias="Job Satisfaction")
    performance_rating: Literal["Low", "Below Average", "Average", "High"] = Field(..., alias="Performance Rating")
    number_of_promotions: int = Field(..., alias="Number of Promotions", ge=0)
    distance_from_home: float = Field(..., alias="Distance from Home", ge=0)
    education_level: Literal["High School", "Associate Degree", "Bachelor’s Degree", "Master’s Degree", "PhD"] = Field(..., alias="Education Level")
    marital_status: Literal["Divorced", "Married", "Single"] = Field(..., alias="Marital Status")
    job_level: Literal["Entry", "Mid", "Senior"] = Field(..., alias="Job Level")
    company_size: Literal["Small", "Medium", "Large"] = Field(..., alias="Company Size")
    company_tenure: int = Field(..., alias="Company Tenure", ge=0)
    remote_work: Literal["Yes", "No"] = Field(..., alias="Remote Work")
    leadership_opportunities: Literal["Yes", "No"] = Field(..., alias="Leadership Opportunities")
    innovation_opportunities: Literal["Yes", "No"] = Field(..., alias="Innovation Opportunities")
    company_reputation: Literal["Very Poor", "Poor", "Good", "Excellent"] = Field(..., alias="Company Reputation")
    employee_recognition: Literal["Very Low", "Low", "Medium", "High"] = Field(..., alias="Employee Recognition")
    overtime: Literal["Yes", "No"] = Field(..., alias="Overtime")

class PredictionResponse(BaseModel):
    employee_id: str
    attrition_prediction: int = Field(..., description="0 for Stayed, 1 for Left")
    attrition_status: str = Field(..., description="'Stayed' or 'Left'")
    probability_of_leaving: float = Field(..., description="Probability score between 0 and 1")


