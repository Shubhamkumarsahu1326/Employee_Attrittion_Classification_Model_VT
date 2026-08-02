import os
import sys
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException

# Import verification schemas
from app.api.schemas import EmployeeDataInput, PredictionResponse

# Connect with your established MLOps tracking tools
from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.s3_storage import S3StorageVault

app = FastAPI(title="Employee Attrition Prediction API with Fixed Ordering Guardrails")

# Resolve model path dependencies dynamically relative to file location
BASE_DIR = Path(__file__).resolve().parents[2]  # Safely step out from app/api/ to mlopstools root directory
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_rf.joblib")
ENCODER_PATH = os.path.join(BASE_DIR, "model", "rf_onehot_encoder.joblib")

model = None
encoder = None
training_categorical_cols = None
s3_vault = None

# Smart Initialization Layer: Loads production files locally, triggers fallback simulation in CI/CD Environments
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        model = joblib.load(MODEL_PATH)
        artifact = joblib.load(ENCODER_PATH)
        encoder = artifact['encoder']
        training_categorical_cols = artifact['categorical_cols']
        logger.info("✅ Pre-trained Production Model and Encoder successfully loaded into memory.")
    else:
        logger.warning(f"⚠️ Warning: Configuration artifacts missing at {MODEL_PATH}. Initializing CI/CD Testing Mock.")
        
        # --- 🛡️ GITHUB ACTIONS CI/CD SIMULATION ENGINE ---
        # Programmatically mocks the structures of your Scikit-Learn training outputs 
        # to allow GitHub's runner to validate end-to-end Pydantic parsing workflows without heavy file weights.
        class MockModel:
            # Captures the exact expected columns layout output by the numeric/one-hot concatenation layer
            feature_names_in_ = [
                'Age', 'Years at Company', 'Monthly Income', 'Company Tenure', 
                'Number of Promotions', 'Distance from Home', 'Number of Dependents',
                'cat_0', 'cat_1', 'cat_2', 'cat_3', 'cat_4', 'cat_5', 'cat_6', 
                'cat_7', 'cat_8', 'cat_9', 'cat_10', 'cat_11', 'cat_12', 'cat_13', 'cat_14'
            ]
            def predict(self, df): return [0]
            def predict_proba(self, df): return [[0.85, 0.15]]
            
        class MockEncoder:
            def transform(self, df): return [[0.0] * 15]
            def get_feature_names_out(self, cols): return [f"cat_{i}" for i in range(15)]
            
        model = MockModel()
        encoder = MockEncoder()
        training_categorical_cols = [
            'Gender', 'Job Role', 'Job Level', 'Work-Life Balance', 'Job Satisfaction', 
            'Performance Rating', 'Education Level', 'Marital Status', 'Company Size', 
            'Remote Work', 'Leadership Opportunities', 'Innovation Opportunities', 
            'Company Reputation', 'Employee Recognition', 'Overtime'
        ]
        # ---------------------------------------------------------------------------------

    # Establish sandbox connection session with your S3 storage engine configuration block
    s3_vault = S3StorageVault()
except Exception as e:
    raise CustomException(e, sys)

@app.get("/")
def home():
    return {'message': "This is an employee attrition prediction project with active cloud auditing logging."}

@app.post("/predict", response_model=PredictionResponse)
def predict_attrition(data: EmployeeDataInput):
    if model is None or encoder is None:
        raise HTTPException(status_code=503, detail="Pipeline classification elements are offline.")
    
    try:
        logger.info("Received prediction request payload.")
        
        # 1. Parse incoming dictionary fields matching exact Pydantic capitalization aliases
        input_dict = data.model_dump(by_alias=True)
        emp_id = input_dict.pop("Employee ID")
        df = pd.DataFrame([input_dict])
        
        # Guard against minor raw dataset structural schema anomalies
        if 'Number of Dependents' not in df.columns:
            df['Number of Dependents'] = 0
            
        # 2. Segment categorical features matching your original training footprint array
        categorical_cols = training_categorical_cols
        numerical_cols = [col for col in df.columns if col not in categorical_cols]
        
        # 3. Transform categorical text metrics using your pre-trained encoder artifact
        encoded_cats = encoder.transform(df[categorical_cols])
        encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
        encoded_cats_df = pd.DataFrame(encoded_cats, columns=encoded_feature_names, index=df.index)
        
        # 4. Construct the initial raw feature matrix layout layout
        final_features_df = pd.concat([df[numerical_cols], encoded_cats_df], axis=1)
        
        # --- 🛡️ THE PERMANENT FIX: CRITICAL COLUMN ALIGNMENT GUARDRAIL ---
        # Fetch the exact feature order your pre-trained model was trained on (.fit)
        if hasattr(model, "feature_names_in_"):
            expected_training_order = model.feature_names_in_
            # Force the dataframe into that exact sequence. Mismatched or missing elements default to 0.
            final_features_df = final_features_df.reindex(columns=expected_training_order, fill_value=0)
        else:
            raise ValueError("Pre-trained model artifact lacks internal 'feature_names_in_' signature matrix records.")
        # -----------------------------------------------------------------
        
        # 5. Execute model inference routines
        prediction = int(model.predict(final_features_df)[0])
        
        if hasattr(model, "predict_proba"):
            prob_matrix = model.predict_proba(final_features_df)
            # Handle standard 2D predict_proba matrix arrays cleanly
            probability = float(prob_matrix[0][1]) if isinstance(prob_matrix[0], (list, pd.Series, bytes, type(pd.Series.values))) or len(prob_matrix[0]) > 1 else float(prob_matrix[0])
        else:
            probability = 1.0 if prediction == 1 else 0.0
            
        status = "Left" if prediction == 1 else "Stayed"
        
        # --- RESILIENT CLOUD AUDITING LOG PATTERN ---
        audit_payload = data.model_dump(by_alias=True)
        audit_payload["predicted_status"] = status
        audit_payload["probability_score"] = probability
        audit_payload["inference_timestamp"] = datetime.now().isoformat()
        
        # Save a backup locally on the drive first
        local_audit_dir = os.path.join(os.getcwd(), "logs", "inference_audit")
        os.makedirs(local_audit_dir, exist_ok=True)
        
        log_filename = f"inference_{emp_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        local_log_path = os.path.join(local_audit_dir, log_filename)
        pd.DataFrame([audit_payload]).to_csv(local_log_path, index=False)
        
        # Attempt cloud upload stream safely
        s3_target_key = f"live_inference_logs/{datetime.now().strftime('%Y/%m/%d')}/{log_filename}"
        if s3_vault and s3_vault.cloud_active:
            s3_vault.upload_file_to_s3(local_log_path, s3_target_key)
            
        logger.info(f"Prediction successful for: {emp_id}. Local audit baseline saved successfully.")
        
        return PredictionResponse(
            employee_id=str(emp_id),
            attrition_prediction=prediction,
            probability_of_leaving=probability,
            attrition_status=status
        )
        
    except Exception as e:
        logger.error(f"Inference pipeline execution failure loop crashed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference pipeline failed: {str(e)}")
