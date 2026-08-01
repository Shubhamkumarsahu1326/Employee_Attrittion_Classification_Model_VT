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

app = FastAPI(title="Employee Attrition Prediction API with Cloud Auditing")

# Resolve model path dependencies dynamically relative to file location
BASE_DIR = Path(__file__).resolve().parents[2]  # Targets mlopstools root from app/api/app.py
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_rf.joblib")
ENCODER_PATH = os.path.join(BASE_DIR, "model", "rf_onehot_encoder.joblib")

model = None
encoder = None
training_categorical_cols = None
s3_vault = None

# Initialize core assets and establish secure AWS S3 connection session
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        model = joblib.load(MODEL_PATH)
        artifact = joblib.load(ENCODER_PATH)
        encoder = artifact['encoder']
        training_categorical_cols = artifact['categorical_cols']
        logger.info("✅ Pre-trained Model and Encoder successfully loaded into memory.")
    else:
        logger.warning(f"⚠️ Warning: Configuration artifacts missing at {MODEL_PATH}.")

    # Establish handshake connection session with your AWS S3 bucket
    s3_vault = S3StorageVault()
    logger.info("✅ Core Ingestion Server connected securely to S3 storage bucket cloud tier.")
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
        
        # 1. Dump data with original field aliases so it matches what your encoder expects
        input_dict = data.model_dump(by_alias=True)
        emp_id = input_dict.pop("Employee ID")
        df = pd.DataFrame([input_dict])
        
        # Guard against minor raw dataset layout anomalies 
        if 'Number of Dependents' not in df.columns:
            df['Number of Dependents'] = 0
            
        # 2. Transform categorical features using your saved encoder artifact
        categorical_cols = training_categorical_cols
        numerical_cols = [col for col in df.columns if col not in categorical_cols]
        
        encoded_cats = encoder.transform(df[categorical_cols])
        encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
        encoded_cats_df = pd.DataFrame(encoded_cats, columns=encoded_feature_names, index=df.index)
        
        # 3. Construct the merged raw feature matrix layout
        final_features_df = pd.concat([df[numerical_cols], encoded_cats_df], axis=1)
        
        # --- FIXED STEP: DYNAMICALLY REORDER COLUMNS TO MATCH TRAINING ---
        # Fetch the exact column sequence array that your pre-trained model was trained on
        if hasattr(model, "feature_names_in_"):
            expected_order = model.feature_names_in_
            # Reindex instantly re-orders columns to match the training layout precisely
            final_features_df = final_features_df.reindex(columns=expected_order)
        # -----------------------------------------------------------------
        
        # 4. Execute pre-trained model inference routines
        prediction = int(model.predict(final_features_df)[0])
        
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(final_features_df)[0][1])
        else:
            probability = 1.0 if prediction == 1 else 0.0
            
        status = "Left" if prediction == 1 else "Stayed"
        
        # --- CLOUD AUDITING PATTERN: SAVE AUDIT ENTRY TO AWS S3 ---        # --- RESILIENT AUDITING PATTERN ---
        audit_payload = data.model_dump(by_alias=True)
        audit_payload["predicted_status"] = status
        audit_payload["probability_score"] = probability
        audit_payload["inference_timestamp"] = datetime.now().isoformat()
        
        # 1. Always save a copy inside a dedicated local audit folder
        local_audit_dir = os.path.join(os.getcwd(), "logs", "inference_audit")
        os.makedirs(local_audit_dir, exist_ok=True)
        
        log_filename = f"inference_{emp_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        local_log_path = os.path.join(local_audit_dir, log_filename)
        pd.DataFrame([audit_payload]).to_csv(local_log_path, index=False)
        
        # 2. Attempt S3 upload. If it throws a signature error, it will log it and continue smoothly!
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

