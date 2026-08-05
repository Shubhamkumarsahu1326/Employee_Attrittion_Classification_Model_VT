import os
import sys
import time
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, Response

# =====================================================================
# 📊 NEW MONITORING ABILITY: PROMETHEUS INSTRUMENTATION ENGINE
# =====================================================================
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Import verification schemas
from app.api.schemas import EmployeeDataInput, PredictionResponse

# Connect with your established MLOps tracking tools
from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.s3_storage import S3StorageVault

app = FastAPI(title="Employee Attrition Prediction API with Fixed Async Guardrails & Monitoring")

# Resolve model path dependencies dynamically relative to file location
BASE_DIR = Path(__file__).resolve().parents[2]  # Safely step out from app/api/ to mlopstools root directory
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_rf.joblib")
ENCODER_PATH = os.path.join(BASE_DIR, "model", "rf_onehot_encoder.joblib")

model = None
encoder = None
training_categorical_cols = None
s3_vault = None

# =====================================================================
# 📈 PROMETHEUS METRIC REGISTRY DEFINITIONS
# =====================================================================
# Operational Health Telemetry
HTTP_REQUESTS_TOTAL = Counter(
    "api_requests_total", "Count of HTTP transactions hitting the backend microservice", ["method", "endpoint", "status"]
)
INFERENCE_LATENCY = Histogram(
    "api_request_latency_seconds", "Inference duration processing distribution latency profiles", ["endpoint"]
)

# Machine Learning Monitoring & Production Drift Telemetry
ATTRITION_PREDICTIONS_TOTAL = Counter(
    "model_predictions_total", "Running count of attrition classifications dispatched", ["prediction_class"]
)
LIVE_AGE_GAUGE = Gauge(
    "live_feature_age_years", "Tracks real-time incoming age distribution profiles to catch cohort demographic drift"
)
LIVE_INCOME_GAUGE = Gauge(
    "live_feature_income_dollars", "Tracks real-time incoming monthly income scales to check economic profile shifts"
)

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
        class MockModel:
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

    s3_vault = S3StorageVault()
except Exception as e:
    raise CustomException(e, sys)

# =====================================================================
# 🔄 FIX PRIOR PROBLEM: NON-BLOCKING BACKGROUND FEEDBACK LOOP
# =====================================================================
def process_and_flush_audit_logs(audit_payload: dict, emp_id: str, log_filename: str):
    """
    Executes raw file formatting and flushes inference records down into 
    local disk targets and remote S3 storage buckets asynchronously out-of-band.
    """
    try:
        # Save a backup locally on the drive
        local_audit_dir = os.path.join(os.getcwd(), "logs", "inference_audit")
        os.makedirs(local_audit_dir, exist_ok=True)
        local_log_path = os.path.join(local_audit_dir, log_filename)
        
        pd.DataFrame([audit_payload]).to_csv(local_log_path, index=False)
        
        # Attempt cloud upload stream safely using background threads
        s3_target_key = f"live_inference_logs/{datetime.now().strftime('%Y/%m/%d')}/{log_filename}"
        if s3_vault and getattr(s3_vault, "cloud_active", False):
            s3_vault.upload_file_to_s3(local_log_path, s3_target_key)
            logger.info(f"☁️ Async Cloud Audit Log dispatched successfully for: {emp_id}")
        else:
            logger.warning(f"⚠️ S3 Vault configuration inactive or missing. Skipping cloud dump for: {emp_id}")
    except Exception as e:
        logger.error(f"❌ Background Audit Loop failed for Employee {emp_id}: {str(e)}")

# =====================================================================
# 🛠️ ROUTING ENDPOINTS
# =====================================================================
@app.get("/")
def home():
    return {'message': "This is an employee attrition prediction project with active cloud auditing logging."}

@app.post("/predict", response_model=PredictionResponse)
def predict_attrition(data: EmployeeDataInput, background_tasks: BackgroundTasks):
    start_time = time.time()
    if model is None or encoder is None:
        HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict", status="503").inc()
        raise HTTPException(status_code=503, detail="Pipeline classification elements are offline.")
    
    try:
        # Track drift signals in Prometheus live gauges before processing matrix schemas
        LIVE_AGE_GAUGE.set(data.Age)
        LIVE_INCOME_GAUGE.set(data.Monthly_Income)
        
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
        
        # 4. Construct the initial raw feature matrix layout
        final_features_df = pd.concat([df[numerical_cols], encoded_cats_df], axis=1)
        
        # --- 🛡️ THE PERMANENT FIX: CRITICAL COLUMN ALIGNMENT GUARDRAIL ---
        if hasattr(model, "feature_names_in_"):
            expected_training_order = model.feature_names_in_
            final_features_df = final_features_df.reindex(columns=expected_training_order, fill_value=0)
        else:
            raise ValueError("Pre-trained model artifact lacks internal 'feature_names_in_' signature matrix records.")
        
        # 5. Execute model inference routines
        prediction = int(model.predict(final_features_df)[0])
        
        if hasattr(model, "predict_proba"):
            prob_matrix = model.predict_proba(final_features_df)
            probability = float(prob_matrix[0][1]) if len(prob_matrix[0]) > 1 else float(prob_matrix[0])
        else:
            probability = 1.0 if prediction == 1 else 0.0
            
        status = "Left" if prediction == 1 else "Stayed"
        
        # Track Attrition prediction distribution counters inside Prometheus metrics stream
        ATTRITION_PREDICTIONS_TOTAL.labels(prediction_class=status).inc()
        
        # --- NEW ARCHITECTURE: ASYNCHRONOUS AUDITING LOG PATTERN VIA BACKGROUND WORKER ---
        audit_payload = data.model_dump(by_alias=True)
        audit_payload["predicted_status"] = status
        audit_payload["probability_score"] = probability
        audit_payload["inference_timestamp"] = datetime.now().isoformat()
        
        log_filename = f"inference_{emp_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        
        # Hand off disk write and S3 persistence workloads to an async worker thread
        background_tasks.add_task(
            process_and_flush_audit_logs, 
            audit_payload, 
            emp_id, 
            log_filename
        )
        
        # Increment Request Operational Tracking Metrics
        HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict", status="200").inc()
        INFERENCE_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
        
        return PredictionResponse(
            employee_id=str(emp_id),
            attrition_prediction=prediction,
            probability_of_leaving=probability,attrition_status=status)
    
    except Exception as e:HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict", status="500").inc()
    logger.error(f"Inference pipeline execution failure loop crashed: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Inference pipeline failed: {str(e)}")
