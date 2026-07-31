import os
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from schemas import EmployeeDataInput, PredictionResponse

app = FastAPI(title="Employee Attrition Prediction API")

# Define paths using your local files (keep them in the same folder as app.py)
MODEL_PATH = r"C:\Users\ASUS\Desktop\mlopstools\.mlopsenv\best_rf.joblib"
ENCODER_PATH =r"C:\Users\ASUS\Desktop\mlopstools\.mlopsenv\rf_onehot_encoder.joblib"



# Initialize global tracking states
model = None
encoder = None
training_categorical_cols = None

if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    model = joblib.load(MODEL_PATH)
    
    # Unpack the dictionary artifact bundle
    artifact = joblib.load(ENCODER_PATH)
    encoder = artifact['encoder']
    training_categorical_cols = artifact['categorical_cols']
    
    print("✅ Model and Dynamic Dictionary Encoder loaded successfully!")
else:
    print("⚠️ Warning: Configuration files missing. Running in simulator mode.")
@app.get("/")
def home():
    return({'message':"This is an employee attrition prediction project"})

@app.post("/predict", response_model=PredictionResponse)
def predict_attrition(data: EmployeeDataInput):
    if model is None or encoder is None:
        raise HTTPException(status_code=503, detail="Pipeline elements are offline.")
    
    try:
        # Convert incoming payload to dict matching dataset string labels
        input_dict = data.model_dump(by_alias=True)
        emp_id = input_dict.pop("Employee ID")
        df = pd.DataFrame([input_dict])
        
        # Guard against the extra feature if it is missing from the frontend form
        if 'Number of Dependents' not in df.columns:
            df['Number of Dependents'] = 0
            
        # 1. Use the exact categorical tracking order straight from your saved artifact
        categorical_cols = training_categorical_cols
        
        # 2. Derive numerical columns by extracting everything that isn't categorical
        numerical_cols = [col for col in df.columns if col not in categorical_cols]
        
        # 3. Transform categorical text metrics using the unpacked encoder instance
        encoded_cats = encoder.transform(df[categorical_cols])
        encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
        encoded_cats_df = pd.DataFrame(encoded_cats, columns=encoded_feature_names, index=df.index)
        
        # 4. Construct the feature matrix matching your original training DataFrame layout
        # Ensure numerical metrics precede the one-hot arrays as expected by the RF architecture
        final_features_df = pd.concat([df[numerical_cols], encoded_cats_df], axis=1)
        
        # 5. Execute model inference
        prediction = int(model.predict(final_features_df)[0])
        
        if hasattr(model, "predict_proba"):
            # Fetch likelihood probability score for index 1 (Leaving)
            probability = float(model.predict_proba(final_features_df)[0][1])
        else:
            probability = 1.0 if prediction == 1 else 0.0
            
        status = "Left" if prediction == 1 else "Stayed"
        
        return PredictionResponse(
            employee_id=emp_id,
            attrition_prediction=prediction,
            attrition_status=status,
            probability_of_leaving=probability
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline failed: {str(e)}")

