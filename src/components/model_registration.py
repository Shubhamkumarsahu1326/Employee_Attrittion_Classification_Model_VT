import os
import sys
import yaml
import joblib
import mlflow
from src.utils.logger import logger
from src.utils.exception import CustomException

class ModelRegistration:
    def __init__(self):
        # Use clean, simple relative paths to match your working test_mlflow file
        config_path = "config/config.yaml"
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file missing at: {os.path.abspath(config_path)}")
            
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
            
        self.registration_config = self.config["artifacts"]["model_trainer"]
        self.mlflow_config = self.config["mlflow"]

    def register_pre_trained_artifacts(self):
        logger.info("Initiating Step 3: MLflow Model Registration workflow.")
        print("🚀 Starting registration loop...")
        try:
            # Map configuration paths directly
            model_src = self.registration_config["model_path"]
            encoder_src = self.registration_config["encoder_path"]

            print(f"Checking for model file at: {os.path.abspath(model_src)}")
            if not os.path.exists(model_src) or not os.path.exists(encoder_src):
                raise FileNotFoundError(
                    f"Pre-trained artifacts missing! Ensure they exist in your model/ directory."
                )

            # Load the artifacts to confirm integrity
            model = joblib.load(model_src)
            print("✅ Pre-trained model binary loaded successfully into memory.")

            # Connect to MLflow Instance
            mlflow.set_tracking_uri(self.mlflow_config["tracking_uri"])
            mlflow.set_experiment(self.mlflow_config["experiment_name"])
            print(f"Connecting to MLflow Tracking Server at: {self.mlflow_config['tracking_uri']}")

            # Start Registration Session
            with mlflow.start_run(run_name="Pre_Trained_RF_Baseline") as run:
                print(f"⚡ MLflow session open. Active Run ID: {run.info.run_id}")

                # Log training hyperparameters from config parameters
                rf_params = self.config["model_params"]["RandomForest"]
                mlflow.log_params(rf_params)
                print("Logged hyperparameter configurations.")

                # Log the encoder bundle as a generic workflow asset file
                mlflow.log_artifact(encoder_src, artifact_path="preprocessing_transforms")
                print("Logged data encoder transform bundle.")

                # Register the core Random Forest Scikit-Learn Model
                print("Registering model binaries into MLflow registry vault (this may take a few seconds)...")
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model_binaries",
                    registered_model_name="HR_Attrition_RF_Model"
                )
                
                print(f"🎉 SUCCESS! Step 3 Complete.")
                print(f"📁 Experiment Name: {self.mlflow_config['experiment_name']}")
                print(f"🔗 Registered Run ID: {run.info.run_id}")

        except Exception as e:
            print(f"❌ Execution crashed inside logic loop! Detail: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    registrar = ModelRegistration()
    registrar.register_pre_trained_artifacts()
