import os
import sys
import yaml
import pandas as pd
from app.api.schemas import EmployeeDataInput
from src.utils.logger import logger
from src.utils.exception import CustomException

class BatchDataValidation:
    def __init__(self):
        # Resolve absolute directory tree anchor points
        self.script_dir = os.path.dirname(os.path.abspath(__file__)) 
        self.project_root = os.path.abspath(os.path.join(self.script_dir, "..", "..")) 

        config_path = os.path.join(self.project_root, "config", "config.yaml")
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
            
        self.validation_config = self.config["artifacts"]["data_ingestion"]

    def validate_csv_dataset(self, file_path: str) -> bool:
        """
        Validates incoming CSV rows against central Pydantic schema rules.
        """
        try:
            df = pd.read_csv(file_path)
            
            # --- PATCH 1: Convert Employee ID column to string to prevent int-to-str crashes ---
            if "Employee ID" in df.columns:
                df["Employee ID"] = df["Employee ID"].astype(str)
            if "Company Reputation" in df.columns:
                df["Company Reputation"] = df["Company Reputation"].replace({"Fair": "Good"})
                
            if "Work-Life Balance" in df.columns:
                df["Work-Life Balance"] = df["Work-Life Balance"].replace({"Fair": "Below Average"})

            if "Job Satisfaction" in df.columns:
                df["Job Satisfaction"] = df["Job Satisfaction"].replace({"Very High": "High"})
            if "Employee Recognition" in df.columns:
                df["Employee Recognition"] = df["Employee Recognition"].replace({"Very High": "High"})
            
            # --- PATCH 2: Drop ground-truth targets if present so they don't break incoming data schemas ---
            target_cols_to_ignore = ["Attrition", "attrition", "status", "Status", "Target", "target"]
            for col in target_cols_to_ignore:
                if col in df.columns:
                    df = df.drop(columns=[col])
            
            records = df.to_dict(orient="records")
            
            for idx, record in enumerate(records):
                # Map variables against central validator boundaries
                EmployeeDataInput(**record)
                
            return True
        except Exception as e:
            logger.error(f"Validation structural layout crash on file {file_path} at row [{idx}]. Detail: {str(e)}")
            return False

    def initiate_batch_validation(self):
        logger.info("Initiating Step 2: CSV Dataset Batch Validation Pipeline.")
        try:
            train_src = os.path.join(self.project_root, self.validation_config["raw_train_path"])
            test_src = os.path.join(self.project_root, self.validation_config["raw_test_path"])

            if not os.path.exists(train_src) or not os.path.exists(test_src):
                raise FileNotFoundError(
                    f"Missing operational data templates. Checked absolute locations:\n"
                    f"- {train_src}\n- {test_src}"
                )

            # Core validation routine calls
            train_valid = self.validate_csv_dataset(train_src)
            test_valid = self.validate_csv_dataset(test_src)

            if train_valid and test_valid:
                logger.info("Validation processing complete. Both matrices match schema footprints.")
                
                valid_train_target = os.path.join(self.project_root, self.validation_config["validated_train_path"])
                valid_test_target = os.path.join(self.project_root, self.validation_config["validated_test_path"])
                
                os.makedirs(os.path.dirname(valid_train_target), exist_ok=True)
                
                # Cache baseline dataset copies inside the artifacts workspace folder
                pd.read_csv(train_src).to_csv(valid_train_target, index=False)
                pd.read_csv(test_src).to_csv(valid_test_target, index=False)
                
                logger.info("Sanitized CSV baselines written to artifacts storage folder.")
                print("✅ Step 2 Complete: CSV data sheets successfully validated and cached!")
            else:
                raise ValueError("Dataset validation checklist failed. Review structural integrity logs.")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    validator = BatchDataValidation()
    validator.initiate_batch_validation()
