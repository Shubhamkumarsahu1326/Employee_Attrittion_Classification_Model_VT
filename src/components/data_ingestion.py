import os
import sys
import yaml
import pandas as pd
from sklearn.model_model_selection import train_test_split
from dataclasses import dataclass

# Connect with your established MLOps tracking tools
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.components.data_validation_schema import validate_dataframe

@dataclass
class DataIngestionConfig:
    raw_data_path: str
    train_data_path: str
    test_data_path: str

class DataIngestion:
    def __init__(self):
        # Load path routing settings dynamically from central config
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)
            
        ingestion_paths = config["artifacts"]["data_ingestion"]
        self.ingestion_config = DataIngestionConfig(
            raw_data_path=ingestion_paths["raw_data_path"],
            train_data_path=ingestion_paths["train_data_path"],
            test_data_path=ingestion_paths["test_data_path"]
        )

    def initiate_data_ingestion(self) -> tuple[str, str]:
        logging.info("Starting Data Ingestion execution cycle.")
        try:
            # 1. Read the raw data file tracking parameter path
            if not os.path.exists(self.ingestion_config.raw_data_path):
                raise FileNotFoundError(f"Missing raw baseline dataset file at: {self.ingestion_config.raw_data_path}")
                
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info(f"Loaded raw dataset file successfully. Matrix shape: {df.shape}")

            # 2. Trigger Automated Validation Engine Check
            logging.info("Triggering automated dataset structural integrity validation check.")
            validate_dataframe(df)
            logging.info("Automated structural validation check passed successfully. No corruption detected.")

            # 3. Create target directory structures for artifact splits
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # 4. Perform Data Partition Split
            logging.info("Executing train-test split operations.")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # 5. Save the processed data partitions to your artifacts directory
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train and test data partition splits exported successfully.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Exception encountered during data ingestion run runtime loop.")
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Test script runtime execution
    try:
        obj = DataIngestion()
        train_path, test_path = obj.initiate_data_ingestion()
        print(f"✅ Ingestion Successful!\nTrain Path: {train_path}\nTest Path: {test_path}")
    except Exception as e:
        print(f"❌ Execution Failure: {e}")
