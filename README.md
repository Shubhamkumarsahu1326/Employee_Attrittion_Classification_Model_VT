# comletemlops
## The project execution 
The project was implemented in different phases:
- Initialization and version control setup: Initialized Git and DVC at the very beginning to track code and data versions simultaneously before training.
- Data ingestion and automated validation: Developed data ingestion pipelines coupled with rigorous input validation frameworks to catch corrupted training data early.
- Experiment tracking and model registration: Modularized model training routines while logging all hyperparameters, metrics, and data hashes directly into MLflow.
- Decoupled API and user interface development: Built a high-performance FastAPI backend alongside a separate, lightweight Streamlit UI that communicates strictly via HTTP requests.
- Robust data validation and S3 integration: Implemented Pydantic schemas for strict requestresponse validation while securing data storage layers with AWS S3.
- Comprehensive automated testing integration: Wrote a multi-tier testing suite covering isolated code unit tests, API endpoint integration tests, and model output validations.
- Containerization and CI/CD pipeline automation: Packaged all application layers into standard Docker containers and configured GitHub Actions to trigger automated test-and-build workflows.
- Cloud deployment and infrastructure provisioning: Deployed the containerized applications to AWS compute services utilizing structured infrastructure configurations rather than manual setups.
- Observability and model monitoring implementation: Integrated Prometheus and Grafana dashboards to track live system performance metrics and catch production data drift.
- Feedback loop and continuous retraining setup: Formed a production feedback mechanism by capturing live inference logs in S3 to power future model retraining cycles.
### change of plans for execution 
 since i only had access to the free tier for aws the ec2 instances wew very small and the docker image were large it would be task for some other time ,the s3 were also not implemented inearly stages as the access key instances were consisting of special characters like + etc which windows were ignoring so it was skipped 

 ###problems while execution 
 - train serve skrew 
 - s3 implementation
 - ec2 implementation