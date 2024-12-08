# End-to-End ML Deployment for Network Security Prediction 🚀

This project demonstrates my ability to design and deploy a complete machine learning pipeline for **network security prediction**. The system is production-ready, scalable, and integrates modern MLOps practices. It showcases my skills in building modular systems, cloud deployment, and CI/CD pipelines.

---

## Key Highlights of the Project ✨

### 1. **End-to-End Machine Learning Pipeline**
- Designed and implemented a full ML pipeline covering:
  - Data ingestion
  - Data preprocessing and feature engineering
  - Model training with hyperparameter tuning using `GridSearchCV` and looping through multiple models
  - Model serialization and artifact storage
  - Deployment and inference serving

### 2. **Experiment Tracking**
- Automated hyperparameter tuning and evaluation with **GridSearchCV**.
- Tracked the best model, its hyperparameters, metrics, and artifacts in **MLflow** and integrated it with **DagsHub** for centralized experiment tracking and collaboration.

### 3. **Model Deployment with FastAPI**
- Built a RESTful API with **FastAPI** to serve the ML model for real-time inference.
- The **inference step** is integrated as a POST request to the FastAPI endpoint for seamless prediction serving.

### 4. **Cloud-Native Artifact Management**
- Leveraged **AWS S3** to store:
  - Model artifacts
  - Preprocessed datasets
  - Logs and evaluation metrics for reproducibility and versioning.

### 5. **CI/CD Pipelines**
- Designed a **Continuous Integration pipeline** with **GitHub Actions** to automate:
  - Code linting
  - Unit tests for ML pipelines
  - Docker image creation
- Built a **Continuous Delivery pipeline** to:
  - Push Docker images to **AWS Elastic Container Registry (ECR)**.
  - Deploy the model on an **AWS EC2 instance**.

### 6. **Continuous Deployment with GitHub Runner**
- Configured **self-hosted GitHub Runners** on AWS EC2 to automate model deployment and updates.
- Integrated version control with robust workflows to ensure minimal downtime during deployment.

### 7. **Scalable and Maintainable Code**
- Followed best practices in modular coding, configuration management with Python `dataclasses`, and integrated robust logging and exception handling.

---

## Technologies Used 🛠️

- **Languages**: Python
- **Frameworks**: FastAPI, scikit-learn
- **Experiment Tracking**: MLflow, DagsHub
- **Cloud Services**: AWS S3, AWS ECR, AWS EC2
- **Containerization**: Docker
- **CI/CD Tools**: GitHub Actions, GitHub Runner
- **Orchestration**: Bash scripting for deployment tasks
- **MLOps Practices**: Model serialization, artifact storage, modular pipelines

