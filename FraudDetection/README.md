# 🔎 Fraud Detection MLOps Pipeline

## Overview

This repository contains an **end-to-end MLOps pipeline** for detecting fraudulent financial transactions using machine learning.

The project demonstrates how a machine learning model can be:

- trained and evaluated
- tracked using experiment management
- deployed as a REST API
- containerized using Docker
- automatically built using CI/CD pipelines

The goal is to simulate a **production-ready machine learning workflow** commonly used in real-world data science and MLOps environments.

---

## Problem Statement

Fraud detection is a critical problem in the financial industry. Traditional rule-based systems often fail to detect evolving fraud patterns.

Machine learning models can identify anomalies and suspicious transaction patterns using historical transaction data.

This project builds a system that:

- trains a fraud detection model
- exposes predictions through a scalable API
- logs predictions for monitoring and analysis
- automates build pipelines using CI/CD

---

## Technology Stack

| Component | Technology |
|----------|-------------|
| API Framework | FastAPI |
| Machine Learning | Scikit-Learn |
| Experiment Tracking | MLflow |
| Data Processing | Pandas |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Model Serialization | Joblib |

---

## Project Architecture

The system follows a modular machine learning architecture separating **data processing, model training, inference, and deployment**.

```
Client Request
      │
      ▼
FastAPI REST API
      │
      ▼
Feature Processing
      │
      ▼
Trained ML Model
      │
      ▼
Fraud Prediction
      │
      ▼
Prediction Logging
```

---

## Machine Learning Pipeline

The training pipeline consists of the following steps:

1. Load dataset
2. Encode categorical variables
3. Split dataset into training and testing sets
4. Scale numerical features
5. Train a RandomForest classifier
6. Log experiments using MLflow
7. Save model artifacts

Saved artifacts:

```
models/model.pkl
models/scaler.pkl
models/features.pkl
```

These artifacts are used during the inference stage.

---

## Model Evaluation

The model performance is evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

All metrics and parameters are tracked using MLflow.

---

## Experiment Tracking

The project uses **MLflow** to track model experiments.

Start the MLflow UI:

```bash
mlflow ui
```

Open the dashboard:

```
http://localhost:5000
```

MLflow stores:

- parameters
- model metrics
- trained artifacts

---

## API Service

The trained model is exposed through a REST API using FastAPI.

Start the API server:

```bash
uvicorn api.app:app --reload
```

Interactive documentation:

```
http://localhost:8000/docs
```

---

## Prediction Endpoint

**POST /predict**

Example request:

```json
{
  "amount": 200,
  "transaction_hour": 14,
  "merchant_category": "electronics",
  "foreign_transaction": 0,
  "location_mismatch": 0,
  "device_trust_score": 80,
  "velocity_last_24h": 3,
  "cardholder_age": 45
}
```

Example response:

```json
{
  "fraud_prediction": 0
}
```

---

## Docker Deployment

Build Docker image:

```bash
docker build -t fraud-detection-api .
```

Run container:

```bash
docker run -p 8000:8000 fraud-detection-api
```

Access API documentation:

```
http://localhost:8000/docs
```

---

## CI/CD Pipeline

This project includes an automated CI/CD pipeline using GitHub Actions.

Pipeline workflow:

```
Code Push
   │
   ▼
GitHub Repository
   │
   ▼
GitHub Actions Workflow
   │
   ▼
Docker Image Build
   │
   ▼
Push Image to Docker Hub
```

This ensures that every code update automatically builds a deployable container image.

---

## Prediction Logging

All prediction requests are logged to:

```
logs/predictions.csv
```

Each record includes:

- input features
- predicted fraud label
- timestamp

This logging system enables:

- monitoring prediction usage
- analyzing model behavior
- future data drift detection

---

## Future Improvements

Possible enhancements for production-grade systems:

- Data Drift Detection
- Model Versioning
- Automated Retraining Pipelines
- Real-Time Streaming Inference
- Kubernetes Deployment
- Building an app using Streamlit

---

## Key MLOps Concepts Demonstrated

This project demonstrates several core MLOps principles:

- Modular machine learning pipeline design
- API-based model serving
- experiment tracking
- containerized deployment
- CI/CD automation
- prediction logging

---

## Author

Akshay Bharadwaj

---

# License

This project is licensed under [MIT License](LICENSE)
