# End-to-End MLOps Pipeline for Financial Market Direction Prediction

## Overview

This project demonstrates an end-to-end MLOps workflow for predicting the next-day direction of the NIFTY 50 index using machine learning. The objective is to classify whether the market will move **UP** or **DOWN** on the next trading day based on historical price-derived features.

The project covers the complete machine learning lifecycle, including:

* Data Ingestion
* Feature Engineering
* Model Training
* Experiment Tracking with MLflow
* Model Comparison
* Best Model Selection
* Model Serialization
* FastAPI Deployment
* Docker Containerization

---

## Architecture

```text
Yahoo Finance
      │
      ▼
Data Ingestion
      │
      ▼
Feature Engineering
      │
      ▼
Model Training
      │
      ▼
MLflow Tracking
      │
      ▼
Model Comparison
      │
      ▼
Best Model Selection
      │
      ▼
Model Serialization
      │
      ▼
FastAPI Deployment
      │
      ▼
Docker Container
```

---

## Project Structure

```text
nifty-mlops/
│
├── app.py
├── train_pipeline.py
│
├── models/
│   └── nifty_best_model.pkl
│
├── mlruns/
│
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

---

## Dataset

Historical NIFTY 50 index data is downloaded directly from Yahoo Finance using the `yfinance` library.

Ticker Used:

```text
^NSEI
```

Date Range:

```text
2015 - Present
```

---

## Feature Engineering

The following technical indicators are generated:

### Returns

* Return_1d
* Return_3d
* Return_5d

### Moving Averages

* MA_5
* MA_10
* MA_20

### Volatility

* Volatility_5
* Volatility_10

### Target Variable

```python
Target = 1 if Tomorrow_Close > Today_Close
Target = 0 otherwise
```

---

## Models Compared

The project trains and compares multiple machine learning models:

### Logistic Regression

A simple linear baseline model.

### Random Forest

An ensemble tree-based classifier.

### XGBoost

A gradient boosting framework optimized for tabular datasets.

### LightGBM

A fast gradient boosting framework widely used in industry.

---

## Experiment Tracking with MLflow

MLflow is used for:

* Experiment Tracking
* Parameter Logging
* Metric Logging
* Model Artifact Storage
* Model Comparison

Tracked Metrics:

* Accuracy
* Precision
* Recall
* F1 Score

Example MLflow Dashboard:

```text
Nifty_Direction_Prediction

├── LogisticRegression
├── RandomForest
├── XGBoost
└── LightGBM
```

---

## Training

Run the training pipeline:

```bash
python train_pipeline.py
```

The script will:

1. Download market data
2. Generate features
3. Train multiple models
4. Log experiments to MLflow
5. Select the best model
6. Save the best model

Output:

```text
models/nifty_best_model.pkl
```

---

## MLflow UI

Launch MLflow:

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

to visualize experiments and compare model performance.

---

## FastAPI Deployment

Start the API server:

```bash
uvicorn app:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### Health Check

```http
GET /
```

Response:

```json
{
  "status": "running"
}
```

---

### Prediction Endpoint

```http
POST /predict
```

Sample Request:

```json
{
  "Return_1d": 0.01,
  "Return_3d": 0.03,
  "Return_5d": 0.05,
  "MA_5": 25000,
  "MA_10": 24950,
  "MA_20": 24800,
  "Volatility_5": 0.02,
  "Volatility_10": 0.03
}
```

Sample Response:

```json
{
  "prediction": "UP"
}
```

---

## Docker

Build Docker Image:

```bash
docker build -t nifty-mlops .
```

Run Container:

```bash
docker run -p 8000:8000 nifty-mlops
```

Access API:

```text
http://localhost:8000/docs
```

---

## Technologies Used

### Machine Learning

* Scikit-Learn
* XGBoost
* LightGBM

### MLOps

* MLflow

### API Serving

* FastAPI
* Uvicorn

### Data Processing

* Pandas
* NumPy

### Data Source

* Yahoo Finance
* yfinance

### Containerization

* Docker

---

## Future Improvements

* Hyperparameter Optimization
* Time-Series Cross Validation
* Automated Retraining Pipelines
* CI/CD Integration
* Kubernetes Deployment
* Model Registry Integration
* Real-Time Market Data Ingestion
* Monitoring and Drift Detection

---

## Author

Jay Dave

M.Tech Computer Science & Engineering
Nirma University

Machine Learning | MLOps | GenAI | Distributed Systems
