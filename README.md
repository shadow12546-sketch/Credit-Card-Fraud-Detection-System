# 🛡️ Credit Card Fraud Detection System

A production-style machine learning application that detects potentially fraudulent credit card transactions in real time using an XGBoost classifier. The project includes an interactive Streamlit dashboard, a FastAPI backend for model serving, explainable fraud scoring, transaction history tracking, and cloud deployment.

## 🚀 Live Demo

### Frontend (Streamlit Dashboard)

https://credit-card-fraud-detection-system-5ehn8ngcvmqj6ppeejhq5b.streamlit.app

### Backend (FastAPI API)

https://credit-card-fraud-detection-system-gw2k.onrender.com

### API Health Check

https://credit-card-fraud-detection-system-gw2k.onrender.com/health

---

# 📌 Project Overview

Financial fraud causes billions of dollars in losses every year. This project demonstrates how machine learning can be used to identify suspicious transactions before they are approved.

The system:

* Analyzes transaction details in real time
* Generates a fraud risk score
* Classifies transactions as OK or ALERT
* Explains the key factors behind each prediction
* Tracks card transaction history
* Provides an interactive analytics dashboard

---

# ✨ Features

### Machine Learning

* XGBoost-based fraud detection model
* Time-aware feature engineering
* Class imbalance handling
* Probability-based risk scoring
* Explainable predictions

### Dashboard

* Real-time fraud scoring
* Interactive risk gauge
* Fraud explanation panel
* Transaction history analysis
* Session logging
* Scenario simulation
* Risk trend visualization

### Backend API

* FastAPI REST endpoints
* Model serving
* Card history tracking
* Metadata endpoints
* Health monitoring

### Deployment

* Streamlit Community Cloud
* Render Cloud Platform
* GitHub CI/CD integration

---

# 🏗️ System Architecture

```text
User
 │
 ▼
Streamlit Dashboard
 │
 │ HTTP Requests
 ▼
FastAPI Backend
 │
 ▼
XGBoost Fraud Detection Model
 │
 ▼
Fraud Prediction Response
```

---

# 🧰 Tech Stack

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* XGBoost
* Scikit-Learn
* Joblib

### Backend

* FastAPI
* Pydantic
* Uvicorn

### Frontend

* Streamlit
* Altair

### Deployment & Version Control

* Git
* GitHub
* Render
* Streamlit Community Cloud

---

# 📊 Model Performance

| Metric    | Score |
| --------- | ----- |
| ROC-AUC   | 0.999 |
| PR-AUC    | 0.928 |
| Precision | 0.756 |
| Recall    | 0.898 |
| F1 Score  | 0.821 |
| F2 Score  | 0.865 |
| Threshold | 0.805 |

### Confusion Matrix

| Metric          | Count  |
| --------------- | ------ |
| True Positives  | 1017   |
| False Positives | 328    |
| True Negatives  | 193041 |
| False Negatives | 116    |

---

# 📂 Project Structure

```text
Credit-Card-Fraud-Detection-System/
│
├── Dashboard.py
├── api.py
├── train_model.py
├── error_analysis.py
├── make_seed.py
├── Analyze dataset.py
│
├── model_output/
│   ├── fraud_model.joblib
│   ├── cards_seed.csv
│   ├── metrics.json
│   ├── threshold_table.csv
│   ├── missed_frauds.csv
│   └── shap_summary.png
│
├── requirements.txt
├── requirements-api.txt
├── dataset_analysis.md
└── .streamlit/
```

---

# 📁 Dataset

This project uses the Kaggle Credit Card Fraud Detection dataset containing simulated credit card transactions with fraud labels. The dataset includes merchant information, transaction amounts, timestamps, geographic locations, customer profiles, and fraud indicators.

### Kaggle Dataset

https://www.kaggle.com/datasets/kartik2112/fraud-detection

### Dataset Characteristics

* ~1.85 Million Transactions
* Highly Imbalanced Fraud Classes
* Transaction Metadata
* Merchant Information
* Geographic Features
* Customer Profiles
* Fraud Labels

Dataset source information matches the commonly used Kaggle fraud-detection dataset referenced by similar fraud detection projects.

---

# 🔌 API Endpoints

### Health Check

```http
GET /health
```

### Model Metadata

```http
GET /meta
```

### Card Information

```http
GET /card/{token}
```

### Fraud Prediction

```http
POST /score
```

Example request:

```json
{
  "card_token": "123456",
  "amount": 950,
  "category": "shopping_net",
  "merchant": "merchant_name",
  "state": "TX",
  "timestamp": "2026-10-01T12:00:00",
  "commit": false
}
```

---

# ⚙️ Local Setup

## Clone Repository

```bash
git clone https://github.com/shadow12546-sketch/Credit-Card-Fraud-Detection-System.git
cd Credit-Card-Fraud-Detection-System
```

## Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

## Install Dependencies

### Dashboard

```bash
pip install -r requirements.txt
```

### Backend

```bash
pip install -r requirements-api.txt
```

---

# ▶️ Run Locally

### Start Backend

```bash
uvicorn api:app --reload --port 8000
```

### Start Dashboard

```bash
streamlit run Dashboard.py
```

---

# 📈 Future Improvements

* Real-time transaction streaming
* Docker containerization
* Kubernetes deployment
* User authentication
* Database integration
* Advanced SHAP explainability
* Model retraining pipeline
* Monitoring and alerting

---

# 👨‍💻 Author

Developed as a machine learning and full-stack deployment project demonstrating fraud detection, model serving, explainable AI, and cloud deployment.

GitHub:
https://github.com/shadow12546-sketch/Credit-Card-Fraud-Detection-System
