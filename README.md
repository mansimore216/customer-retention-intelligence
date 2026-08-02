# Customer Churn Prediction & Retention Intelligence

An end-to-end **AI-powered customer retention intelligence platform** that predicts customer churn, explains the key factors driving churn risk using **SHAP explainability**, and recommends actionable retention strategies.

Built using a trained **XGBoost classification model** and the **Telco Customer Churn dataset**, the application provides both **individual customer-level churn prediction** and **dataset-level retention analytics** through a professional Flask web dashboard.

---

## 🚀 Overview

Customer churn is a major challenge for subscription-based businesses. Identifying customers who are likely to leave is only the first step — businesses also need to understand **why a customer is at risk** and determine **what action should be taken to retain them**.

This project addresses that problem by combining:

* Machine learning-based churn prediction
* SHAP-powered explainable AI
* Customer risk segmentation
* Customer-level churn scoring
* Business-focused retention recommendations
* Dataset-wide churn analytics
* Interactive visualization dashboard

The platform transforms raw customer information into actionable retention intelligence.

---

## ✨ Key Features

### 🔮 1. Live Customer Churn Prediction

Enter a customer's information through an interactive form and receive a real-time churn prediction.

The system processes new customer inputs using the same preprocessing pipeline used during model development and generates:

* Churn probability
* Risk classification
* Risk level
* Explainable churn factors
* Personalized retention recommendations

Risk levels are categorized as:

| Risk Level | Churn Probability |
| ---------- | ----------------- |
| 🟢 Low     | < 30%             |
| 🟡 Medium  | 30% – 60%         |
| 🔴 High    | ≥ 60%             |

---

### 🧠 2. Explainable AI with SHAP

Instead of providing only a prediction, the platform explains **why the model considers a customer risky**.

A live SHAP `TreeExplainer` analyzes the XGBoost prediction and identifies the most influential factors contributing to the customer's churn risk.

The application visually distinguishes between:

* 🔴 **Risk-increasing factors**
* 🟢 **Risk-decreasing factors**

This helps users understand the reasoning behind each prediction and makes the ML system more interpretable for business decision-making.

---

### 💡 3. Actionable Retention Recommendations

The system goes beyond churn prediction by connecting important churn factors with potential retention actions.

Based on the customer's risk drivers, the application recommends relevant interventions such as:

* Contract upgrade opportunities
* Customer support engagement
* Service assistance
* Security and protection service recommendations
* Billing or payment-related interventions
* Personalized retention offers

This creates a complete workflow:

**Predict → Explain → Recommend → Retain**

---

### 📊 4. Executive Dashboard

The dashboard provides an overview of customer retention risk across the complete dataset.

It includes:

* Total Customers
* High-Risk Customers
* Average Churn Probability
* Retention Opportunities
* Customer Risk Distribution

The dashboard dynamically scores the dataset using the ML model instead of relying on a pre-generated static prediction CSV.

---

### 📈 5. Customer Insights & Analytics

The Customer Insights page provides interactive visual analysis of churn behavior.

The application analyzes churn rates across important business dimensions, including:

* Contract type
* Customer tenure
* Monthly charges
* Internet service
* Payment method

These insights help identify customer segments that may require targeted retention strategies.

---

### ⚙️ 6. Verified Preprocessing Pipeline

The application rebuilds the preprocessing pipeline used during model training, including:

* Missing-value handling
* Median imputation for numerical features
* Most-frequent imputation for categorical features
* Standard scaling for numerical features
* One-hot encoding for categorical features

The rebuilt pipeline was verified against the notebook's saved processed test dataset using numerical comparison with `np.allclose()`.

This ensures that new customer inputs are transformed consistently with the data used during model training and evaluation.

---

### ❤️ 7. Real-Time Backend Health Monitoring

The application includes a backend health indicator in the navigation header.

The frontend communicates with the Flask backend to monitor whether the prediction engine and application services are available.

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────────┐
                    │      Customer Input      │
                    │   Web Prediction Form    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │       Flask Backend      │
                    │       /api/predict       │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Preprocessing Pipeline │
                    │                          │
                    │ • Missing Value Handling │
                    │ • Scaling                │
                    │ • One-Hot Encoding       │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     XGBoost Model        │
                    │   Churn Probability      │
                    └────────────┬────────────┘
                                 │
                       ┌─────────┴─────────┐
                       ▼                   ▼
             ┌─────────────────┐   ┌──────────────────┐
             │  SHAP Explainer │   │ Risk Classification│
             │                 │   │                  │
             │ Why will they   │   │ Low / Medium /   │
             │ potentially     │   │ High             │
             │ churn?          │   │                  │
             └────────┬────────┘   └────────┬─────────┘
                      │                     │
                      └──────────┬──────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │ Retention Recommendation│
                    │        Engine            │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     Web Dashboard        │
                    │                          │
                    │ Prediction + Explanation │
                    │ + Recommendations        │
                    └─────────────────────────┘
```

---

## 🧠 Machine Learning Pipeline

The project uses a supervised machine learning pipeline for customer churn classification.

### Data

The model is trained using the **Telco Customer Churn dataset**, containing customer demographic, account, service, and billing information.

Important feature categories include:

* Customer demographics
* Tenure
* Phone services
* Internet services
* Online security
* Technical support
* Streaming services
* Contract type
* Paperless billing
* Payment method
* Monthly charges
* Total charges

### Preprocessing

The preprocessing pipeline handles both numerical and categorical features.

```text
Raw Customer Data
       │
       ▼
Data Cleaning
       │
       ├── Numerical Features
       │       ├── Median Imputation
       │       └── Standard Scaling
       │
       └── Categorical Features
               ├── Most-Frequent Imputation
               └── One-Hot Encoding
       │
       ▼
Processed Feature Matrix
       │
       ▼
XGBoost Classifier
       │
       ▼
Churn Probability
       │
       ▼
Risk Classification
```

The application uses the same train/test split configuration as the original training pipeline:

```python
train_test_split(
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The trained XGBoost model is loaded directly from the saved model file and is **not retrained when the application starts**.

---

## 📊 Model Performance

The trained XGBoost model achieved approximately:

* **ROC-AUC: ~0.836** on the notebook's held-out test set

The model is used as-is by the Flask application.

> **Note:** The performance metric reported above is based on the existing model evaluation performed during the project's ML development process.

---

## 🔍 Explainable AI Workflow

For every individual prediction, the application uses SHAP to analyze the model's decision.

```text
Customer Data
     │
     ▼
Preprocessing
     │
     ▼
XGBoost Prediction
     │
     ├───────────────┐
     ▼               ▼
Churn Probability   SHAP Analysis
     │               │
     │               ▼
     │        Top Risk Factors
     │               │
     └───────┬───────┘
             ▼
      Risk Classification
             │
             ▼
   Retention Recommendations
```

This allows the platform to answer three important questions:

> **Will the customer churn?**

> **Why is the customer likely to churn?**

> **What can the business do about it?**

---

## 🛠️ Tech Stack

### Machine Learning

* Python
* XGBoost
* Scikit-learn
* SHAP
* NumPy
* Pandas

### Backend

* Flask
* Python REST APIs

### Frontend

* HTML5
* CSS3
* JavaScript
* Chart-based data visualization

### Data & Model

* Telco Customer Churn Dataset
* Trained XGBoost classification model

---

## 📁 Project Structure

```text
churn_project/
│
├── app.py
├── ml_engine.py
├── requirements.txt
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── notebooks/
│   └── models/
│       └── final_xgboost_churn_model.pkl
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   └── insights.html
│
└── static/
    ├── css/
    │   └── style.css
    │
    └── js/
        ├── dashboard.js
        ├── predict.js
        └── insights.js
```

### Main Components

| File                            | Responsibility                                                                |
| ------------------------------- | ----------------------------------------------------------------------------- |
| `app.py`                        | Flask application, routes, API endpoints                                      |
| `ml_engine.py`                  | Preprocessing, model loading, SHAP explanations, predictions, recommendations |
| `index.html`                    | Main retention intelligence dashboard                                         |
| `predict.html`                  | Customer prediction interface                                                 |
| `insights.html`                 | Churn analytics and business insights                                         |
| `dashboard.js`                  | Dashboard data and visualizations                                             |
| `predict.js`                    | Prediction form and result rendering                                          |
| `insights.js`                   | Customer insights visualizations                                              |
| `style.css`                     | Application UI and design system                                              |
| `final_xgboost_churn_model.pkl` | Trained XGBoost churn prediction model                                        |

---

## 🌐 Application Pages

### 1. Dashboard

Provides a high-level overview of customer churn risk.

**Includes:**

* Total customer count
* High-risk customer count
* Average churn probability
* Retention opportunities
* Overall risk distribution

---

### 2. Predict Churn

Allows users to enter individual customer information.

**Workflow:**

```text
Enter Customer Details
        ↓
Generate Prediction
        ↓
View Churn Probability
        ↓
View Risk Level
        ↓
Analyze SHAP Factors
        ↓
View Retention Recommendations
```

---

### 3. Customer Insights

Provides dataset-level churn analysis.

The page analyzes churn patterns based on:

* Contract
* Tenure
* Monthly Charges
* Internet Service
* Payment Method

These visualizations help identify high-risk customer segments and potential areas for retention campaigns.

---

## 🔌 API Endpoints

### Health Check

```http
GET /api/health
```

Checks the status of the Flask application and ML engine.

---

### Form Schema

```http
GET /api/form-schema
```

Returns the customer input schema used to dynamically drive the prediction form.

---

### Dashboard Statistics

```http
GET /api/dashboard-stats
```

Returns dataset-level model scoring statistics for the dashboard.

---

### Risk Distribution

```http
GET /api/risk-distribution
```

Returns the distribution of customers across churn-risk categories.

---

### Customer Insights

```http
GET /api/insights
```

Returns churn analytics for different customer segments.

---

### Live Churn Prediction

```http
POST /api/predict
```

Accepts customer information and returns:

* Churn probability
* Risk classification
* SHAP-based explanations
* Retention recommendations

Example request:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70,
  "TotalCharges": 840
}
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd churn_project
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Flask server:

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

The application will initialize the ML engine, load the trained XGBoost model, rebuild the verified preprocessing pipeline, and initialize the SHAP explainer.

> **Note:** The first application startup may take longer because SHAP and its dependencies may require additional initialization. Subsequent requests do not recreate the model or SHAP explainer.

---

## 🧪 Testing

The application can be tested through the web interface or directly through the API.

### Test the health endpoint

```bash
curl http://127.0.0.1:5000/api/health
```

### Test dashboard statistics

```bash
curl http://127.0.0.1:5000/api/dashboard-stats
```

### Test live prediction

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"No","Dependents":"No","tenure":12,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":70,"TotalCharges":840}'
```

### Error Handling

The API handles common invalid requests, including:

* Missing required fields
* Invalid or non-JSON request bodies
* Unknown routes

---

## 📸 Screenshots

Add screenshots of the application here after running the project.

### Dashboard

![Dashboard Screenshot](screenshots/dashboard.png)

### Churn Prediction

![Prediction Screenshot](screenshots/prediction.png)

### SHAP Explainability & Recommendations

![Explainability Screenshot](screenshots/explainability.png)

### Customer Insights

![Insights Screenshot](screenshots/insights.png)

> Create a `screenshots/` folder in the repository and add your actual screenshots before pushing this README.

---

## 💼 Business Value

The platform is designed to support customer retention teams by helping them:

* Identify customers at high risk of churn
* Prioritize customers for retention campaigns
* Understand the key drivers behind churn risk
* Segment customers based on risk
* Take targeted retention actions
* Reduce reliance on purely reactive customer retention strategies

Instead of simply predicting churn, the platform aims to provide **decision support for retention teams**.

---

## 🎯 Project Highlights

This project demonstrates practical experience with:

* End-to-end machine learning workflows
* XGBoost classification
* Imbalanced classification handling
* Scikit-learn preprocessing pipelines
* Flask API development
* REST API integration
* Explainable AI using SHAP
* Real-time ML inference
* Interactive dashboards
* Data visualization
* Business-oriented ML applications
* Model-to-application deployment

---

## 🔮 Future Enhancements

Potential future improvements include:

* Customer retention campaign tracking
* Automated email or SMS retention workflows
* Customer segmentation using clustering
* Cost-sensitive churn prediction
* Retention campaign ROI estimation
* Model monitoring and drift detection
* Automated model retraining pipeline
* Authentication and role-based access
* Database integration for customer records
* Cloud deployment
* Advanced experiment tracking and MLOps integration

---

## 📌 Disclaimer

This project is intended for educational and portfolio purposes. The churn predictions and retention recommendations are generated by a machine learning model trained on the Telco Customer Churn dataset and should not be treated as guaranteed predictions of real-world customer behavior.

---

## 👩‍💻 Author

**Mansi More**



⭐ If you found this project interesting, consider giving the repository a star!
