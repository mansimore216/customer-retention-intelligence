# Customer Churn Prediction & Retention Intelligence

A Flask web app that predicts whether a customer will churn, explains *why*
using live SHAP explainability, and recommends concrete retention actions —
built on your trained XGBoost model and the Telco Customer Churn dataset.

## What changed from the uploaded project

The original project had a working ML pipeline in the notebooks, but the
Flask app was a **lookup tool over a static CSV** (`customer_retention_report.csv`)
rather than a live prediction system, and it had no way to score a customer
who wasn't already in that file.

**Root cause found during inspection:** `03_Data_Preprocessing.ipynb` fits a
scikit-learn `ColumnTransformer` (median-impute + `StandardScaler` for
numeric columns, most-frequent-impute + `OneHotEncoder` for categorical
columns), but only saves its *output* (`X_test_processed.pkl`) — the fitted
transformer object itself was never serialized. Without it, there was no way
to correctly preprocess a brand-new customer's input to match what the model
expects.

**Fix:** `ml_engine.py` rebuilds that exact pipeline (same cleaning steps,
same `train_test_split(test_size=0.20, random_state=42, stratify=y)`) and
fits it at server startup. This was verified with `np.allclose()` against
the notebook's saved `X_test_processed.pkl` — the rebuilt pipeline produces
**numerically identical output**, so predictions are trustworthy and
consistent with how the model was trained and evaluated.

### Summary of changes

| File | Change |
|---|---|
| `ml_engine.py` | **New.** Rebuilds the verified preprocessing pipeline, loads the trained model, builds a live `shap.TreeExplainer`, and exposes `predict()` (single customer, with explanation + recommendations) and dataset-wide scoring for the dashboard/insights pages. |
| `app.py` | **Rewritten.** Removed the CSV-lookup routes. Added `/api/predict` (live prediction), `/api/form-schema` (drives the form), `/api/dashboard-stats`, `/api/risk-distribution`, `/api/insights`, `/api/health`. Pages: `/`, `/predict`, `/insights`. |
| `templates/predict.html` + `static/js/predict.js` | **New.** Replaces the old customer-lookup page with a real input form (Customer Info / Account Info / Services / Billing sections), a result panel (risk badge, probability bar, SHAP factor chart), and a recommendations list. |
| `templates/index.html` + `static/js/dashboard.js` | **Rewritten.** Dashboard cards (Total Customers, High-Risk Customers, Average Churn Probability, Retention Opportunities) now come from live model scoring of the full dataset, not the static CSV. |
| `templates/insights.html` + `static/js/insights.js` | **Rewritten.** Charts (churn distribution, churn rate by contract/tenure/monthly charges/internet service/payment method) are computed live from the cleaned dataset's actual `Churn` labels. |
| `templates/base.html` | Updated navigation (Dashboard / Predict Churn / Customer Insights) and a live backend-health indicator in the header. |
| `static/css/style.css` | Kept your existing dark, professional design system; added styles for the form, toggle inputs, result hero, factor chart, and recommendation cards. |
| `requirements.txt` | Updated to the actual runtime dependencies (`shap`, `xgboost`, etc.) with minimum versions. |
| `test_app.py` | **Removed** — this was an unrelated Streamlit "hello world" stub, not part of the app (and the brief explicitly excludes Streamlit). |

The trained model (`notebooks/models/final_xgboost_churn_model.pkl`) was
**not retrained** — it's loaded and used as-is (verified ROC-AUC ≈ 0.836 on
the notebook's held-out test set).

## Project structure

```
churn_project/
├── app.py                     # Flask routes
├── ml_engine.py                # Preprocessing + model + SHAP + recommendations
├── requirements.txt
├── data/
│   └── Telco-Customer-Churn.csv
├── notebooks/                  # Original EDA/training notebooks (unchanged)
│   └── models/
│       └── final_xgboost_churn_model.pkl
├── templates/
│   ├── base.html
│   ├── index.html               # Dashboard
│   ├── predict.html             # Prediction form + results
│   └── insights.html            # Analytics charts
└── static/
    ├── css/style.css
    └── js/
        ├── dashboard.js
        ├── predict.js
        └── insights.js
```

## Installation

```bash
cd churn_project
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

> **First-run note:** `shap`'s first import compiles some numba-accelerated
> code, which can take ~10–15 seconds. This only happens once per process
> start (not per request) — you'll see log lines up through
> `ChurnEngine ready.` before the server starts accepting requests.

## Testing it

1. **Dashboard (`/`)** — confirm the four summary cards populate (Total
   Customers, High-Risk Customers, Average Churn Probability, Retention
   Opportunities) and the risk-distribution donut chart renders.
2. **Predict Churn (`/predict`)** — the form loads pre-filled with sensible
   defaults. Try:
   - The defaults as-is (month-to-month, fiber, electronic check, low
     tenure) → should come back **High risk**.
   - Switch Contract to "Two year", Tenure to 60, InternetService to "DSL",
     OnlineSecurity/TechSupport to "Yes" → should come back **Low risk**.
   - Check that the SHAP factor chart shows orange bars (risk-increasing)
     and green bars (risk-decreasing), and that the recommendation cards
     are relevant to the top factors shown.
3. **Customer Insights (`/insights`)** — confirm all six charts render
   (churn distribution, and churn rate by contract/tenure/monthly
   charges/internet service/payment method).
4. **API directly**, e.g.:
   ```bash
   curl http://127.0.0.1:5000/api/health
   curl http://127.0.0.1:5000/api/dashboard-stats
   curl -X POST http://127.0.0.1:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"gender":"Female","SeniorCitizen":0,"Partner":"No","Dependents":"No",
          "tenure":12,"PhoneService":"Yes","MultipleLines":"No",
          "InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No",
          "DeviceProtection":"No","TechSupport":"No","StreamingTV":"No",
          "StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes",
          "PaymentMethod":"Electronic check","MonthlyCharges":70,"TotalCharges":840}'
   ```
5. **Error handling** — POST to `/api/predict` with a missing field returns
   `400` with a `missing_fields` list; a non-JSON body returns `400`; an
   unknown route returns `404`. All were tested during development.

## Notes on risk thresholds

Churn probability is bucketed as: **Low** &lt; 30%, **Medium** 30–60%,
**High** &ge; 60%. These live in `RISK_THRESHOLDS` in `ml_engine.py` if you
want to tune them.

## Files you can delete

- Nothing else is required — `notebooks/` is kept for reference/reproducibility
  but isn't imported by the running app except for the trained model file
  inside `notebooks/models/`.
