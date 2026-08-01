"""
Customer Retention Intelligence Platform
ML Engine

Owns everything related to the trained model:
  - loading + cleaning the raw Telco dataset
  - rebuilding the exact preprocessing pipeline used during training
  - loading the trained XGBoost model
  - live single-customer prediction with SHAP-based explanations
  - full-dataset scoring for dashboard / insights endpoints

IMPORTANT NOTE ON THE PREPROCESSOR
-----------------------------------
The training notebooks (03_Data_Preprocessing.ipynb) fit a scikit-learn
ColumnTransformer (median-impute + StandardScaler for numeric columns,
most-frequent-impute + OneHotEncoder for categorical columns) but never
serialized the *fitted* ColumnTransformer object to disk - only its
*output* (X_test_processed.pkl, feature_names.pkl) was saved.

To serve live predictions we must reproduce that exact fitted transformer.
This module rebuilds it deterministically (same cleaning steps, same
train_test_split random_state=42 / test_size=0.20 / stratify=y) and its
output has been verified to be numerically identical (np.allclose) to the
notebook's saved X_test_processed.pkl. See dev notes in README.md.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger("retention-platform.ml_engine")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "notebooks" / "models" / "final_xgboost_churn_model.pkl"

# ---------------------------------------------------------------------------
# Raw feature schema (exactly what the model was trained on)
# ---------------------------------------------------------------------------

NUMERIC_FIELDS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FIELDS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

RAW_FIELD_ORDER = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

# Human-friendly labels + short explanations used in the UI / recommendations
FIELD_LABELS = {
    "gender": "Gender",
    "SeniorCitizen": "Senior Citizen Status",
    "Partner": "Has a Partner",
    "Dependents": "Has Dependents",
    "tenure": "Tenure (months)",
    "PhoneService": "Phone Service",
    "MultipleLines": "Multiple Lines",
    "InternetService": "Internet Service Type",
    "OnlineSecurity": "Online Security Add-on",
    "OnlineBackup": "Online Backup Add-on",
    "DeviceProtection": "Device Protection Add-on",
    "TechSupport": "Tech Support Add-on",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
    "Contract": "Contract Type",
    "PaperlessBilling": "Paperless Billing",
    "PaymentMethod": "Payment Method",
    "MonthlyCharges": "Monthly Charges",
    "TotalCharges": "Total Charges To Date",
}

REQUIRED_FIELDS = list(RAW_FIELD_ORDER)

DEFAULT_VALUES = {
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
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0,
}

RISK_THRESHOLDS = (0.30, 0.60)  # Low < 0.30 <= Medium < 0.60 <= High


# ---------------------------------------------------------------------------
# Retention recommendation rules
# Keyed by raw field name -> callable(value, row) -> (condition, action text)
# Only fires when the field is among the top churn-increasing factors.
# ---------------------------------------------------------------------------

def _recommend_for_field(field, value):
    recs = {
        "Contract": {
            "Month-to-month": "Offer a discounted upgrade to a 1-year or 2-year contract to lock in loyalty.",
        },
        "PaymentMethod": {
            "Electronic check": "Encourage a switch to automatic payment (bank transfer or credit card) with a small incentive.",
        },
        "InternetService": {
            "Fiber optic": "Review fiber pricing/reliability complaints; consider a service credit or speed/price review.",
        },
        "TechSupport": {
            "No": "Offer a free trial of the Tech Support add-on to reduce service friction.",
        },
        "OnlineSecurity": {
            "No": "Offer a free trial of the Online Security add-on to increase perceived value.",
        },
        "OnlineBackup": {
            "No": "Bundle Online Backup at a discount to increase service stickiness.",
        },
        "DeviceProtection": {
            "No": "Offer Device Protection as part of a loyalty bundle.",
        },
        "PaperlessBilling": {
            "Yes": "Confirm billing clarity - unexpected paperless billing charges are a common churn trigger.",
        },
        "MultipleLines": {
            "No": "Cross-sell a Multiple Lines plan with a bundled discount.",
        },
        "Partner": {},
        "Dependents": {},
        "StreamingTV": {},
        "StreamingMovies": {},
        "PhoneService": {},
        "gender": {},
    }
    return recs.get(field, {}).get(value)


def tenure_recommendation(tenure_value):
    if tenure_value is not None and tenure_value <= 6:
        return "Assign a dedicated onboarding specialist - new customers in their first months are highest-risk."
    return "Schedule a proactive loyalty check-in call before the next renewal."


def charges_recommendation():
    return "Offer a personalized discount or bundle pricing to improve perceived value for the price paid."


def senior_recommendation():
    return "Provide simplified, higher-touch customer support tailored to senior customers."


# ---------------------------------------------------------------------------
# Data loading / cleaning
# ---------------------------------------------------------------------------

def load_and_clean_data() -> pd.DataFrame:
    """Loads the raw Telco churn CSV and applies the exact cleaning steps
    used in the training notebooks (03_Data_Preprocessing.ipynb)."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates()

    df = df.reset_index(drop=True)
    return df


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Rebuilds the exact ColumnTransformer from 03_Data_Preprocessing.ipynb
    and fits it on X_train. Verified to reproduce the notebook's saved
    X_test_processed.pkl output exactly (np.allclose, atol=1e-6)."""

    numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_transformer, numerical_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    preprocessor.fit(X_train)
    return preprocessor


def build_raw_field_to_columns_map(preprocessor: ColumnTransformer) -> dict:
    """Maps each raw input field name to the list of column indices it
    expands to in the transformed (one-hot encoded) feature matrix, so
    per-row SHAP contributions can be grouped back to human-readable
    fields."""

    transformed_names = preprocessor.get_feature_names_out().tolist()
    mapping: dict[str, list[int]] = {field: [] for field in RAW_FIELD_ORDER}

    for idx, name in enumerate(transformed_names):
        if name.startswith("numerical__"):
            raw_field = name[len("numerical__"):]
        elif name.startswith("categorical__"):
            remainder = name[len("categorical__"):]
            # remainder looks like "<RawField>_<Category>" - match against
            # known raw field names (longest match first, since some
            # category values themselves contain underscores).
            raw_field = None
            for candidate in sorted(CATEGORICAL_FIELDS, key=len, reverse=True):
                if remainder == candidate or remainder.startswith(candidate + "_"):
                    raw_field = candidate
                    break
            if raw_field is None:
                continue
        else:
            continue

        mapping.setdefault(raw_field, []).append(idx)

    return mapping


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ChurnEngine:
    """Holds the fitted preprocessor, trained model and SHAP explainer, and
    exposes prediction + dataset scoring methods. Built once at app
    startup."""

    def __init__(self):
        self.ready = False
        self.error = None

        self.full_df = None
        self.preprocessor = None
        self.model = None
        self.explainer = None
        self.field_to_columns = None
        self.scored_df = None

    def initialize(self):
        try:
            logger.info("Loading and cleaning dataset from %s", DATA_PATH)
            self.full_df = load_and_clean_data()

            X_all = self.full_df.drop(columns=["Churn"])
            y_all = self.full_df["Churn"]

            X_train, _X_test, y_train, _y_test = train_test_split(
                X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
            )

            logger.info("Fitting preprocessing pipeline on training split (%d rows)", len(X_train))
            self.preprocessor = build_preprocessor(X_train)
            self.field_to_columns = build_raw_field_to_columns_map(self.preprocessor)

            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Trained model not found at: {MODEL_PATH}")

            logger.info("Loading trained XGBoost model from %s", MODEL_PATH)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = joblib.load(MODEL_PATH)

            import shap  # imported lazily; heavy dependency

            logger.info("Building SHAP TreeExplainer")
            self.explainer = shap.TreeExplainer(self.model)

            logger.info("Scoring full dataset (%d customers) for dashboard/insights", len(X_all))
            self.scored_df = self._score_dataframe(X_all)
            self.scored_df["Churn"] = y_all.values

            self.ready = True
            self.error = None
            logger.info("ChurnEngine ready.")

        except Exception as exc:  # noqa: BLE001
            self.ready = False
            self.error = str(exc)
            logger.exception("Failed to initialize ChurnEngine.")

    # ------------------------------------------------------------------
    def _coerce_row(self, payload: dict) -> pd.DataFrame:
        row = {}
        for field in RAW_FIELD_ORDER:
            value = payload.get(field, DEFAULT_VALUES[field])

            if field in NUMERIC_FIELDS:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = float(DEFAULT_VALUES[field])
                if field == "SeniorCitizen":
                    value = 1 if value >= 0.5 else 0
                if field == "tenure":
                    value = max(0, min(72, round(value)))
                if field in ("MonthlyCharges", "TotalCharges"):
                    value = max(0.0, value)
            else:
                value = str(value) if value is not None else DEFAULT_VALUES[field]

            row[field] = value

        return pd.DataFrame([row], columns=RAW_FIELD_ORDER)

    def _score_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        X_processed = self.preprocessor.transform(X)
        probabilities = self.model.predict_proba(X_processed)[:, 1]

        out = X.reset_index(drop=True).copy()
        out["Churn_Probability"] = probabilities
        out["Risk_Level"] = [risk_level_label(p) for p in probabilities]
        return out

    # ------------------------------------------------------------------
    def predict(self, payload: dict) -> dict:
        if not self.ready:
            raise RuntimeError(self.error or "Engine not ready.")

        input_row = self._coerce_row(payload)
        X_processed = self.preprocessor.transform(input_row)

        probability = float(self.model.predict_proba(X_processed)[0, 1])
        prediction = "Yes" if probability >= 0.5 else "No"
        risk = risk_level_label(probability)

        shap_row = self.explainer.shap_values(X_processed)
        shap_row = np.asarray(shap_row).reshape(-1)

        field_impacts = []
        for field in RAW_FIELD_ORDER:
            col_indices = self.field_to_columns.get(field, [])
            if not col_indices:
                continue
            impact = float(np.sum(shap_row[col_indices]))
            field_impacts.append({
                "field": field,
                "label": FIELD_LABELS.get(field, field),
                "value": to_native(input_row.iloc[0][field]),
                "impact": impact,
            })

        field_impacts.sort(key=lambda item: abs(item["impact"]), reverse=True)
        max_abs_impact = max((abs(f["impact"]) for f in field_impacts), default=1.0) or 1.0

        top_factors = []
        for item in field_impacts[:8]:
            top_factors.append({
                "field": item["field"],
                "label": item["label"],
                "value": item["value"],
                "direction": "increases_risk" if item["impact"] > 0 else "decreases_risk",
                "impact": round(item["impact"], 5),
                "relative_strength": round(abs(item["impact"]) / max_abs_impact, 4),
            })

        recommendations = self._build_recommendations(
            risk, field_impacts, input_row.iloc[0]
        )

        explanation = self._build_explanation_sentence(probability, risk, top_factors)

        return {
            "prediction": prediction,
            "churn_probability": round(probability, 4),
            "risk_level": risk,
            "explanation": explanation,
            "top_factors": top_factors,
            "recommendations": recommendations,
            "input_echo": {k: to_native(v) for k, v in input_row.iloc[0].to_dict().items()},
        }

    def _build_explanation_sentence(self, probability, risk, top_factors):
        increasing = [f for f in top_factors if f["direction"] == "increases_risk"][:3]
        decreasing = [f for f in top_factors if f["direction"] == "decreases_risk"][:2]

        pct = round(probability * 100, 1)

        if risk == "Low":
            base = (
                f"This customer has a {pct}% predicted probability of churning, "
                f"which is low risk. The current profile looks stable."
            )
        elif risk == "Medium":
            base = (
                f"This customer has a {pct}% predicted probability of churning, "
                f"a medium level of risk worth monitoring."
            )
        else:
            base = (
                f"This customer has a {pct}% predicted probability of churning, "
                f"which is high risk and warrants proactive retention action."
            )

        if increasing:
            names = ", ".join(f["label"] for f in increasing)
            base += f" The biggest factors pushing risk up are: {names}."

        if decreasing:
            names = ", ".join(f["label"] for f in decreasing)
            base += f" Factors helping retention: {names}."

        return base

    def _build_recommendations(self, risk, field_impacts, input_row):
        recommendations = []
        seen = set()

        positive_factors = [f for f in field_impacts if f["impact"] > 0]

        for item in positive_factors[:6]:
            field = item["field"]
            value = item["value"]

            if field == "tenure":
                text = tenure_recommendation(value)
            elif field in ("MonthlyCharges", "TotalCharges"):
                text = charges_recommendation()
            elif field == "SeniorCitizen" and int(value) == 1:
                text = senior_recommendation()
            else:
                text = _recommend_for_field(field, value)

            if text and text not in seen:
                recommendations.append(text)
                seen.add(text)

            if len(recommendations) >= 5:
                break

        if risk in ("High",) and "Contact the customer directly for a retention conversation." not in seen:
            recommendations.insert(0, "Contact the customer directly for a retention conversation.")

        if not recommendations:
            recommendations.append(
                "No urgent retention action needed - continue standard engagement."
            )

        return recommendations[:6]

    # ------------------------------------------------------------------
    def dashboard_stats(self) -> dict:
        if not self.ready:
            raise RuntimeError(self.error or "Engine not ready.")

        df = self.scored_df
        total_customers = len(df)
        high_risk = int((df["Risk_Level"] == "High").sum())
        medium_risk = int((df["Risk_Level"] == "Medium").sum())
        avg_prob = float(df["Churn_Probability"].mean())
        retention_opportunities = int(df["Risk_Level"].isin(["Medium", "High"]).sum())

        return {
            "total_customers": total_customers,
            "high_risk_customers": high_risk,
            "medium_risk_customers": medium_risk,
            "average_churn_probability": round(avg_prob, 4),
            "retention_opportunities": retention_opportunities,
        }

    def risk_distribution(self) -> dict:
        if not self.ready:
            raise RuntimeError(self.error or "Engine not ready.")

        order = ["Low", "Medium", "High"]
        counts = self.scored_df["Risk_Level"].value_counts()
        return {label: int(counts.get(label, 0)) for label in order}

    def insights(self) -> dict:
        if not self.ready:
            raise RuntimeError(self.error or "Engine not ready.")

        df = self.scored_df

        churn_distribution = {
            "Churned": int((df["Churn"] == 1).sum()),
            "Retained": int((df["Churn"] == 0).sum()),
        }

        by_contract = _clean_rate_dict(
            df.groupby("Contract")["Churn"].mean().mul(100).round(2)
        )

        tenure_bins = [0, 12, 24, 48, 1000]
        tenure_labels = ["0-12 mo", "13-24 mo", "25-48 mo", "49+ mo"]
        df_tenure = df.copy()
        df_tenure["TenureGroup"] = pd.cut(
            df_tenure["tenure"], bins=tenure_bins, labels=tenure_labels, include_lowest=True
        )
        by_tenure = _clean_rate_dict(
            df_tenure.groupby("TenureGroup", observed=False)["Churn"]
            .mean().mul(100).round(2).reindex(tenure_labels)
        )

        charge_bins = [0, 35, 70, 105, 1000]
        charge_labels = ["$0-35", "$35-70", "$70-105", "$105+"]
        df_charges = df.copy()
        df_charges["ChargeGroup"] = pd.cut(
            df_charges["MonthlyCharges"], bins=charge_bins, labels=charge_labels, include_lowest=True
        )
        by_charges = _clean_rate_dict(
            df_charges.groupby("ChargeGroup", observed=False)["Churn"]
            .mean().mul(100).round(2).reindex(charge_labels)
        )

        by_internet = _clean_rate_dict(
            df.groupby("InternetService")["Churn"].mean().mul(100).round(2)
        )

        by_payment = _clean_rate_dict(
            df.groupby("PaymentMethod")["Churn"].mean().mul(100).round(2)
        )

        return {
            "churn_distribution": churn_distribution,
            "churn_rate_by_contract": by_contract,
            "churn_rate_by_tenure_group": by_tenure,
            "churn_rate_by_monthly_charges": by_charges,
            "churn_rate_by_internet_service": by_internet,
            "churn_rate_by_payment_method": by_payment,
        }


def to_native(value):
    """Converts numpy/pandas scalar types to plain Python types so the
    Flask JSON encoder never chokes on them (e.g. numpy.int64)."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _clean_rate_dict(series: pd.Series) -> dict:
    """Converts a grouped rate Series to a plain dict, mapping any NaN
    (empty bucket) to None so it serializes safely as JSON."""
    result = {}
    for key, value in series.to_dict().items():
        result[str(key)] = None if pd.isna(value) else to_native(value)
    return result


def risk_level_label(probability: float) -> str:
    low_cut, high_cut = RISK_THRESHOLDS
    if probability < low_cut:
        return "Low"
    if probability < high_cut:
        return "Medium"
    return "High"
