"""
Customer Retention Intelligence Platform
Flask Backend

Run:
    python app.py

Then open:
    http://127.0.0.1:5000
"""

import logging

from flask import Flask, jsonify, render_template, request

from ml_engine import DEFAULT_VALUES, FIELD_LABELS, RAW_FIELD_ORDER, ChurnEngine

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("retention-platform")


# ============================================================================
# FLASK APP + ML ENGINE
# ============================================================================

app = Flask(__name__)

engine = ChurnEngine()
engine.initialize()


def require_engine():
    if not engine.ready:
        return jsonify({
            "error": "Model engine unavailable.",
            "detail": engine.error,
        }), 503
    return None


# ============================================================================
# PAGE ROUTES
# ============================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict")
def predict_page():
    return render_template("predict.html")


@app.route("/insights")
def insights_page():
    return render_template("insights.html")


# ============================================================================
# HEALTH
# ============================================================================

@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok" if engine.ready else "error",
        "model_ready": engine.ready,
        "detail": engine.error,
    })


# ============================================================================
# FORM SCHEMA (drives the prediction form dynamically on the frontend)
# ============================================================================

FIELD_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


@app.route("/api/form-schema")
def api_form_schema():
    return jsonify({
        "field_order": RAW_FIELD_ORDER,
        "labels": FIELD_LABELS,
        "options": FIELD_OPTIONS,
        "defaults": DEFAULT_VALUES,
    })


# ============================================================================
# PREDICTION
# ============================================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():
    err = require_engine()
    if err:
        return err

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be JSON."}), 400

    missing = [f for f in RAW_FIELD_ORDER if f not in payload or payload[f] in (None, "")]
    if missing:
        return jsonify({
            "error": "Missing required fields.",
            "missing_fields": missing,
        }), 400

    try:
        result = engine.predict(payload)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed.")
        return jsonify({"error": "Prediction failed.", "detail": str(exc)}), 500

    return jsonify(result)


# ============================================================================
# DASHBOARD / INSIGHTS
# ============================================================================

@app.route("/api/dashboard-stats")
def api_dashboard_stats():
    err = require_engine()
    if err:
        return err
    return jsonify(engine.dashboard_stats())


@app.route("/api/risk-distribution")
def api_risk_distribution():
    err = require_engine()
    if err:
        return err
    return jsonify(engine.risk_distribution())


@app.route("/api/insights")
def api_insights():
    err = require_engine()
    if err:
        return err
    return jsonify(engine.insights())


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Resource not found."}), 404


@app.errorhandler(500)
def server_error(error):
    logger.exception("Unhandled server error: %s", error)
    return jsonify({"error": "Internal server error."}), 500


# ============================================================================
# START APPLICATION
# ============================================================================

if __name__ == "__main__":
    print("Starting Customer Retention Intelligence...")
    print("Model engine ready:", engine.ready)
    if not engine.ready:
        print("Model engine error:", engine.error)

    # use_reloader=False: the reloader re-executes this module in a child
    # process, which would reload the dataset/model/SHAP explainer twice on
    # every startup (SHAP's first import triggers a ~10s numba JIT warm-up).
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5000)
