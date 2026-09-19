"""Main application routes."""

from types import SimpleNamespace

from flask import Blueprint, current_app, jsonify, render_template, request
from sqlalchemy import text
from app.models.db import PredictionHistory, db
from app.services.feature_extraction import FeatureExtractionError
from app.services.url_security import URLValidationError, redact_url
from app.ml.preprocessing import (
    FEATURE_ALLOWED_VALUES,
    FEATURE_COLUMNS,
    TARGET_LABELS,
    validate_feature_vector,
)


main_blueprint = Blueprint("main", __name__)


@main_blueprint.get("/")
def index():
    """Render the project landing page."""

    return render_template("index.html")


@main_blueprint.get("/health")
def health():
    """Return a lightweight application health response."""

    ready = current_app.extensions.get("prediction_service") is not None
    return jsonify({"status": "ok" if ready else "degraded", "service": "ebanking-phishing-detection"}), (200 if ready else 503)


@main_blueprint.get("/status")
def system_status():
    model_loaded = current_app.extensions.get("prediction_service") is not None
    model_integrity = "Verified" if current_app.config.get("MODEL_SHA256") else "Not configured"
    database = "Available"
    try:
        db.session.execute(text("SELECT 1"))
    except Exception:
        database = "Unavailable"
    return render_template(
        "status.html",
        model_loaded=model_loaded,
        model_integrity=model_integrity,
        database=database,
        feature_extraction="Available observations / mapping restricted",
    )


@main_blueprint.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("index.html")
    payload = request.get_json(silent=True) or {}
    url = (request.form.get("url") or payload.get("url") or "").strip()
    service = current_app.extensions.get("prediction_service")
    if service is None:
        return render_template("result.html", error="Prediction model is unavailable."), 503
    try:
        observations = service.analyze_url(url)
        safe_url = redact_url(url)
        db.session.add(PredictionHistory(
            url=safe_url, prediction=None, label=None, features=observations,
            prediction_status="feature_mapping_unavailable",
        ))
        db.session.commit()
        return render_template(
            "result.html", url=safe_url, unavailable=True,
            observations=observations,
            error="Website analysis completed, but a model prediction was not produced because the original Dataset 379 feature-encoding rules could not be verified for all required features.",
        ), 200
    except (URLValidationError, FeatureExtractionError, ValueError) as exc:
        db.session.rollback()
        return render_template("result.html", error=str(exc), url=url), 400
    except Exception:
        db.session.rollback()
        current_app.logger.exception("Prediction request failed")
        return render_template("result.html", error="Unable to analyze this URL.", url=url), 502


@main_blueprint.route("/predict/features", methods=["GET", "POST"])
def predict_features():
    if request.method == "GET":
        return render_template(
            "feature_vector.html",
            feature_columns=FEATURE_COLUMNS,
            feature_values=FEATURE_ALLOWED_VALUES,
        )
    payload = request.get_json(silent=True) or request.form
    try:
        raw_features = {column: payload.get(column) for column in FEATURE_COLUMNS}
        prediction, features, explanation = current_app.extensions[
            "prediction_service"
        ].explain_features(raw_features)
        label = TARGET_LABELS[prediction]
        db.session.add(PredictionHistory(
            url="Dataset Feature Vector Test", prediction=prediction, label=label,
            features=features, prediction_status="predicted",
        ))
        db.session.commit()
        return render_template(
            "result.html", url="Dataset Feature Vector Test",
            prediction=prediction, label=label, features=features,
            explanation=explanation, feature_vector_test=True,
        )
    except (ValueError, FeatureExtractionError) as exc:
        db.session.rollback()
        return render_template(
            "result.html", url="Dataset Feature Vector Test",
            error=f"Model input unavailable: {exc}", feature_vector_test=True,
        ), 400
    except Exception:
        db.session.rollback()
        current_app.logger.exception("Feature-vector prediction failed")
        return render_template("result.html", error="Unable to process the feature vector."), 502


@main_blueprint.get("/history")
def history():
    rows = PredictionHistory.query.order_by(PredictionHistory.created_at.desc()).limit(100).all()
    service = current_app.extensions.get("prediction_service")
    history_rows = []
    for row in rows:
        rule_count = None
        if service is not None and row.prediction_status == "predicted":
            try:
                rule_count = service.explanations.explain(row.features)["matching_rule_count"]
            except (TypeError, ValueError):
                rule_count = None
        history_rows.append(
            SimpleNamespace(
                id=row.id,
                url=row.url,
                prediction_status=row.prediction_status,
                label=row.label,
                created_at=row.created_at,
                rule_count=rule_count,
            )
        )
    return render_template("history.html", history=history_rows)
