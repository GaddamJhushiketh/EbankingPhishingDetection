"""Flask application package."""

import os
from pathlib import Path

from flask import Flask

from config import CONFIGURATIONS


def create_app(config_name=None):
    """Create and configure the Flask application."""

    selected_config = config_name or os.getenv("FLASK_ENV", "development")
    config_class = CONFIGURATIONS.get(selected_config, CONFIGURATIONS["development"])

    application = Flask(__name__, instance_relative_config=True)
    application.config.from_object(config_class)
    if selected_config == "production":
        config_class.init_app(application)

    Path(application.instance_path).mkdir(parents=True, exist_ok=True)

    from app.models.db import db, ensure_history_schema
    db.init_app(application)
    with application.app_context():
        db.create_all()
        ensure_history_schema()

    from app.services.prediction import PredictionService
    from app.services.domain_information import DomainInformationService
    from app.services.traffic_provider import TrafficProvider
    from app.utils.model_integrity import ModelIntegrityError
    try:
        application.extensions["prediction_service"] = PredictionService(
            application.config["MODEL_FILE"],
            expected_hash=application.config.get("MODEL_SHA256"),
            production=selected_config == "production",
            domain_information=(
                DomainInformationService(timeout=application.config["DOMAIN_INFO_TIMEOUT"])
                if application.config.get("DOMAIN_INFO_PROVIDER") == "whois"
                else DomainInformationService(client=None)
            ),
            traffic_provider=TrafficProvider(),
        )
    except ModelIntegrityError:
        application.extensions["prediction_service"] = None

    from app.routes.main import main_blueprint
    from app.services.analytics import AnalyticsService
    application.extensions["analytics_service"] = AnalyticsService(
        model=application.extensions.get("prediction_service"),
        root=Path(application.root_path).parent,
        integrity_configured=bool(application.config.get("MODEL_SHA256")),
    )

    application.register_blueprint(main_blueprint)

    return application
