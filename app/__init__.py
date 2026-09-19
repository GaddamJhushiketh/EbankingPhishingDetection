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

    Path(application.instance_path).mkdir(parents=True, exist_ok=True)

    from app.routes.main import main_blueprint

    application.register_blueprint(main_blueprint)

    return application
