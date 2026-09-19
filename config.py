"""Application configuration."""

import os
import secrets
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


class Config:
    """Base configuration shared by all environments."""

    SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_urlsafe(32)
    PROJECT_ROOT = BASE_DIR
    INSTANCE_PATH = BASE_DIR / "instance"
    DATA_PATH = BASE_DIR / "data"
    MODEL_PATH = BASE_DIR / "models"
    MODEL_FILE = MODEL_PATH / "associative_classifier.pkl"
    MODEL_SHA256 = os.getenv("MODEL_SHA256")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL") or (
        f"mysql+pymysql://{os.getenv('DB_USER', '')}:{os.getenv('DB_PASSWORD', '')}"
        f"@{os.getenv('DB_HOST', '')}:{os.getenv('DB_PORT', '3306')}/{os.getenv('DB_NAME', 'phishing')}"
        if os.getenv("DB_HOST") else "sqlite:///history.db"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class DevelopmentConfig(Config):
    """Configuration used during local development."""

    DEBUG = True


class TestingConfig(Config):
    """Configuration used by automated tests."""

    TESTING = True
    SECRET_KEY = "testing-secret-key"
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"


class ProductionConfig(Config):
    """Configuration used for production deployments."""

    DEBUG = False

    @classmethod
    def init_app(cls, application):
        if not os.getenv("SECRET_KEY"):
            raise RuntimeError("SECRET_KEY must be configured in production")
        if not os.getenv("MODEL_SHA256"):
            raise RuntimeError("MODEL_SHA256 must be configured in production")


CONFIGURATIONS = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}
