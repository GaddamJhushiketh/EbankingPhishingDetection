"""Application configuration."""

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


class Config:
    """Base configuration shared by all environments."""

    SECRET_KEY = os.getenv("SECRET_KEY", "development-only-secret-key")
    PROJECT_ROOT = BASE_DIR
    INSTANCE_PATH = BASE_DIR / "instance"
    DATA_PATH = BASE_DIR / "data"
    MODEL_PATH = BASE_DIR / "models"


class DevelopmentConfig(Config):
    """Configuration used during local development."""

    DEBUG = True


class TestingConfig(Config):
    """Configuration used by automated tests."""

    TESTING = True
    SECRET_KEY = "testing-secret-key"


class ProductionConfig(Config):
    """Configuration used for production deployments."""

    DEBUG = False


CONFIGURATIONS = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}
