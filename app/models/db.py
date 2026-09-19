"""Database extension and prediction history model."""
from sqlalchemy import inspect, text
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def ensure_history_schema() -> None:
    """Apply only additive compatibility changes to an existing history table."""

    columns = {column["name"] for column in inspect(db.engine).get_columns("prediction_history")}
    if "prediction_status" not in columns:
        db.session.execute(
            text(
                "ALTER TABLE prediction_history "
                "ADD COLUMN prediction_status VARCHAR(40) "
                "NOT NULL DEFAULT 'predicted'"
            )
        )
        db.session.commit()


class PredictionHistory(db.Model):
    __tablename__ = "prediction_history"
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(2048), nullable=False)
    prediction = db.Column(db.Integer, nullable=True)
    label = db.Column(db.String(32), nullable=True)
    features = db.Column(db.JSON, nullable=False)
    prediction_status = db.Column(
        db.String(40), nullable=False, default="predicted", server_default="predicted"
    )
    created_at = db.Column(db.DateTime, server_default=db.func.now(), nullable=False)
