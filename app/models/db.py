"""Database extension and prediction history model."""
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


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
