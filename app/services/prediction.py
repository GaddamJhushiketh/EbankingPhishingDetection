"""One-time verified model loading and prediction orchestration."""
from __future__ import annotations
import logging
from pathlib import Path
from app.ml.associative_classifier import AssociativeClassifier
from app.utils.model_integrity import load_verified_model
from .feature_extraction import FeatureExtractionService
from .explanation import ExplanationService
from .dataset379_encoder import Dataset379Encoder

log = logging.getLogger(__name__)


class PredictionService:
    def __init__(self, model_path: Path, *, expected_hash=None, production=False, extractor=None):
        self.model_path = Path(model_path).resolve()
        self.extractor = extractor or FeatureExtractionService()
        self.encoder = Dataset379Encoder()
        self.model = load_verified_model(
            self.model_path, AssociativeClassifier.load,
            expected_hash=expected_hash, production=production,
        )
        self.explanations = ExplanationService(self.model)

    def analyze_url(self, url: str) -> dict[str, object]:
        return self.extractor.analyze(url)

    def encode_live_observations(self, observations: dict[str, object]) -> dict[str, object]:
        return self.encoder.encode(observations)

    def predict_features(self, features: dict[str, int]) -> tuple[int, dict[str, int]]:
        from app.ml.preprocessing import validate_feature_vector
        validated = validate_feature_vector(features)
        return self.model.predict_one(validated), validated

    def explain_features(self, features: dict[str, int]) -> tuple[int, dict[str, int], dict]:
        prediction, validated = self.predict_features(features)
        return prediction, validated, self.explanations.explain(validated)
