import os
import numpy as np
import logging
from typing import Tuple, Optional, Union
from .pitch_gender import PitchGenderClassifier
from .ml_classifier import MLGenderClassifier, load_ml_classifier

logger = logging.getLogger(__name__)

class IntegratedGenderClassifier:
    """Integrated gender classifier using both pitch-based and ML-based approaches."""

    def __init__(self, use_ml: bool = True, ml_model_path: str = "models/ml_gender_classifier.pkl",
                 pitch_male_threshold: float = 85.0, pitch_female_threshold: float = 165.0):
        """
        Initialize integrated classifier.

        Args:
            use_ml: Whether to use ML classifier (if available)
            ml_model_path: Path to trained ML model
            pitch_male_threshold: Pitch threshold for male classification
            pitch_female_threshold: Pitch threshold for female classification
        """
        self.use_ml = use_ml
        self.ml_model_path = ml_model_path
        self.pitch_classifier = PitchGenderClassifier(pitch_male_threshold, pitch_female_threshold)
        self.ml_classifier = None

        # Try to load ML classifier if requested
        if self.use_ml:
            try:
                if os.path.exists(self.ml_model_path):
                    self.ml_classifier = load_ml_classifier(self.ml_model_path)
                    logger.info("ML classifier loaded successfully")
                else:
                    logger.warning(f"ML model not found at {self.ml_model_path}, falling back to pitch-based classification")
            except Exception as e:
                logger.warning(f"Failed to load ML classifier: {e}, falling back to pitch-based classification")

    def classify(self, audio: np.ndarray, sr: int) -> Tuple[str, float]:
        """
        Classify gender using integrated approach.

        Priority:
        1. ML classifier (if available and trained)
        2. Pitch-based classifier (fallback)

        Returns:
            Tuple of (gender_label, confidence_score)
        """
        # Try ML classification first if available
        if self.ml_classifier and self.ml_classifier.is_trained:
            try:
                # Resample audio if necessary for ML classifier
                if sr != self.ml_classifier.sample_rate:
                    import librosa
                    audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=self.ml_classifier.sample_rate)
                else:
                    audio_resampled = audio

                ml_label, ml_confidence = self.ml_classifier.predict(audio_resampled)

                # If ML confidence is high enough, use it
                if ml_confidence >= 70.0:
                    logger.debug(".1f")
                    return ml_label, ml_confidence
                else:
                    logger.debug(".1f")

            except Exception as e:
                logger.warning(f"ML classification failed: {e}, falling back to pitch-based")

        # Fall back to pitch-based classification
        pitch_label, pitch_confidence = self.pitch_classifier.classify(audio, sr)

        logger.debug(".1f")
        return pitch_label, pitch_confidence

    def is_ml_available(self) -> bool:
        """Check if ML classifier is available and trained."""
        return self.ml_classifier is not None and self.ml_classifier.is_trained

    def get_classifier_info(self) -> dict:
        """Get information about available classifiers."""
        return {
            "ml_available": self.is_ml_available(),
            "ml_model_path": self.ml_model_path if self.is_ml_available() else None,
            "pitch_thresholds": {
                "male": self.pitch_classifier.male_threshold,
                "female": self.pitch_classifier.female_threshold
            }
        }

def create_classifier(config: dict) -> IntegratedGenderClassifier:
    """Factory function to create classifier from configuration."""
    use_ml = config.get('classification', {}).get('use_ml', True)
    ml_model_path = config.get('classification', {}).get('ml_model_path', 'models/ml_gender_classifier.pkl')
    pitch_male_threshold = config.get('classification', {}).get('pitch_male_threshold', 85.0)
    pitch_female_threshold = config.get('classification', {}).get('pitch_female_threshold', 165.0)

    return IntegratedGenderClassifier(
        use_ml=use_ml,
        ml_model_path=ml_model_path,
        pitch_male_threshold=pitch_male_threshold,
        pitch_female_threshold=pitch_female_threshold
    )

# Global classifier instance
_classifier_instance = None

def get_classifier(config: dict = None) -> IntegratedGenderClassifier:
    """Get or create global classifier instance."""
    global _classifier_instance

    if _classifier_instance is None:
        if config is None:
            # Default configuration
            config = {
                'classification': {
                    'use_ml': True,
                    'ml_model_path': 'models/ml_gender_classifier.pkl',
                    'pitch_male_threshold': 85.0,
                    'pitch_female_threshold': 165.0
                }
            }
        _classifier_instance = create_classifier(config)

    return _classifier_instance
