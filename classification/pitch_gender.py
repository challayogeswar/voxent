import librosa
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class PitchGenderClassifier:
    """Pitch-based gender classification for voice samples."""

    def __init__(self, male_threshold: float = 85.0, female_threshold: float = 165.0):
        """
        Initialize pitch-based gender classifier.

        Args:
            male_threshold: Pitch threshold below which samples are classified as male (Hz)
            female_threshold: Pitch threshold above which samples are classified as female (Hz)
        """
        self.male_threshold = male_threshold
        self.female_threshold = female_threshold

    def estimate_pitch(self, audio: np.ndarray, sr: int) -> float:
        """Estimate the fundamental frequency (pitch) of audio."""
        pitches, mags = librosa.piptrack(y=audio, sr=sr)

        # Get pitches above median magnitude
        pitch_values = pitches[mags > np.median(mags)]

        if len(pitch_values) == 0:
            return 0.0

        # Return mean of non-zero pitches
        non_zero_pitches = pitch_values[pitch_values > 0]
        return np.mean(non_zero_pitches) if len(non_zero_pitches) > 0 else 0.0

    def classify(self, audio: np.ndarray, sr: int) -> Tuple[str, float]:
        """
        Classify gender based on pitch.

        Returns:
            Tuple of (gender_label, confidence_score)
        """
        pitch = self.estimate_pitch(audio, sr)

        if pitch == 0.0:
            # Cannot determine pitch
            return "unknown", 0.0

        if pitch < self.male_threshold:
            # Likely male
            confidence = min(100.0, (self.male_threshold - pitch) / self.male_threshold * 100)
            return "male", confidence
        elif pitch > self.female_threshold:
            # Likely female
            confidence = min(100.0, (pitch - self.female_threshold) / (300 - self.female_threshold) * 100)
            return "female", confidence
        else:
            # Ambiguous range - lower confidence
            distance_to_male = abs(pitch - self.male_threshold)
            distance_to_female = abs(pitch - self.female_threshold)

            if distance_to_male < distance_to_female:
                confidence = max(10.0, (self.female_threshold - pitch) / (self.female_threshold - self.male_threshold) * 50)
                return "male", confidence
            else:
                confidence = max(10.0, (pitch - self.male_threshold) / (self.female_threshold - self.male_threshold) * 50)
                return "female", confidence

def estimate_pitch(audio, sr):
    """Legacy function for backward compatibility."""
    classifier = PitchGenderClassifier()
    return classifier.estimate_pitch(audio, sr)
