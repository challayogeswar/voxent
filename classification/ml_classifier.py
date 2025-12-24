import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import librosa
import soundfile as sf
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class MLGenderClassifier:
    """Machine Learning-based gender classification for voice samples."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.feature_names = [
            'pitch_mean', 'pitch_std', 'pitch_range',
            'mfcc_mean', 'mfcc_std', 'mfcc_range',
            'chroma_mean', 'chroma_std',
            'spectral_centroid', 'spectral_bandwidth',
            'zero_crossing_rate', 'rms_energy'
        ]

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract acoustic features from audio for classification."""
        features = []

        # Pitch-based features
        pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
        pitch_values = pitches[magnitudes > np.max(magnitudes) * 0.1]
        if len(pitch_values) > 0:
            features.extend([
                np.mean(pitch_values),  # pitch_mean
                np.std(pitch_values),   # pitch_std
                np.max(pitch_values) - np.min(pitch_values)  # pitch_range
            ])
        else:
            features.extend([0, 0, 0])

        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        features.extend([
            np.mean(mfccs, axis=1).mean(),  # mfcc_mean
            np.std(mfccs, axis=1).mean(),   # mfcc_std
            np.max(mfccs) - np.min(mfccs)   # mfcc_range
        ])

        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
        features.extend([
            np.mean(chroma),  # chroma_mean
            np.std(chroma)    # chroma_std
        ])

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)
        features.extend([
            np.mean(spectral_centroid),    # spectral_centroid
            np.mean(spectral_bandwidth)    # spectral_bandwidth
        ])

        # Other features
        zcr = librosa.feature.zero_crossing_rate(audio)
        rms = librosa.feature.rms(y=audio)
        features.extend([
            np.mean(zcr),  # zero_crossing_rate
            np.mean(rms)   # rms_energy
        ])

        return np.array(features)

    def load_training_data(self, metadata_file: str, dataset_dir: str,
                          min_confidence: float = 70.0) -> Tuple[np.ndarray, np.ndarray]:
        """Load verified training data from metadata and audio files."""
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        # Load metadata
        df = pd.read_csv(metadata_file)

        # Filter high-confidence samples
        verified_df = df[df['confidence'] >= min_confidence].copy()

        if len(verified_df) == 0:
            raise ValueError(f"No samples found with confidence >= {min_confidence}")

        logger.info(f"Loading {len(verified_df)} verified samples for training")

        features_list = []
        labels_list = []

        for _, row in verified_df.iterrows():
            try:
                # Construct file path
                file_path = os.path.join(dataset_dir, row['label'], row['file'])

                if not os.path.exists(file_path):
                    continue

                # Load audio
                audio, sr = sf.read(file_path)

                # Resample if necessary
                if sr != self.sample_rate:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

                # Extract features
                features = self.extract_features(audio)
                features_list.append(features)
                labels_list.append(1 if row['label'] == 'male' else 0)  # 1=male, 0=female

            except Exception as e:
                logger.warning(f"Error processing {row['file']}: {e}")
                continue

        if len(features_list) == 0:
            raise ValueError("No valid training samples found")

        X = np.array(features_list)
        y = np.array(labels_list)

        logger.info(f"Successfully loaded {len(X)} training samples")
        return X, y

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, float]:
        """Train the ML classifier."""
        logger.info("Training ML gender classifier...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        # Get detailed metrics
        report = classification_report(y_test, y_pred, target_names=['female', 'male'], output_dict=True)

        self.is_trained = True

        results = {
            'accuracy': accuracy,
            'precision_male': report['male']['precision'],
            'recall_male': report['male']['recall'],
            'f1_male': report['male']['f1-score'],
            'precision_female': report['female']['precision'],
            'recall_female': report['female']['recall'],
            'f1_female': report['female']['f1-score']
        }

        logger.info(".2%")
        return results

    def predict(self, audio: np.ndarray) -> Tuple[str, float]:
        """Predict gender from audio features."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Extract features
        features = self.extract_features(audio)

        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]

        label = 'male' if prediction == 1 else 'female'
        confidence = probabilities[prediction] * 100  # Convert to percentage

        return label, confidence

    def save_model(self, filepath: str):
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model not trained. Cannot save.")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'sample_rate': self.sample_rate,
            'is_trained': self.is_trained
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.sample_rate = model_data['sample_rate']
        self.is_trained = model_data['is_trained']

        logger.info(f"Model loaded from {filepath}")

def train_ml_classifier(metadata_file: str = "data/voice_dataset/metadata.csv",
                       dataset_dir: str = "data/voice_dataset",
                       model_path: str = "models/ml_gender_classifier.pkl",
                       min_confidence: float = 70.0) -> Dict[str, float]:
    """Convenience function to train and save ML classifier."""
    # Create models directory
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Initialize classifier
    classifier = MLGenderClassifier()

    # Load training data
    X, y = classifier.load_training_data(metadata_file, dataset_dir, min_confidence)

    # Train model
    results = classifier.train(X, y)

    # Save model
    classifier.save_model(model_path)

    return results

def load_ml_classifier(model_path: str = "models/ml_gender_classifier.pkl") -> MLGenderClassifier:
    """Load a trained ML classifier."""
    classifier = MLGenderClassifier()
    classifier.load_model(model_path)
    return classifier

if __name__ == "__main__":
    # Example usage
    try:
        results = train_ml_classifier()
        print("Training Results:")
        for metric, value in results.items():
            print(".3f")
    except Exception as e:
        print(f"Training failed: {e}")