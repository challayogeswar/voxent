import os
import numpy as np
import soundfile as sf
import librosa
from typing import Dict, List, Tuple
import pandas as pd

class QualityMetrics:
    """Quality assessment metrics for voice dataset."""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def calculate_snr(self, audio: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio."""
        # Simple SNR calculation using signal power vs noise power
        signal_power = np.mean(audio ** 2)
        noise_power = np.var(audio) * 0.01  # Estimate noise as small percentage of variance
        if noise_power == 0:
            return 100.0  # Very high SNR if no noise detected
        return 10 * np.log10(signal_power / noise_power)

    def calculate_clipping_ratio(self, audio: np.ndarray, threshold: float = 0.99) -> float:
        """Calculate percentage of samples that are clipped."""
        return np.mean(np.abs(audio) >= threshold) * 100

    def calculate_silence_ratio(self, audio: np.ndarray, threshold: float = 0.01) -> float:
        """Calculate percentage of silence in audio."""
        return np.mean(np.abs(audio) < threshold) * 100

    def calculate_frequency_balance(self, audio: np.ndarray) -> Dict[str, float]:
        """Calculate frequency balance metrics."""
        # Compute FFT
        fft = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)

        # Frequency bands
        low_freq = (freqs >= 80) & (freqs < 300)    # Low frequencies
        mid_freq = (freqs >= 300) & (freqs < 3400)  # Mid frequencies
        high_freq = (freqs >= 3400) & (freqs < 8000) # High frequencies

        # Calculate power in each band
        low_power = np.mean(np.abs(fft[low_freq]) ** 2) if np.any(low_freq) else 0
        mid_power = np.mean(np.abs(fft[mid_freq]) ** 2) if np.any(mid_freq) else 0
        high_power = np.mean(np.abs(fft[high_freq]) ** 2) if np.any(high_freq) else 0

        total_power = low_power + mid_power + high_power
        if total_power == 0:
            return {"low_balance": 0, "mid_balance": 0, "high_balance": 0}

        return {
            "low_balance": low_power / total_power,
            "mid_balance": mid_power / total_power,
            "high_balance": high_power / total_power
        }

    def calculate_zero_crossing_rate(self, audio: np.ndarray) -> float:
        """Calculate zero crossing rate (related to noisiness)."""
        return np.mean(librosa.feature.zero_crossing_rate(audio))

    def assess_audio_quality(self, audio_path: str) -> Dict[str, float]:
        """Comprehensive audio quality assessment."""
        try:
            # Load audio
            audio, sr = sf.read(audio_path)

            # Ensure mono
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Resample if necessary
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

            # Calculate metrics
            snr = self.calculate_snr(audio)
            clipping = self.calculate_clipping_ratio(audio)
            silence = self.calculate_silence_ratio(audio)
            freq_balance = self.calculate_frequency_balance(audio)
            zcr = self.calculate_zero_crossing_rate(audio)

            # Overall quality score (0-100)
            quality_score = self._calculate_overall_quality(snr, clipping, silence, freq_balance, zcr)

            return {
                "snr": snr,
                "clipping_ratio": clipping,
                "silence_ratio": silence,
                "zero_crossing_rate": zcr,
                "quality_score": quality_score,
                **freq_balance
            }

        except Exception as e:
            print(f"Error assessing quality for {audio_path}: {e}")
            return {
                "snr": 0,
                "clipping_ratio": 100,
                "silence_ratio": 100,
                "zero_crossing_rate": 0,
                "quality_score": 0,
                "low_balance": 0,
                "mid_balance": 0,
                "high_balance": 0
            }

    def _calculate_overall_quality(self, snr: float, clipping: float, silence: float,
                                 freq_balance: Dict, zcr: float) -> float:
        """Calculate overall quality score from individual metrics."""
        score = 0

        # SNR score (good SNR > 20dB)
        if snr > 30:
            score += 30
        elif snr > 20:
            score += 20
        elif snr > 10:
            score += 10

        # Clipping penalty (less than 1% clipping is good)
        if clipping < 1:
            score += 20
        elif clipping < 5:
            score += 10

        # Silence penalty (less than 20% silence is good for voice)
        if silence < 20:
            score += 20
        elif silence < 50:
            score += 10

        # Frequency balance (good balance across bands)
        balance_score = 0
        if freq_balance["mid_balance"] > 0.4:  # Mid frequencies should dominate for voice
            balance_score += 15
        if freq_balance["high_balance"] > 0.1:  # Some high frequency content
            balance_score += 5
        score += balance_score

        # Zero crossing rate (moderate ZCR is good for voice)
        if 0.05 < zcr < 0.3:
            score += 10

        return min(100, score)

def assess_dataset_quality(dataset_dir: str) -> Dict[str, any]:
    """Assess quality of entire dataset."""
    print("Assessing dataset quality...")

    metrics = QualityMetrics()
    quality_data = []

    categories = ['male', 'female', 'uncertain']
    total_files = 0

    for category in categories:
        category_dir = os.path.join(dataset_dir, category)
        if not os.path.exists(category_dir):
            continue

        files = [f for f in os.listdir(category_dir) if f.endswith('.wav')]

        for file in files:
            file_path = os.path.join(category_dir, file)
            quality_metrics = metrics.assess_audio_quality(file_path)

            quality_data.append({
                "file": file,
                "category": category,
                **quality_metrics
            })

        total_files += len(files)

    if not quality_data:
        return {"error": "No files found in dataset"}

    # Convert to DataFrame for analysis
    df = pd.DataFrame(quality_data)

    # Calculate summary statistics
    summary = {
        "total_files": total_files,
        "average_quality_score": df["quality_score"].mean(),
        "quality_score_std": df["quality_score"].std(),
        "high_quality_files": len(df[df["quality_score"] > 70]),
        "low_quality_files": len(df[df["quality_score"] < 30]),
        "average_snr": df["snr"].mean(),
        "average_clipping": df["clipping_ratio"].mean(),
        "average_silence": df["silence_ratio"].mean(),
        "category_breakdown": {}
    }

    # Category-wise breakdown
    for category in categories:
        cat_data = df[df["category"] == category]
        if len(cat_data) > 0:
            summary["category_breakdown"][category] = {
                "count": len(cat_data),
                "avg_quality": cat_data["quality_score"].mean(),
                "high_quality": len(cat_data[cat_data["quality_score"] > 70])
            }

    return summary

def filter_low_quality_files(dataset_dir: str, quality_threshold: float = 30.0) -> List[str]:
    """Identify and return list of low-quality files for removal."""
    print(f"Filtering files with quality score below {quality_threshold}...")

    metrics = QualityMetrics()
    low_quality_files = []

    categories = ['male', 'female', 'uncertain']

    for category in categories:
        category_dir = os.path.join(dataset_dir, category)
        if not os.path.exists(category_dir):
            continue

        files = [f for f in os.listdir(category_dir) if f.endswith('.wav')]

        for file in files:
            file_path = os.path.join(category_dir, file)
            quality_metrics = metrics.assess_audio_quality(file_path)

            if quality_metrics["quality_score"] < quality_threshold:
                low_quality_files.append(file_path)

    return low_quality_files

if __name__ == "__main__":
    # Example usage
    dataset_dir = "data/voice_dataset"
    if os.path.exists(dataset_dir):
        summary = assess_dataset_quality(dataset_dir)
        print("Dataset Quality Summary:")
        print(f"Total files: {summary['total_files']}")
        print(".2f")
        print(f"High quality files (>70): {summary['high_quality_files']}")
        print(f"Low quality files (<30): {summary['low_quality_files']}")

        # Filter low quality files
        low_quality = filter_low_quality_files(dataset_dir)
        if low_quality:
            print(f"\nFound {len(low_quality)} low-quality files:")
            for file in low_quality[:5]:  # Show first 5
                print(f"  {file}")
            if len(low_quality) > 5:
                print(f"  ... and {len(low_quality) - 5} more")
    else:
        print(f"Dataset directory {dataset_dir} not found")