"""
Quality Assessment Module for VOXENT v3.0
Phase 2 AI Enhancement: Automatic quality scoring for audio segments

This module provides AI-powered quality assessment for audio segments,
including SNR, speech clarity, clipping detection, and more.

Features:
- Audio Quality Score (0-100)
  - Background noise level (SNR)
  - Clipping/distortion detection
  - Dynamic range analysis
  
- Speech Quality Score (0-100)
  - Clarity of pronunciation
  - Speaking rate appropriateness
  - Pitch variety
  
- Segment Usefulness Score (0-100)
  - Duration appropriateness
  - Overlapping speech detection
  - Voice stability
"""

import numpy as np
import librosa
import logging
from typing import Dict, Optional, List
from pathlib import Path

try:
    from scipy.signal import find_peaks
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import pystoi
    PYSTOI_AVAILABLE = True
except ImportError:
    PYSTOI_AVAILABLE = False

logger = logging.getLogger(__name__)


class AudioAnalyzer:
    """Analyze various audio quality metrics"""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
    
    def compute_snr(self, audio: np.ndarray, 
                   noise_duration: float = 0.5) -> float:
        """
        Estimate Signal-to-Noise Ratio
        
        Args:
            audio: Audio time series
            noise_duration: Duration to estimate noise (in seconds)
            
        Returns:
            SNR in dB
        """
        try:
            # Assume beginning of audio is mostly noise
            noise_samples = int(noise_duration * self.sr)
            noise = audio[:min(noise_samples, len(audio))]
            
            # Power of noise
            noise_power = np.mean(noise ** 2)
            
            # Power of signal (excluding assumed noise part)
            signal_power = np.mean(audio[noise_samples:] ** 2)
            
            if noise_power == 0 or signal_power == 0:
                return 0.0
            
            snr = 10 * np.log10(signal_power / noise_power)
            return float(np.clip(snr, -20, 80))  # Clip to reasonable range
        except Exception as e:
            logger.warning(f"SNR computation failed: {e}")
            return 0.0
    
    def detect_clipping(self, audio: np.ndarray, 
                       threshold: float = 0.98) -> float:
        """
        Detect clipping/distortion in audio
        
        Returns:
            Clipping percentage (0-100)
        """
        try:
            # Normalize audio to -1 to 1 range
            max_val = np.max(np.abs(audio))
            if max_val == 0:
                return 0.0
            
            normalized = audio / max_val
            
            # Find samples near peak amplitude (potential clipping)
            clipped = np.abs(normalized) > threshold
            clipping_percent = 100 * np.sum(clipped) / len(audio)
            
            return float(clipping_percent)
        except Exception as e:
            logger.warning(f"Clipping detection failed: {e}")
            return 0.0
    
    def compute_dynamic_range(self, audio: np.ndarray) -> float:
        """
        Compute dynamic range in dB
        
        Returns:
            Dynamic range (dB)
        """
        try:
            # Compute RMS on small windows
            frame_length = int(0.02 * self.sr)  # 20ms
            hop_length = frame_length // 2
            
            rms = librosa.feature.rms(y=audio, frame_length=frame_length,
                                      hop_length=hop_length)[0]
            
            if np.max(rms) == 0:
                return 0.0
            
            max_rms = np.max(rms)
            min_rms = np.min(rms[rms > 0]) if np.any(rms > 0) else max_rms
            
            if min_rms == 0:
                return 0.0
            
            dynamic_range = 20 * np.log10(max_rms / min_rms)
            return float(np.clip(dynamic_range, 0, 60))
        except Exception as e:
            logger.warning(f"Dynamic range computation failed: {e}")
            return 0.0
    
    def estimate_background_noise(self, audio: np.ndarray) -> float:
        """
        Estimate background noise level (0-100, lower is better)
        
        Returns:
            Noise score (0-100)
        """
        try:
            # Use spectral subtraction approach
            # Compute spectrogram
            D = librosa.stft(audio)
            magnitude = np.abs(D)
            
            # Estimate noise as minimum magnitude per frequency
            noise_profile = np.min(magnitude, axis=1)
            
            # Signal-to-noise ratio per frequency
            mean_magnitude = np.mean(magnitude, axis=1)
            snr_freq = 20 * np.log10(mean_magnitude / (noise_profile + 1e-10))
            
            # Average SNR across frequencies
            avg_snr = np.mean(snr_freq)
            
            # Convert to 0-100 scale (lower noise = higher score)
            noise_score = 100 - np.clip(avg_snr / 2, 0, 100)
            return float(noise_score)
        except Exception as e:
            logger.warning(f"Noise estimation failed: {e}")
            return 50.0  # Default to medium noise
    
    def detect_silence(self, audio: np.ndarray,
                      silence_threshold: float = 0.02) -> float:
        """
        Detect percentage of silence in audio
        
        Returns:
            Silence percentage (0-100)
        """
        try:
            # Compute RMS energy
            rms = librosa.feature.rms(y=audio)[0]
            
            # Threshold
            energy_threshold = silence_threshold * np.max(rms)
            silent_frames = rms < energy_threshold
            
            silence_percent = 100 * np.sum(silent_frames) / len(rms)
            return float(silence_percent)
        except Exception as e:
            logger.warning(f"Silence detection failed: {e}")
            return 0.0


class SpeechQualityAnalyzer:
    """Analyze speech-specific quality metrics"""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
    
    def extract_pitch_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract pitch-related features"""
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sr)
            
            # Get pitch values where magnitude is significant
            index = magnitudes > np.median(magnitudes)
            pitch_values = pitches[index]
            
            if len(pitch_values) == 0:
                return {
                    'pitch_mean': 0.0,
                    'pitch_std': 0.0,
                    'pitch_variety': 0.0,
                }
            
            return {
                'pitch_mean': float(np.mean(pitch_values)),
                'pitch_std': float(np.std(pitch_values)),
                'pitch_variety': float(np.std(pitch_values) / np.mean(pitch_values)),
            }
        except Exception as e:
            logger.warning(f"Pitch extraction failed: {e}")
            return {
                'pitch_mean': 0.0,
                'pitch_std': 0.0,
                'pitch_variety': 0.0,
            }
    
    def estimate_speaking_rate(self, audio: np.ndarray) -> float:
        """
        Estimate speaking rate in words per minute
        
        Returns:
            Estimated WPM
        """
        try:
            # Compute onset strength (syllable approximation)
            onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
            
            # Detect peaks (syllables)
            peaks, _ = signal.find_peaks(onset_env, height=np.median(onset_env))
            
            # Estimate syllables (approximation: ~2 syllables per word)
            num_syllables = len(peaks)
            num_words = num_syllables / 2
            
            # Duration in minutes
            duration_minutes = len(audio) / self.sr / 60
            
            if duration_minutes == 0:
                return 0.0
            
            wpm = num_words / duration_minutes
            return float(np.clip(wpm, 0, 300))  # Typical range 100-200 WPM
        except Exception as e:
            logger.warning(f"Speaking rate estimation failed: {e}")
            return 0.0
    
    def compute_formant_clarity(self, audio: np.ndarray) -> float:
        """
        Compute formant clarity score (0-100)
        
        Returns:
            Clarity score (higher = clearer)
        """
        try:
            # Spectral analysis
            S = np.abs(librosa.stft(audio))
            
            # Spectral centroid (formant indicator)
            cent = librosa.feature.spectral_centroid(S=S, sr=self.sr)[0]
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(S=S, sr=self.sr)[0]
            
            # MFCC coefficients
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
            
            # Clarity: variance in spectral features suggests formants
            clarity_score = (np.std(cent) + np.std(rolloff)) / 1000
            clarity_score = 100 * np.clip(clarity_score, 0, 1)
            
            return float(clarity_score)
        except Exception as e:
            logger.warning(f"Formant clarity computation failed: {e}")
            return 50.0


class QualityAssessor:
    """
    Main quality assessment module for audio segments
    """
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
        self.audio_analyzer = AudioAnalyzer(sr=sr)
        self.speech_analyzer = SpeechQualityAnalyzer(sr=sr)
    
    def assess_segment(self, audio: np.ndarray) -> Dict:
        """
        Assess quality of audio segment
        
        Args:
            audio: Audio time series
            
        Returns:
            Dictionary with quality assessment
        """
        duration = len(audio) / self.sr
        
        # Audio quality metrics
        snr = self.audio_analyzer.compute_snr(audio)
        clipping = self.audio_analyzer.detect_clipping(audio)
        dynamic_range = self.audio_analyzer.compute_dynamic_range(audio)
        background_noise = self.audio_analyzer.estimate_background_noise(audio)
        silence = self.audio_analyzer.detect_silence(audio)
        
        # Speech quality metrics
        pitch_features = self.speech_analyzer.extract_pitch_features(audio)
        speaking_rate = self.speech_analyzer.estimate_speaking_rate(audio)
        clarity = self.speech_analyzer.compute_formant_clarity(audio)
        
        # Composite scores
        audio_quality_score = self._compute_audio_quality_score(
            snr, clipping, dynamic_range, background_noise, silence
        )
        
        speech_quality_score = self._compute_speech_quality_score(
            clarity, speaking_rate, pitch_features['pitch_variety']
        )
        
        usefulness_score = self._compute_usefulness_score(
            duration, silence, dynamic_range
        )
        
        overall_score = (audio_quality_score + speech_quality_score + usefulness_score) / 3
        
        return {
            'duration': duration,
            'audio_quality': {
                'snr_db': snr,
                'clipping_percent': clipping,
                'dynamic_range_db': dynamic_range,
                'background_noise_score': background_noise,
                'silence_percent': silence,
                'score': audio_quality_score,
            },
            'speech_quality': {
                'clarity': clarity,
                'speaking_rate_wpm': speaking_rate,
                'pitch_mean': pitch_features['pitch_mean'],
                'pitch_variety': pitch_features['pitch_variety'],
                'score': speech_quality_score,
            },
            'usefulness': {
                'duration_score': self._score_duration(duration),
                'silence_penalty': silence,
                'dynamic_range_score': min(100, dynamic_range * 2),
                'score': usefulness_score,
            },
            'overall_quality_score': overall_score,
            'rating': self._rate_quality(overall_score),
        }
    
    def assess_file(self, audio_file: str) -> Dict:
        """
        Assess quality of audio file
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Quality assessment dictionary
        """
        try:
            audio, _ = librosa.load(audio_file, sr=self.sr, mono=True)
            result = self.assess_segment(audio)
            result['file'] = str(audio_file)
            return result
        except Exception as e:
            logger.error(f"Failed to assess {audio_file}: {e}")
            return {
                'file': str(audio_file),
                'error': str(e),
                'overall_quality_score': 0.0,
                'rating': 'error'
            }
    
    def assess_batch(self, audio_files: List[str]) -> List[Dict]:
        """
        Assess quality of multiple files
        
        Args:
            audio_files: List of audio file paths
            
        Returns:
            List of assessment results
        """
        results = []
        for audio_file in audio_files:
            result = self.assess_file(audio_file)
            results.append(result)
        
        return results
    
    def _compute_audio_quality_score(self, snr: float, clipping: float,
                                    dynamic_range: float, noise: float,
                                    silence: float) -> float:
        """Combine audio metrics into score (0-100)"""
        # SNR: -20 to 40 dB maps to 0-100
        snr_score = 100 * (snr + 20) / 60
        
        # Clipping: should be 0%, lower is better
        clipping_score = 100 - clipping
        
        # Dynamic range: should be 15-60 dB
        dr_score = 100 * np.clip((dynamic_range - 5) / 60, 0, 1)
        
        # Noise: should be low (0-100 scale, lower is better)
        noise_score = 100 - noise
        
        # Silence: should be minimal
        silence_score = 100 - silence
        
        # Weighted combination
        score = (
            snr_score * 0.25 +
            clipping_score * 0.15 +
            dr_score * 0.20 +
            noise_score * 0.20 +
            silence_score * 0.20
        )
        
        return float(np.clip(score, 0, 100))
    
    def _compute_speech_quality_score(self, clarity: float,
                                     speaking_rate: float,
                                     pitch_variety: float) -> float:
        """Combine speech metrics into score (0-100)"""
        # Clarity: direct score
        clarity_score = clarity
        
        # Speaking rate: 100-200 WPM is good
        rate_score = 100 - np.abs(speaking_rate - 150) / 1.5
        rate_score = np.clip(rate_score, 0, 100)
        
        # Pitch variety: 0.1-0.5 is good
        pitch_score = 100 - np.abs(pitch_variety - 0.3) / 0.3 * 50
        pitch_score = np.clip(pitch_score, 0, 100)
        
        # Weighted combination
        score = (
            clarity_score * 0.4 +
            rate_score * 0.3 +
            pitch_score * 0.3
        )
        
        return float(np.clip(score, 0, 100))
    
    def _compute_usefulness_score(self, duration: float,
                                 silence: float,
                                 dynamic_range: float) -> float:
        """Compute usefulness for dataset"""
        # Duration: 2-30 seconds is good
        duration_score = self._score_duration(duration)
        
        # Silence: should be minimal
        silence_score = 100 - silence
        
        # Dynamic range: should be reasonable
        dr_score = min(100, dynamic_range * 2)
        
        score = (
            duration_score * 0.4 +
            silence_score * 0.3 +
            dr_score * 0.3
        )
        
        return float(np.clip(score, 0, 100))
    
    def _score_duration(self, duration: float) -> float:
        """Score audio duration (2-30 seconds is ideal)"""
        if duration < 0.5:
            return 0.0
        elif duration < 2:
            return 50 + 25 * (duration - 0.5) / 1.5
        elif duration <= 30:
            return 100.0
        elif duration <= 60:
            return 100 - 50 * (duration - 30) / 30
        else:
            return 0.0
    
    def _rate_quality(self, score: float) -> str:
        """Convert score to quality rating"""
        if score >= 90:
            return "excellent"
        elif score >= 75:
            return "good"
        elif score >= 60:
            return "acceptable"
        elif score >= 40:
            return "poor"
        else:
            return "bad"


if __name__ == "__main__":
    """Example usage"""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Quality Assessment for VOXENT")
    parser.add_argument('--input', help='Input audio file')
    parser.add_argument('--batch', help='Directory with audio files')
    parser.add_argument('--output', help='Output JSON report')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    assessor = QualityAssessor()
    
    if args.input:
        result = assessor.assess_file(args.input)
        print(json.dumps(result, indent=2))
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
    
    if args.batch:
        from pathlib import Path
        audio_files = list(Path(args.batch).glob("*.wav")) + \
                     list(Path(args.batch).glob("*.mp3"))
        
        results = assessor.assess_batch([str(f) for f in audio_files])
        
        # Sort by quality score
        results = sorted(results, key=lambda x: x.get('overall_quality_score', 0), 
                        reverse=True)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
        
        print(f"Assessed {len(results)} files")
        print(f"Top 5 best quality:")
        for i, r in enumerate(results[:5]):
            print(f"  {i+1}. {r['file']}: {r['overall_quality_score']:.1f} ({r['rating']})")
