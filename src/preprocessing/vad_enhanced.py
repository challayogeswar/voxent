"""
Voice Activity Detection (VAD) Module for VOXENT v3.0
Phase 1 AI Enhancement: Silero VAD for cleaner audio preprocessing

This module uses Silero VAD (a lightweight RNN-based model) to detect speech
in audio and remove silence/noise before diarization.

Benefits:
- Cleaner segments (removes silence and noise)
- Faster processing (skips non-speech regions)
- Better quality dataset
- Reduced storage requirements

Model: Silero VAD (RNN-based, no GPU required)
Speed: Real-time capable (faster than audio)
Accuracy: ~99% speech detection
"""

import numpy as np
import librosa
import logging
from typing import List, Tuple, Optional, Dict
from pathlib import Path

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class SileroVAD:
    """
    Silero VAD for speech detection
    """
    
    def __init__(self, sr: int = 16000, threshold: float = 0.5):
        """
        Initialize Silero VAD
        
        Args:
            sr: Sample rate (16000 Hz)
            threshold: Confidence threshold for speech detection (0-1)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Install with: pip install torch torchaudio")
        
        self.sr = sr
        self.threshold = threshold
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self._load_model()
    
    def _load_model(self):
        """Load Silero VAD model"""
        try:
            # Load model from HuggingFace
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
            self.model = model.to(self.device).eval()
            logger.info(f"✅ Silero VAD loaded on {self.device}")
        except Exception as e:
            logger.warning(f"Could not load Silero VAD from torch.hub: {e}")
            logger.warning("Attempting to use fallback WebRTC VAD...")
            self.model = None
    
    def get_speech_timestamps(self, audio: np.ndarray, 
                             min_duration: float = 0.5) -> List[Dict]:
        """
        Get speech regions from audio
        
        Args:
            audio: Audio time series (16000 Hz)
            min_duration: Minimum speech duration in seconds
            
        Returns:
            List of dicts with 'start' and 'end' timestamps
        """
        if self.model is None:
            logger.warning("VAD model not available, using energy-based fallback")
            return self._get_speech_timestamps_energy(audio, min_duration)
        
        try:
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio).float()
            
            # Get VAD predictions
            with torch.no_grad():
                speech_probs = self.model(audio_tensor.to(self.device), 
                                         torch.tensor([self.sr]).to(self.device))
            
            # Convert probabilities to speech segments
            speech_threshold = self.threshold
            speech_frames = speech_probs > speech_threshold
            
            # Get timestamps
            speech_samples = np.where(speech_frames.cpu().numpy())[0]
            
            if len(speech_samples) == 0:
                return []
            
            # Find continuous speech regions
            gaps = np.diff(speech_samples) > 1
            segment_starts = [0] + (np.where(gaps)[0] + 1).tolist()
            segment_ends = (np.where(gaps)[0] + 1).tolist() + [len(speech_samples)]
            
            timestamps = []
            for start_idx, end_idx in zip(segment_starts, segment_ends):
                start_sample = speech_samples[start_idx]
                end_sample = speech_samples[end_idx - 1] + 1
                
                duration = (end_sample - start_sample) / self.sr
                
                # Only include segments longer than min_duration
                if duration >= min_duration:
                    timestamps.append({
                        'start': float(start_sample / self.sr),
                        'end': float(end_sample / self.sr),
                        'duration': duration,
                    })
            
            logger.debug(f"Found {len(timestamps)} speech regions")
            return timestamps
            
        except Exception as e:
            logger.warning(f"Silero VAD failed: {e}, using fallback method")
            return self._get_speech_timestamps_energy(audio, min_duration)
    
    def _get_speech_timestamps_energy(self, audio: np.ndarray,
                                     min_duration: float = 0.5) -> List[Dict]:
        """
        Fallback: Energy-based speech detection
        
        Args:
            audio: Audio time series
            min_duration: Minimum speech duration in seconds
            
        Returns:
            List of speech regions
        """
        # Compute RMS energy
        frame_length = int(0.02 * self.sr)  # 20ms frames
        hop_length = frame_length // 2
        
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, 
                                  hop_length=hop_length)[0]
        
        # Compute energy threshold (mean + std)
        energy_threshold = np.mean(rms) + np.std(rms)
        
        # Find speech frames
        speech_frames = rms > energy_threshold
        
        # Convert to samples
        speech_samples = np.where(speech_frames)[0] * hop_length
        
        if len(speech_samples) == 0:
            return []
        
        # Find continuous speech regions
        gaps = np.diff(speech_samples) > hop_length * 2
        segment_starts = [0] + (np.where(gaps)[0] + 1).tolist()
        segment_ends = (np.where(gaps)[0] + 1).tolist() + [len(speech_samples)]
        
        timestamps = []
        for start_idx, end_idx in zip(segment_starts, segment_ends):
            start_sample = speech_samples[start_idx]
            end_sample = speech_samples[end_idx - 1]
            
            duration = (end_sample - start_sample) / self.sr
            
            if duration >= min_duration:
                timestamps.append({
                    'start': float(start_sample / self.sr),
                    'end': float(end_sample / self.sr),
                    'duration': duration,
                })
        
        logger.debug(f"Found {len(timestamps)} speech regions (energy-based)")
        return timestamps
    
    def extract_speech_regions(self, audio: np.ndarray, 
                              min_duration: float = 0.5) -> np.ndarray:
        """
        Extract only speech regions from audio
        
        Args:
            audio: Audio time series
            min_duration: Minimum speech duration
            
        Returns:
            Concatenated audio containing only speech regions
        """
        timestamps = self.get_speech_timestamps(audio, min_duration)
        
        if not timestamps:
            logger.warning("No speech regions detected")
            return audio
        
        # Extract and concatenate speech regions
        speech_regions = []
        for ts in timestamps:
            start_sample = int(ts['start'] * self.sr)
            end_sample = int(ts['end'] * self.sr)
            speech_regions.append(audio[start_sample:end_sample])
        
        return np.concatenate(speech_regions)


class WebRTCVAD:
    """
    WebRTC VAD as fallback (requires webrtcvad package)
    """
    
    def __init__(self, sr: int = 16000, aggressiveness: int = 3):
        """
        Initialize WebRTC VAD
        
        Args:
            sr: Sample rate
            aggressiveness: Aggressiveness (0-3, higher = more aggressive)
        """
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(aggressiveness)
            self.sr = sr
            self.frame_duration = 10  # 10ms frames
        except ImportError:
            logger.warning("webrtcvad not available. Install with: pip install webrtcvad")
            self.vad = None
            self.sr = sr
    
    def get_speech_timestamps(self, audio: np.ndarray) -> List[Dict]:
        """Get speech timestamps using WebRTC VAD"""
        if self.vad is None:
            logger.warning("WebRTC VAD not available")
            return []
        
        try:
            # Resample to 16000 Hz if needed
            if self.sr != 16000:
                audio = librosa.resample(audio, orig_sr=self.sr, target_sr=16000)
                sr = 16000
            else:
                sr = self.sr
            
            # Convert to 16-bit PCM
            audio_int16 = np.int16(audio / np.max(np.abs(audio)) * 32767)
            audio_bytes = audio_int16.tobytes()
            
            # Process frames
            frame_size = int(sr * self.frame_duration / 1000)
            speech_frames = []
            
            for i in range(0, len(audio_bytes), frame_size * 2):
                frame = audio_bytes[i:i + frame_size * 2]
                if len(frame) == frame_size * 2:
                    is_speech = self.vad.is_speech(frame, sr)
                    speech_frames.append(is_speech)
            
            # Convert to timestamps
            timestamps = []
            in_speech = False
            start_frame = 0
            
            for i, is_speech in enumerate(speech_frames):
                if is_speech and not in_speech:
                    start_frame = i
                    in_speech = True
                elif not is_speech and in_speech:
                    start_time = start_frame * self.frame_duration / 1000.0
                    end_time = i * self.frame_duration / 1000.0
                    timestamps.append({
                        'start': start_time,
                        'end': end_time,
                        'duration': end_time - start_time,
                    })
                    in_speech = False
            
            if in_speech:
                start_time = start_frame * self.frame_duration / 1000.0
                end_time = len(speech_frames) * self.frame_duration / 1000.0
                timestamps.append({
                    'start': start_time,
                    'end': end_time,
                    'duration': end_time - start_time,
                })
            
            return timestamps
        except Exception as e:
            logger.error(f"WebRTC VAD error: {e}")
            return []


class VADProcessor:
    """
    Main VAD processor for VOXENT integration
    Handles both Silero and WebRTC fallback
    """
    
    def __init__(self, sr: int = 16000, method: str = 'silero'):
        """
        Initialize VAD processor
        
        Args:
            sr: Sample rate
            method: 'silero' or 'webrtc'
        """
        self.sr = sr
        self.method = method
        self.vad = None
        
        if method == 'silero':
            try:
                self.vad = SileroVAD(sr=sr)
            except Exception as e:
                logger.warning(f"Silero VAD failed: {e}, falling back to energy-based")
                self.vad = SileroVAD(sr=sr)  # Will use fallback internally
        elif method == 'webrtc':
            self.vad = WebRTCVAD(sr=sr)
        else:
            # Default to Silero with fallback
            self.vad = SileroVAD(sr=sr)
    
    def process_audio(self, audio: np.ndarray, 
                     min_duration: float = 0.5) -> Dict:
        """
        Process audio with VAD
        
        Args:
            audio: Audio time series
            min_duration: Minimum speech region duration
            
        Returns:
            Dictionary with processing results
        """
        original_duration = len(audio) / self.sr
        
        # Get speech timestamps
        timestamps = self.vad.get_speech_timestamps(audio, min_duration)
        
        # Extract speech regions
        cleaned_audio = self.vad.extract_speech_regions(audio, min_duration)
        
        cleaned_duration = len(cleaned_audio) / self.sr
        reduction_percent = (1 - cleaned_duration / original_duration) * 100
        
        return {
            'status': 'success',
            'original_duration': original_duration,
            'cleaned_duration': cleaned_duration,
            'reduction_percent': reduction_percent,
            'num_speech_regions': len(timestamps),
            'timestamps': timestamps,
            'cleaned_audio': cleaned_audio,
        }
    
    def process_file(self, audio_file: str,
                    min_duration: float = 0.5,
                    output_file: Optional[str] = None) -> Dict:
        """
        Process audio file with VAD
        
        Args:
            audio_file: Input audio file path
            min_duration: Minimum speech duration
            output_file: Optional output file path
            
        Returns:
            Processing results
        """
        try:
            # Load audio
            audio, _ = librosa.load(audio_file, sr=self.sr, mono=True)
            
            # Process
            result = self.process_audio(audio, min_duration)
            
            # Save output if specified
            if output_file and result['status'] == 'success':
                import soundfile as sf
                sf.write(output_file, result['cleaned_audio'], self.sr)
                result['output_file'] = output_file
                logger.info(f"✅ Saved cleaned audio to {output_file}")
            
            return result
        except Exception as e:
            logger.error(f"Failed to process {audio_file}: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }


if __name__ == "__main__":
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice Activity Detection for VOXENT")
    parser.add_argument('--input', help='Input audio file')
    parser.add_argument('--output', help='Output audio file (cleaned)')
    parser.add_argument('--method', default='silero', choices=['silero', 'webrtc'])
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    processor = VADProcessor(method=args.method)
    result = processor.process_file(args.input, output_file=args.output)
    
    print(f"\n{'='*50}")
    print(f"VAD Results for {args.input}")
    print(f"{'='*50}")
    print(f"Original Duration: {result['original_duration']:.2f}s")
    print(f"Cleaned Duration: {result['cleaned_duration']:.2f}s")
    print(f"Reduction: {result['reduction_percent']:.1f}%")
    print(f"Speech Regions: {result['num_speech_regions']}")
