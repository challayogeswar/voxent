"""
VOXENT AI Integration Module
Integrates AI enhancements into the main VOXENT pipeline

This module provides a unified interface for accessing all AI features:
- ML-based gender classification
- Voice Activity Detection (VAD)
- Quality Assessment
- And future enhancements

Usage:
    from ai_integration import AIEnhancementEngine
    
    engine = AIEnhancementEngine(config)
    
    # Preprocess with VAD
    result = engine.preprocess_audio(audio)
    
    # Classify gender with ML
    gender_result = engine.classify_gender(audio)
    
    # Assess quality
    quality_result = engine.assess_quality(audio)
"""

import logging
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
import librosa
import yaml

from classification.ml_gender_classifier_v3 import MLGenderClassifier
from preprocessing.vad_enhanced import VADProcessor
from quality.quality_assessor import QualityAssessor

logger = logging.getLogger(__name__)


class AIEnhancementEngine:
    """
    Unified interface for all AI enhancements in VOXENT
    """
    
    def __init__(self, config_path: Optional[str] = None, 
                 config_dict: Optional[Dict] = None):
        """
        Initialize AI engine
        
        Args:
            config_path: Path to config.yaml
            config_dict: Configuration dictionary (if config_path not provided)
        """
        self.config = None
        self.vad_processor = None
        self.gender_classifier = None
        self.quality_assessor = None
        
        # Load configuration
        if config_path:
            self.load_config(config_path)
        elif config_dict:
            self.config = config_dict
        else:
            self.config = self._get_default_config()
        
        # Initialize components
        self._initialize_components()
        
        logger.info("✅ AI Enhancement Engine initialized")
    
    def load_config(self, config_path: str):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"✓ Configuration loaded: {config_path}")
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'ai_enhancements': {
                'enabled': True,
                'phase1': {
                    'enabled': True,
                    'vad': {
                        'enabled': True,
                        'method': 'silero',
                        'threshold': 0.5,
                    }
                },
                'phase2': {
                    'enabled': False,
                    'quality_assessment': {'enabled': False},
                    'transcription': {'enabled': False},
                    'speaker_embeddings': {'enabled': False},
                }
            },
            'gender_classification': {
                'method': 'ml',
                'ml_classifier': {
                    'enabled': True,
                    'model_type': 'randomforest',
                    'model_path': 'models/gender_classifier',
                    'confidence_threshold': 0.7,
                }
            }
        }
    
    def _initialize_components(self):
        """Initialize AI components based on config"""
        ai_config = self.config.get('ai_enhancements', {})
        
        if not ai_config.get('enabled', True):
            logger.info("AI enhancements disabled in config")
            return
        
        # Initialize VAD if enabled
        phase1 = ai_config.get('phase1', {})
        if phase1.get('enabled', True):
            vad_config = phase1.get('vad', {})
            if vad_config.get('enabled', True):
                try:
                    self.vad_processor = VADProcessor(
                        sr=16000,
                        method=vad_config.get('method', 'silero')
                    )
                    logger.info("✓ VAD processor initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize VAD: {e}")
        
        # Initialize ML gender classifier if enabled
        gender_config = self.config.get('gender_classification', {})
        if gender_config.get('method') == 'ml':
            ml_config = gender_config.get('ml_classifier', {})
            if ml_config.get('enabled', True):
                try:
                    self.gender_classifier = MLGenderClassifier(
                        model_type=ml_config.get('model_type', 'randomforest')
                    )
                    
                    model_path = ml_config.get('model_path', 'models/gender_classifier')
                    if Path(model_path).exists():
                        self.gender_classifier.load_model(model_path)
                        logger.info("✓ ML Gender classifier loaded")
                    else:
                        logger.warning(f"Gender classifier model not found at {model_path}")
                except Exception as e:
                    logger.warning(f"Failed to initialize gender classifier: {e}")
        
        # Initialize quality assessor if enabled
        phase2 = ai_config.get('phase2', {})
        if phase2.get('enabled', False):
            quality_config = phase2.get('quality_assessment', {})
            if quality_config.get('enabled', False):
                try:
                    self.quality_assessor = QualityAssessor(sr=16000)
                    logger.info("✓ Quality assessor initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize quality assessor: {e}")
    
    # ========================================================================
    # VOICE ACTIVITY DETECTION
    # ========================================================================
    
    def preprocess_audio(self, audio: np.ndarray,
                        min_duration: float = 0.5) -> Dict:
        """
        Preprocess audio with VAD
        
        Args:
            audio: Audio time series (16000 Hz)
            min_duration: Minimum speech duration
            
        Returns:
            Dictionary with preprocessing results
        """
        if self.vad_processor is None:
            logger.debug("VAD not available, returning original audio")
            return {
                'status': 'skipped',
                'cleaned_audio': audio,
                'reason': 'VAD not initialized'
            }
        
        try:
            result = self.vad_processor.process_audio(audio, min_duration)
            logger.debug(f"VAD preprocessing: {result['reduction_percent']:.1f}% reduction")
            return result
        except Exception as e:
            logger.warning(f"VAD preprocessing failed: {e}")
            return {
                'status': 'error',
                'cleaned_audio': audio,
                'error': str(e)
            }
    
    # ========================================================================
    # GENDER CLASSIFICATION
    # ========================================================================
    
    def classify_gender(self, audio: np.ndarray) -> Dict:
        """
        Classify speaker gender using ML model
        
        Args:
            audio: Audio time series (16000 Hz)
            
        Returns:
            Gender classification result
        """
        if self.gender_classifier is None:
            logger.debug("ML Gender classifier not available")
            return {
                'gender': 'unknown',
                'confidence': 0.0,
                'method': 'ml_not_available',
                'status': 'error'
            }
        
        try:
            result = self.gender_classifier.predict(audio)
            result['method'] = 'ml'
            result['status'] = 'success'
            return result
        except Exception as e:
            logger.warning(f"Gender classification failed: {e}")
            return {
                'gender': 'unknown',
                'confidence': 0.0,
                'method': 'ml_error',
                'status': 'error',
                'error': str(e)
            }
    
    def classify_gender_file(self, audio_file: str) -> Dict:
        """
        Classify gender for audio file
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Classification result
        """
        try:
            audio, _ = librosa.load(audio_file, sr=16000, mono=True)
            result = self.classify_gender(audio)
            result['file'] = audio_file
            return result
        except Exception as e:
            logger.error(f"Failed to classify {audio_file}: {e}")
            return {
                'file': audio_file,
                'gender': 'unknown',
                'confidence': 0.0,
                'status': 'error',
                'error': str(e)
            }
    
    # ========================================================================
    # QUALITY ASSESSMENT
    # ========================================================================
    
    def assess_quality(self, audio: np.ndarray) -> Dict:
        """
        Assess audio quality
        
        Args:
            audio: Audio time series (16000 Hz)
            
        Returns:
            Quality assessment result
        """
        if self.quality_assessor is None:
            logger.debug("Quality assessor not available")
            return {
                'overall_quality_score': 0.0,
                'status': 'skipped',
                'reason': 'Quality assessor not initialized'
            }
        
        try:
            result = self.quality_assessor.assess_segment(audio)
            result['status'] = 'success'
            return result
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return {
                'overall_quality_score': 0.0,
                'status': 'error',
                'error': str(e)
            }
    
    def assess_quality_file(self, audio_file: str) -> Dict:
        """
        Assess quality of audio file
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Quality assessment result
        """
        try:
            result = self.quality_assessor.assess_file(audio_file)
            result['status'] = 'success'
            return result
        except Exception as e:
            logger.error(f"Failed to assess {audio_file}: {e}")
            return {
                'file': audio_file,
                'overall_quality_score': 0.0,
                'status': 'error',
                'error': str(e)
            }
    
    # ========================================================================
    # FULL SEGMENT PROCESSING
    # ========================================================================
    
    def process_segment(self, audio: np.ndarray,
                       include_quality: bool = False) -> Dict:
        """
        Process audio segment with all AI enhancements
        
        Args:
            audio: Audio time series
            include_quality: Whether to include quality assessment
            
        Returns:
            Dictionary with all analysis results
        """
        results = {
            'original_duration': len(audio) / 16000,
            'preprocessing': {},
            'gender_classification': {},
            'quality_assessment': {},
        }
        
        # 1. Preprocessing (VAD)
        preprocess_result = self.preprocess_audio(audio)
        results['preprocessing'] = preprocess_result
        
        # Use cleaned audio for further processing
        processed_audio = preprocess_result.get('cleaned_audio', audio)
        
        # 2. Gender classification
        if self.gender_classifier is not None:
            gender_result = self.classify_gender(processed_audio)
            results['gender_classification'] = gender_result
        
        # 3. Quality assessment (if enabled)
        if include_quality and self.quality_assessor is not None:
            quality_result = self.assess_quality(processed_audio)
            results['quality_assessment'] = quality_result
        
        return results
    
    # ========================================================================
    # BATCH PROCESSING
    # ========================================================================
    
    def process_batch(self, audio_files: List[str],
                     include_quality: bool = False,
                     save_results: Optional[str] = None) -> List[Dict]:
        """
        Process multiple audio files
        
        Args:
            audio_files: List of audio file paths
            include_quality: Whether to include quality assessment
            save_results: Optional path to save results as JSON
            
        Returns:
            List of processing results
        """
        results = []
        
        for audio_file in audio_files:
            try:
                audio, _ = librosa.load(audio_file, sr=16000, mono=True)
                
                result = self.process_segment(audio, include_quality)
                result['file'] = audio_file
                result['status'] = 'success'
                
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {audio_file}: {e}")
                results.append({
                    'file': audio_file,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Save results if specified
        if save_results:
            try:
                Path(save_results).parent.mkdir(parents=True, exist_ok=True)
                with open(save_results, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results saved to {save_results}")
            except Exception as e:
                logger.warning(f"Failed to save results: {e}")
        
        return results
    
    # ========================================================================
    # STATUS AND INFO
    # ========================================================================
    
    def get_status(self) -> Dict:
        """Get status of all AI components"""
        return {
            'vad_available': self.vad_processor is not None,
            'gender_classifier_available': self.gender_classifier is not None and \
                                          self.gender_classifier.trained,
            'quality_assessor_available': self.quality_assessor is not None,
            'ml_confidence_threshold': self.config.get('gender_classification', {})
                                            .get('ml_classifier', {})
                                            .get('confidence_threshold', 0.7),
        }
    
    def print_status(self):
        """Print status of AI components"""
        status = self.get_status()
        
        print("\n" + "="*70)
        print("VOXENT AI ENHANCEMENT ENGINE STATUS")
        print("="*70)
        
        print(f"VAD (Voice Activity Detection): " + 
              ("✅ READY" if status['vad_available'] else "❌ NOT READY"))
        
        print(f"ML Gender Classifier: " + 
              ("✅ READY" if status['gender_classifier_available'] else "❌ NOT READY"))
        
        print(f"Quality Assessor: " + 
              ("✅ READY" if status['quality_assessor_available'] else "❌ NOT READY"))
        
        print(f"ML Confidence Threshold: {status['ml_confidence_threshold']:.2f}")
        print("\n")


if __name__ == "__main__":
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VOXENT AI Integration")
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--status', action='store_true', help='Show AI status')
    parser.add_argument('--process', type=str, help='Process audio file')
    parser.add_argument('--with-quality', action='store_true', help='Include quality assessment')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize engine
    engine = AIEnhancementEngine(config_path=args.config)
    
    # Show status
    if args.status:
        engine.print_status()
    
    # Process file
    if args.process:
        result = engine.process_segment(*librosa.load(args.process, sr=16000),
                                       include_quality=args.with_quality)
        print(json.dumps(result, indent=2))
