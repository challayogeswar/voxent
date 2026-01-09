"""
VOXENT AI Training Script
Train ML models for gender classification using existing audio data

This script bootstraps training from existing pitch-based classifications,
then continuously improves as new labeled data becomes available.

Usage:
    python train_ai_models.py --data-dir data/voice_dataset --config config.yaml
    python train_ai_models.py --bootstrap  # Quick training from existing labels
    python train_ai_models.py --fine-tune  # Continue training from existing model
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from classification.ml_gender_classifier_v3 import MLGenderClassifier
from preprocessing.vad_enhanced import VADProcessor
from quality.quality_assessor import QualityAssessor

import numpy as np
import librosa
import yaml

logger = logging.getLogger(__name__)


class AIModelTrainer:
    """Train and manage AI models for VOXENT"""
    
    def __init__(self, config_path: str = None, model_dir: str = "models"):
        """
        Initialize trainer
        
        Args:
            config_path: Path to config.yaml
            model_dir: Directory to store models
        """
        self.config = None
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        if config_path:
            self.load_config(config_path)
        
        logger.info(f"✅ Trainer initialized. Model directory: {self.model_dir}")
    
    def load_config(self, config_path: str):
        """Load YAML configuration"""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"✓ Config loaded: {config_path}")
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
    
    def collect_audio_files(self, directory: str, 
                          recursive: bool = True) -> List[Tuple[str, str]]:
        """
        Collect audio files with gender labels from directory structure
        
        Directory structure expected:
        directory/
          ├── male/
          │   ├── speaker_00_male_0001.wav
          │   └── ...
          ├── female/
          │   ├── speaker_01_female_0001.wav
          │   └── ...
          └── ambiguous/
              └── ...
        
        Args:
            directory: Root directory containing gender folders
            recursive: Whether to search recursively
            
        Returns:
            List of (file_path, gender_label) tuples
        """
        data = []
        base_path = Path(directory)
        
        # Known gender folders
        gender_folders = ['male', 'female', 'ambiguous']
        
        for gender in gender_folders:
            gender_path = base_path / gender
            
            if not gender_path.exists():
                logger.warning(f"Gender folder not found: {gender_path}")
                continue
            
            # Find all audio files
            if recursive:
                audio_files = list(gender_path.glob("**/*.wav")) + \
                             list(gender_path.glob("**/*.mp3")) + \
                             list(gender_path.glob("**/*.flac"))
            else:
                audio_files = list(gender_path.glob("*.wav")) + \
                             list(gender_path.glob("*.mp3")) + \
                             list(gender_path.glob("*.flac"))
            
            for audio_file in audio_files:
                data.append((str(audio_file), gender))
                logger.debug(f"Found: {audio_file} ({gender})")
        
        logger.info(f"Collected {len(data)} audio files")
        
        # Print distribution
        from collections import Counter
        distribution = Counter([label for _, label in data])
        for label, count in distribution.items():
            logger.info(f"  {label}: {count} files")
        
        return data
    
    def train_gender_classifier(self, audio_files: List[Tuple[str, str]],
                               model_type: str = "randomforest",
                               test_split: float = 0.2) -> Dict:
        """
        Train ML-based gender classifier
        
        Args:
            audio_files: List of (file_path, gender_label) tuples
            model_type: 'randomforest' or 'xgboost'
            test_split: Proportion of data for testing
            
        Returns:
            Training results dictionary
        """
        logger.info("\n" + "="*70)
        logger.info("TRAINING ML GENDER CLASSIFIER")
        logger.info("="*70)
        
        # Separate paths and labels
        paths, labels = zip(*audio_files)
        
        # Create classifier
        classifier = MLGenderClassifier(model_type=model_type, sr=16000)
        
        # Train
        result = classifier.train(list(paths), list(labels), test_split=test_split)
        
        if result['status'] == 'success':
            # Save model
            classifier_dir = self.model_dir / "gender_classifier"
            classifier.save_model(str(classifier_dir))
            result['model_path'] = str(classifier_dir)
        
        return result
    
    def bootstrap_training(self, voice_dataset_dir: str) -> Dict:
        """
        Bootstrap training from existing voice dataset
        
        Assumes files are already organized by gender in voice_dataset_dir
        
        Args:
            voice_dataset_dir: Path to voice_dataset directory
            
        Returns:
            Training results
        """
        logger.info("\n" + "="*70)
        logger.info("BOOTSTRAP TRAINING FROM EXISTING DATASET")
        logger.info("="*70)
        
        # Collect files
        audio_files = self.collect_audio_files(voice_dataset_dir)
        
        if len(audio_files) < 50:
            logger.warning(f"Only {len(audio_files)} files found. Need at least 50.")
            return {'status': 'error', 'message': 'Insufficient data'}
        
        # Train classifier
        result = self.train_gender_classifier(audio_files)
        
        return result
    
    def assess_quality(self, directory: str) -> List[Dict]:
        """
        Assess quality of audio files
        
        Args:
            directory: Directory with audio files
            
        Returns:
            List of quality assessment results
        """
        logger.info("\n" + "="*70)
        logger.info("QUALITY ASSESSMENT")
        logger.info("="*70)
        
        base_path = Path(directory)
        audio_files = list(base_path.glob("**/*.wav")) + \
                     list(base_path.glob("**/*.mp3"))
        
        assessor = QualityAssessor(sr=16000)
        results = assessor.assess_batch([str(f) for f in audio_files])
        
        # Sort by quality score
        results = sorted(results, key=lambda x: x.get('overall_quality_score', 0),
                        reverse=True)
        
        logger.info(f"Assessed {len(results)} files")
        
        # Print top and bottom files
        logger.info("\nTop 5 best quality:")
        for i, r in enumerate(results[:5]):
            score = r.get('overall_quality_score', 0)
            rating = r.get('rating', 'unknown')
            logger.info(f"  {i+1}. {Path(r['file']).name}: {score:.1f} ({rating})")
        
        logger.info("\nBottom 5 lowest quality:")
        for i, r in enumerate(results[-5:]):
            score = r.get('overall_quality_score', 0)
            rating = r.get('rating', 'unknown')
            logger.info(f"  {i+1}. {Path(r['file']).name}: {score:.1f} ({rating})")
        
        return results
    
    def preprocess_with_vad(self, input_dir: str, output_dir: str) -> Dict:
        """
        Preprocess audio files with VAD
        
        Args:
            input_dir: Directory with input audio files
            output_dir: Directory to save cleaned audio
            
        Returns:
            Processing results
        """
        logger.info("\n" + "="*70)
        logger.info("VOICE ACTIVITY DETECTION PREPROCESSING")
        logger.info("="*70)
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        audio_files = list(input_path.glob("**/*.wav")) + \
                     list(input_path.glob("**/*.mp3"))
        
        processor = VADProcessor(sr=16000, method='silero')
        
        total_reduction = 0
        results = []
        
        for audio_file in audio_files:
            output_file = output_path / audio_file.relative_to(input_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            result = processor.process_file(str(audio_file),
                                          output_file=str(output_file))
            results.append(result)
            
            if result['status'] == 'success':
                total_reduction += result['reduction_percent']
                logger.debug(f"  {audio_file.name}: "
                           f"-{result['reduction_percent']:.1f}% size")
        
        avg_reduction = total_reduction / len(results) if results else 0
        logger.info(f"Processed {len(results)} files")
        logger.info(f"Average size reduction: {avg_reduction:.1f}%")
        
        return {
            'status': 'success',
            'files_processed': len(results),
            'average_reduction_percent': avg_reduction,
            'output_directory': str(output_path),
        }
    
    def generate_training_report(self, results: Dict) -> str:
        """Generate training report"""
        report = """
╔════════════════════════════════════════════════════════════════════════════╗
║                    VOXENT AI TRAINING REPORT                              ║
╚════════════════════════════════════════════════════════════════════════════╝

"""
        
        if 'accuracy' in results:
            report += f"Gender Classification Model:\n"
            report += f"  Status: {results.get('status', 'unknown')}\n"
            report += f"  Accuracy: {results.get('accuracy', 0):.2%}\n"
            report += f"  Samples: {results.get('num_samples', 0)}\n"
            report += f"  Features: {results.get('num_features', 0)}\n"
            report += f"  Classes: {results.get('classes', [])}\n"
            report += f"  Model Type: {results.get('model_type', 'unknown')}\n"
            report += f"  Model Path: {results.get('model_path', 'unknown')}\n"
        
        report += f"\nTimestamp: {Path(__file__).parent.name}\n"
        
        return report


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="VOXENT AI Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_ai_models.py --bootstrap data/voice_dataset
  python train_ai_models.py --assess data/voice_dataset
  python train_ai_models.py --vad data/input_calls data/cleaned
        """
    )
    
    parser.add_argument('--config', type=str, help='Path to config.yaml')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory to store models')
    
    # Training options
    parser.add_argument('--bootstrap', type=str, metavar='DIR',
                       help='Bootstrap train from existing voice dataset')
    parser.add_argument('--model-type', choices=['randomforest', 'xgboost'],
                       default='randomforest', help='Classifier type')
    
    # Assessment
    parser.add_argument('--assess', type=str, metavar='DIR',
                       help='Assess quality of audio files')
    parser.add_argument('--assess-output', type=str, metavar='FILE',
                       help='Save assessment results to JSON')
    
    # Preprocessing
    parser.add_argument('--vad', nargs=2, metavar=('INPUT', 'OUTPUT'),
                       help='Apply VAD preprocessing')
    
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create trainer
    trainer = AIModelTrainer(config_path=args.config, model_dir=args.model_dir)
    
    # Bootstrap training
    if args.bootstrap:
        result = trainer.bootstrap_training(args.bootstrap)
        
        if result['status'] == 'success':
            print(trainer.generate_training_report(result))
            logger.info("✅ Training completed successfully!")
        else:
            logger.error(f"Training failed: {result}")
            sys.exit(1)
    
    # Quality assessment
    if args.assess:
        results = trainer.assess_quality(args.assess)
        
        if args.assess_output:
            with open(args.assess_output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.assess_output}")
    
    # VAD preprocessing
    if args.vad:
        input_dir, output_dir = args.vad
        result = trainer.preprocess_with_vad(input_dir, output_dir)
        logger.info(f"✅ Preprocessing completed: {result}")


if __name__ == "__main__":
    main()
