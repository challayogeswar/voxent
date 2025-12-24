#!/usr/bin/env python3
"""
ML Classifier Training Script for VOXENT

This script trains a machine learning classifier for gender classification
using the processed voice dataset with quality filtering.
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from classification.ml_classifier import train_ml_classifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Train ML classifier for VOXENT")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/voice_dataset/metadata.csv",
        help="Path to metadata CSV file"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/voice_dataset",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/ml_gender_classifier.pkl",
        help="Path to save trained model"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=70.0,
        help="Minimum confidence score for training samples"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even if model exists"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        if os.path.exists(args.config):
            config = load_config(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            logger.warning(f"Configuration file {args.config} not found, using defaults")
            config = {}

        # Override with command line arguments
        metadata_file = args.metadata
        dataset_dir = args.dataset_dir
        model_path = args.model_path
        min_confidence = args.min_confidence

        # Get from config if not specified
        if 'classification' in config:
            class_config = config['classification']
            if not args.metadata:
                metadata_file = class_config.get('metadata_file', metadata_file)
            if not args.dataset_dir:
                dataset_dir = class_config.get('dataset_dir', dataset_dir)
            if not args.model_path:
                model_path = class_config.get('ml_model_path', model_path)
            if not args.min_confidence:
                min_confidence = class_config.get('min_confidence', min_confidence)

        # Check if model already exists
        if os.path.exists(model_path) and not args.force:
            logger.info(f"Model already exists at {model_path}. Use --force to retrain.")
            return

        # Validate inputs
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        logger.info("Starting ML classifier training...")
        logger.info(f"Metadata file: {metadata_file}")
        logger.info(f"Dataset directory: {dataset_dir}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Minimum confidence: {min_confidence}")

        # Train the classifier
        results = train_ml_classifier(
            metadata_file=metadata_file,
            dataset_dir=dataset_dir,
            model_path=model_path,
            min_confidence=min_confidence
        )

        # Print results
        print("\n" + "="*50)
        print("TRAINING RESULTS")
        print("="*50)
        print(f"Overall Accuracy: {results['accuracy']:.3f}")
        print(f"Male Precision: {results['precision_male']:.3f}")
        print(f"Male Recall: {results['recall_male']:.3f}")
        print(f"Male F1-Score: {results['f1_male']:.3f}")
        print(f"Female Precision: {results['precision_female']:.3f}")
        print(f"Female Recall: {results['recall_female']:.3f}")
        print(f"Female F1-Score: {results['f1_female']:.3f}")
        print("="*50)

        logger.info("ML classifier training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()