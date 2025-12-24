import unittest
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Import all modules for integration testing
from preprocessing.audio_loader import load_audio
from preprocessing.normalize import normalize
from preprocessing.vad import remove_silence
from classification import get_classifier
from dataset.organizer import save_sample
from dataset.metadata import append_metadata
from data_augmentation.augment import AudioAugmenter, balance_dataset
from quality_assurance.metrics import QualityMetrics, assess_dataset_quality
from engine.batch_runner import validate_config, process_file

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete VOXENT pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.sample_rate = 16000
        self.config = {
            'sample_rate': self.sample_rate,
            'min_segment_duration': 1.0,
            'male_pitch_threshold': 165,
            'female_pitch_threshold': 255,
            'confidence_margin': 20,
            'enable_augmentation': False
        }

        # Create test audio signal (2 seconds of 440Hz sine wave)
        duration = 2.0
        frequency = 440.0
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        self.test_audio = np.sin(frequency * 2 * np.pi * t).astype(np.float32)

        # Create test directories
        self.input_dir = os.path.join(self.test_dir, 'input_calls')
        self.output_dir = os.path.join(self.test_dir, 'voice_dataset')
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'male'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'female'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'uncertain'), exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)

    def create_test_wav(self, filename, audio=None):
        """Create a test WAV file."""
        if audio is None:
            audio = self.test_audio
        import soundfile as sf
        filepath = os.path.join(self.input_dir, filename)
        sf.write(filepath, audio, self.sample_rate)
        return filepath

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        self.assertTrue(validate_config(self.config))

        # Invalid config - missing key
        invalid_config = self.config.copy()
        del invalid_config['sample_rate']
        with self.assertRaises(ValueError):
            validate_config(invalid_config)

    def test_audio_processing_pipeline(self):
        """Test the complete audio processing pipeline."""
        # Create test file
        test_file = self.create_test_wav('test.wav')

        # Test audio loading
        audio = load_audio(test_file, self.sample_rate)
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(len(audio), len(self.test_audio))

        # Test normalization
        normalized = normalize(audio)
        self.assertIsInstance(normalized, np.ndarray)
        self.assertTrue(np.all(normalized >= -1.0))
        self.assertTrue(np.all(normalized <= 1.0))

        # Test integrated classifier
        classifier = get_classifier(self.config)
        label, conf = classifier.classify(normalized, self.sample_rate)
        self.assertIn(label, ['male', 'female', 'unknown'])
        self.assertIsInstance(conf, (int, float))
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 100.0)

    def test_file_saving_and_metadata(self):
        """Test file saving and metadata management."""
        # Save a test sample
        test_filename = 'test_sample_conf80.wav'
        save_sample(self.test_audio, self.sample_rate, 'male', test_filename, self.output_dir)

        # Check file was created
        expected_path = os.path.join(self.output_dir, 'male', test_filename)
        self.assertTrue(os.path.exists(expected_path))

        # Test metadata append (this will create the file if it doesn't exist)
        metadata_file = os.path.join(self.output_dir, 'metadata.csv')

        test_metadata = {
            'file': test_filename,
            'source': 'test.wav',
            'speaker': 'SPEAKER_00',
            'pitch': 180.0,
            'label': 'male',
            'confidence': 80.0,
            'duration': 2.0,
            'quality_score': 85.0
        }
        append_metadata(metadata_file, test_metadata)

        # Verify metadata was written
        self.assertTrue(os.path.exists(metadata_file))
        df = pd.read_csv(metadata_file)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['file'], test_filename)

    def test_data_augmentation(self):
        """Test data augmentation functionality."""
        augmenter = AudioAugmenter(self.sample_rate)

        # Test individual augmentations
        augmented_noise = augmenter.add_noise(self.test_audio)
        self.assertEqual(len(augmented_noise), len(self.test_audio))

        augmented_pitch = augmenter.change_pitch(self.test_audio)
        self.assertIsInstance(augmented_pitch, np.ndarray)

        augmented_speed = augmenter.change_speed(self.test_audio)
        self.assertIsInstance(augmented_speed, np.ndarray)

        # Test combined augmentation
        augmented = augmenter.apply_augmentation(self.test_audio)
        self.assertIsInstance(augmented, np.ndarray)
        self.assertTrue(np.all(np.abs(augmented) <= 1.0))  # Should be normalized

    def test_quality_metrics(self):
        """Test quality assessment metrics."""
        # Create test file
        test_file = self.create_test_wav('quality_test.wav')

        metrics = QualityMetrics(self.sample_rate)
        quality_result = metrics.assess_audio_quality(test_file)

        # Check all expected metrics are present
        expected_keys = ['snr', 'clipping_ratio', 'silence_ratio', 'zero_crossing_rate',
                        'quality_score', 'low_balance', 'mid_balance', 'high_balance']
        for key in expected_keys:
            self.assertIn(key, quality_result)
            self.assertIsInstance(quality_result[key], (int, float))

        # Quality score should be between 0 and 100
        self.assertGreaterEqual(quality_result['quality_score'], 0)
        self.assertLessEqual(quality_result['quality_score'], 100)

    @patch('dIarization.diarizer.pipeline')
    def test_process_file_integration(self, mock_pipeline):
        """Test the process_file function with mocked diarization."""
        # Mock diarization results
        mock_pipeline.from_pretrained.return_value.return_value = [
            {'start': 0.0, 'end': 2.0, 'speaker': 'SPEAKER_00'}
        ]

        # Create test file
        test_file = self.create_test_wav('integration_test.wav')

        # Process file
        result = process_file(test_file, self.config)

        # Check result structure
        self.assertIn('file', result)
        self.assertIn('segments_processed', result)
        self.assertIn('processing_time', result)
        self.assertEqual(result['file'], 'integration_test.wav')
        self.assertEqual(result['segments_processed'], 1)

    def test_dataset_quality_assessment(self):
        """Test dataset-wide quality assessment."""
        # Create some test files
        for i in range(3):
            self.create_test_wav(f'test_{i}.wav')

        # Create mock dataset structure
        dataset_dir = self.output_dir
        for category in ['male', 'female']:
            cat_dir = os.path.join(dataset_dir, category)
            os.makedirs(cat_dir, exist_ok=True)
            # Copy test files
            import soundfile as sf
            for i in range(2):
                filename = f'{category}_sample_{i}.wav'
                filepath = os.path.join(cat_dir, filename)
                sf.write(filepath, self.test_audio, self.sample_rate)

        # Assess dataset quality
        summary = assess_dataset_quality(dataset_dir)

        # Check summary structure
        self.assertIn('total_files', summary)
        self.assertIn('average_quality_score', summary)
        self.assertIn('category_breakdown', summary)
        self.assertEqual(summary['total_files'], 4)  # 2 male + 2 female

    @patch('dIarization.diarizer.diarize')
    def test_end_to_end_workflow(self, mock_diarize):
        """Test end-to-end workflow simulation with mocked diarization."""
        # Mock diarization to return test segments
        mock_diarize.return_value = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0}
        ]

        # Create test file
        test_file = self.create_test_wav('e2e_test.wav')

        # Test the process_file function with mocked diarization
        config = self.config.copy()
        config['classification'] = {
            'use_ml': False,  # Use pitch-based for testing
            'pitch_male_threshold': 85.0,
            'pitch_female_threshold': 165.0
        }

        result = process_file(test_file, config)

        # Verify processing completed
        self.assertIn('file', result)
        self.assertEqual(result['file'], 'e2e_test.wav')
        self.assertIn('segments_processed', result)
        self.assertEqual(result['segments_processed'], 1)

        # Check that output files were created
        metadata_file = os.path.join(self.output_dir, 'metadata.csv')
        self.assertTrue(os.path.exists(metadata_file))

        # Check metadata content
        df = pd.read_csv(metadata_file)
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertIn(row['label'], ['male', 'female', 'unknown'])
        self.assertGreaterEqual(row['confidence'], 0)
        self.assertLessEqual(row['confidence'], 100)

if __name__ == '__main__':
    unittest.main()