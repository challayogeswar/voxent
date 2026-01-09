"""
VOXENT Main Pipeline
Complete pipeline for speaker separation and voice dataset creation

Usage:
    python voxent_pipeline.py --config config.yaml

Steps:
1. Organize audio files into batches
2. Process each batch with speaker diarization
3. Extract and classify speaker segments
4. Organize by gender
5. Generate metadata and reports
"""

import os
import sys
import argparse
import yaml
import torch
import json
from pathlib import Path
from datetime import datetime

# Import modules
from .batch_organizer import BatchOrganizer
from ..diarization.enhanced_diarizer import EnhancedSpeakerDiarizer
from .batch_processor import IntegratedBatchProcessor


class VoxentPipeline:
    """
    Complete VOXENT pipeline for voice dataset creation
    """
    
    def __init__(self, config_path: str):
        """
        Initialize pipeline with configuration
        
        Args:
            config_path: Path to YAML configuration file
        """
        print(f"\n{'='*70}")
        print(f"VOXENT PIPELINE v2.0")
        print(f"{'='*70}")
        print(f"Configuration: {config_path}\n")
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Validate configuration
        self.validate_config()
        
        # Initialize components
        self.batch_organizer = None
        self.diarizer = None
        self.batch_processor = None
        
        print("✅ Pipeline initialized successfully!\n")
    
    def load_config(self, config_path: str) -> dict:
        """Load and parse YAML configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✓ Configuration loaded: {config_path}")
            return config
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            sys.exit(1)
    
    def validate_config(self):
        """Validate configuration and check for required settings"""
        print("\nValidating configuration...")
        
        # Check HuggingFace token
        hf_token = self.config.get('huggingface', {}).get('token')
        if not hf_token:
            print("⚠️  WARNING: HuggingFace token not set!")
            print("   Required for speaker diarization with pyannote.audio")
            print("   Get token from: https://huggingface.co/settings/tokens")
            print("   Accept model at: https://huggingface.co/pyannote/speaker-diarization-3.1")
            
            response = input("\nContinue without diarization? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
        else:
            print("✓ HuggingFace token configured")
        
        # Check paths
        input_dir = self.config['paths']['input_calls']
        if not os.path.exists(input_dir):
            print(f"❌ Input directory not found: {input_dir}")
            sys.exit(1)
        print(f"✓ Input directory exists: {input_dir}")
        
        # Check GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {gpu_name}")
        else:
            print("⚠️  No GPU detected, will use CPU (slower)")
        
        print("✓ Configuration validated\n")
    
    def setup_directories(self):
        """Create necessary output directories"""
        print("Setting up directories...")
        
        dirs_to_create = [
            self.config['paths']['batches'],
            self.config['paths']['voice_dataset'],
            self.config['paths']['temp_dir'],
            self.config['paths']['logs']
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"  ✓ {dir_path}")
        
        print()
    
    def step1_organize_batches(self):
        """Step 1: Organize audio files into batches"""
        print(f"\n{'#'*70}")
        print("STEP 1: ORGANIZE FILES INTO BATCHES")
        print(f"{'#'*70}\n")
        
        # Initialize batch organizer
        batch_config = {
            'files_per_batch': self.config['batch_organization']['files_per_batch'],
            'batch_size_minutes': self.config['batch_organization']['batch_size_minutes']
        }
        
        self.batch_organizer = BatchOrganizer(batch_config)
        
        # Organize files
        result = self.batch_organizer.organize_directory(
            input_dir=self.config['paths']['input_calls'],
            output_dir=self.config['paths']['batches'],
            copy_files=self.config['batch_organization']['copy_files']
        )
        
        print(f"\n✅ Step 1 complete: {result['num_batches']} batches created")
        return result
    
    def step2_initialize_diarizer(self):
        """Step 2: Initialize speaker diarization pipeline"""
        print(f"\n{'#'*70}")
        print("STEP 2: INITIALIZE SPEAKER DIARIZATION")
        print(f"{'#'*70}\n")
        
        # Diarization configuration
        diarization_config = {
            'min_speakers': self.config['diarization']['min_speakers'],
            'max_speakers': self.config['diarization']['max_speakers'],
            'hf_token': self.config['huggingface']['token']
        }
        
        # Initialize diarizer
        self.diarizer = EnhancedSpeakerDiarizer(diarization_config)
        
        try:
            self.diarizer.initialize_pipeline()
            print("\n✅ Step 2 complete: Diarizer initialized")
            return True
        except Exception as e:
            print(f"\n❌ Error initializing diarizer: {e}")
            return False
    
    def step3_process_batches(self):
        """Step 3: Process all batches with speaker diarization"""
        print(f"\n{'#'*70}")
        print("STEP 3: PROCESS BATCHES WITH SPEAKER DIARIZATION")
        print(f"{'#'*70}\n")
        
        # Batch processor configuration
        processor_config = {
            'batch_size_gpu': self.config['gpu']['batch_size_gpu'],
            'gpu_memory_threshold': self.config['gpu']['memory_threshold'],
            'clear_cache_between_batches': self.config['gpu']['clear_cache_between_batches']
        }
        
        # Initialize batch processor
        self.batch_processor = IntegratedBatchProcessor(self.diarizer, processor_config)
        
        # Process all batches
        result = self.batch_processor.process_all_batches(
            batches_dir=self.config['paths']['batches'],
            output_base_dir=self.config['paths']['voice_dataset']
        )
        
        print(f"\n✅ Step 3 complete: Processed {result['batches_processed']} batches")
        return result
    
    def step4_generate_report(self, results: dict):
        """Step 4: Generate final report"""
        print(f"\n{'#'*70}")
        print("STEP 4: GENERATE FINAL REPORT")
        print(f"{'#'*70}\n")
        
        report = {
            'pipeline_version': '2.0',
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'input_directory': self.config['paths']['input_calls'],
                'output_directory': self.config['paths']['voice_dataset'],
                'files_per_batch': self.config['batch_organization']['files_per_batch'],
                'gpu_batch_size': self.config['gpu']['batch_size_gpu']
            },
            'results': results,
            'statistics': {
                'total_batches': results.get('batches_processed', 0),
                'total_files': results.get('total_files', 0),
                'successful': results.get('successful', 0),
                'failed': results.get('failed', 0),
                'success_rate': results.get('success_rate', 0),
                'processing_time_minutes': results.get('total_processing_time_minutes', 0)
            }
        }
        
        # Save report
        report_path = os.path.join(
            self.config['paths']['logs'],
            f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Step 4 complete: Report saved to {report_path}")
        return report
    
    def run(self):
        """Run complete pipeline"""
        try:
            start_time = datetime.now()
            
            # Setup
            self.setup_directories()
            
            # Step 1: Organize batches
            batch_result = self.step1_organize_batches()
            
            # Step 2: Initialize diarizer
            if not self.step2_initialize_diarizer():
                print("\n❌ Pipeline stopped: Could not initialize diarizer")
                return
            
            # Step 3: Process batches
            processing_result = self.step3_process_batches()
            
            # Step 4: Generate report
            report = self.step4_generate_report(processing_result)
            
            # Final summary
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds() / 60
            
            print(f"\n{'='*70}")
            print("PIPELINE COMPLETE!")
            print(f"{'='*70}")
            print(f"Total time: {total_time:.2f} minutes")
            print(f"Success rate: {report['statistics']['success_rate']:.1f}%")
            print(f"Output directory: {self.config['paths']['voice_dataset']}")
            print(f"{'='*70}\n")
            
        except KeyboardInterrupt:
            print("\n\n❌ Pipeline interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n\n❌ Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="VOXENT Pipeline - Voice Dataset Creation with Speaker Diarization"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--skip-organize',
        action='store_true',
        help='Skip batch organization step (use existing batches)'
    )
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = VoxentPipeline(args.config)
    
    if args.skip_organize:
        print("Skipping batch organization step...")
        pipeline.step2_initialize_diarizer()
        pipeline.step3_process_batches()
    else:
        pipeline.run()


if __name__ == "__main__":
    main()
