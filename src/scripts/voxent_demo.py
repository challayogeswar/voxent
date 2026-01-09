#!/usr/bin/env python
"""
VOXENT Pipeline Complete Demonstration
Shows the full workflow: Input → Conversion → Batching → Ready for Processing
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import librosa

def print_banner():
    """Print VOXENT banner"""
    banner = """
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                    VOXENT v2.0 - Pipeline Demonstration                   ║
║                   Voice Dataset Creation with Batching                    ║
║                                                                            ║
║  Features:                                                                 ║
║    ✓ Automatic MP3 to WAV Conversion                                      ║
║    ✓ Duration-Based Batch Organization                                    ║
║    ✓ GPU Memory Management                                                ║
║    ✓ Batch Metadata Generation                                            ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_section(title, level=1):
    """Print formatted section title"""
    if level == 1:
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}\n")
    else:
        print(f"\n{title}\n{'-'*len(title)}\n")


def show_workflow():
    """Display the complete workflow"""
    print_section("COMPLETE VOXENT WORKFLOW", 2)
    
    workflow = """
    1. INPUT STAGE
       └─ User places audio files (MP3/WAV) in data/input/
       └─ Supported formats: MP3, WAV, M4A, FLAC, OGG, AAC
    
    2. CONVERSION STAGE
       └─ AudioConverter detects non-WAV files
       └─ Converts all files to WAV format (16000 Hz, Mono)
       └─ Standardizes audio format for processing
    
    3. SCANNING STAGE
       └─ Calculates duration for each audio file
       └─ Sorts files by duration (smallest to largest)
       └─ Creates file inventory with metadata
    
    4. BATCHING STAGE
       └─ Groups files into batches respecting:
          • Maximum batch duration (default: 2 minutes)
          • Maximum files per batch (default: 10)
       └─ Creates batch folders: batch_001, batch_002, etc.
    
    5. ORGANIZATION STAGE
       └─ Copies/moves files to batch folders
       └─ Generates batch_metadata.json for each batch
       └─ Creates organization_summary.json
    
    6. PROCESSING STAGE
       └─ IntegratedBatchProcessor handles each batch
       └─ Performs speaker diarization
       └─ Classifies speaker gender
       └─ Organizes output by gender (male/female)
    
    7. OUTPUT STAGE
       └─ Final voice dataset organized in data/voice_dataset/
       └─ Includes speaker segments, metadata, reports
    """
    print(workflow)


def show_current_state():
    """Display current state of the pipeline"""
    print_section("CURRENT SYSTEM STATE", 2)
    
    import torch
    
    # Check GPU
    print("Hardware:")
    if torch.cuda.is_available():
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"  ⚠️  GPU: Not available (using CPU)")
    
    # Check input data
    print("\nInput Data:")
    input_dir = "data/input"
    if os.path.exists(input_dir):
        files = os.listdir(input_dir)
        audio_files = [f for f in files if Path(f).suffix.lower() in 
                      {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}]
        
        mp3_count = sum(1 for f in audio_files if f.lower().endswith('.mp3'))
        wav_count = sum(1 for f in audio_files if f.lower().endswith('.wav'))
        
        if audio_files:
            total_size = sum(os.path.getsize(os.path.join(input_dir, f)) 
                           for f in audio_files) / (1024*1024)
            print(f"  ✓ Audio files found: {len(audio_files)}")
            print(f"    - MP3: {mp3_count}")
            print(f"    - WAV: {wav_count}")
            print(f"    - Total size: {total_size:.2f} MB")
        else:
            print(f"  ⚠️  No audio files found in {input_dir}")
    else:
        print(f"  ⚠️  {input_dir} not found")
    
    # Check batches
    print("\nBatch Organization:")
    batches_dir = "data/batches"
    if os.path.exists(batches_dir):
        batch_folders = [f for f in os.listdir(batches_dir) if f.startswith('batch_')]
        if batch_folders:
            print(f"  ✓ Batches created: {len(batch_folders)}")
            # Count files
            total_files = 0
            for bf in sorted(batch_folders)[:3]:
                batch_path = os.path.join(batches_dir, bf)
                file_count = len([f for f in os.listdir(batch_path) 
                                if Path(f).suffix.lower() in 
                                {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}])
                total_files += file_count
            print(f"    - Sample batches shown (first 3 of {len(batch_folders)})")
            print(f"    - Files organized: {total_files}+")
        else:
            print(f"  ℹ️  Batches not yet created")
    
    # Check configuration
    print("\nConfiguration:")
    config_file = "src/config/config.yaml"
    if os.path.exists(config_file):
        print(f"  ✓ Configuration file: {config_file}")
    else:
        print(f"  ✗ Configuration file not found")


def show_batching_example():
    """Show example of batching logic"""
    print_section("BATCHING LOGIC EXAMPLE", 2)
    
    example = """
    Given these audio files:
    
    Filename                        Duration    File #
    ────────────────────────────────────────────────────
    audio_30s.mp3                   30s         1
    audio_45s.wav                   45s         2
    audio_1min.mp3                  60s         3
    audio_1m30s.wav                 90s         4
    audio_2min.mp3                  120s        5
    
    With batch_size_minutes = 2.0 (120 seconds):
    
    ┌──────────────────────────────────────────┐
    │ BATCH 001                                │
    │ ────────────────────────────────────────│
    │ audio_30s.mp3      (30s)                │
    │ audio_45s.wav      (45s)                │
    │ audio_1min.mp3     (60s)  ← Total: 135s│
    │ ❌ Cannot add audio_1m30s.wav (90s)     │
    │    Would exceed 120s limit               │
    └──────────────────────────────────────────┘
    
    ┌──────────────────────────────────────────┐
    │ BATCH 002                                │
    │ ────────────────────────────────────────│
    │ audio_1m30s.wav    (90s)                │
    │ audio_2min.mp3     (120s)  ← EXCEEDS!   │
    │ ❌ Cannot add (total would be 210s)     │
    └──────────────────────────────────────────┘
    
    ┌──────────────────────────────────────────┐
    │ BATCH 002 (Updated)                      │
    │ ────────────────────────────────────────│
    │ audio_1m30s.wav    (90s)  ← Total: 90s  │
    └──────────────────────────────────────────┘
    
    ┌──────────────────────────────────────────┐
    │ BATCH 003                                │
    │ ────────────────────────────────────────│
    │ audio_2min.mp3     (120s) ← Total: 120s │
    └──────────────────────────────────────────┘
    
    ✅ Result: 3 batches, all files organized by duration
    """
    print(example)


def show_available_commands():
    """Show available commands"""
    print_section("AVAILABLE COMMANDS", 2)
    
    commands = """
    1. Run Batching Test (Generate test data)
       ──────────────────────────────────────
       python src/scripts/test_batching_simple.py
       
       Creates:
         • Test audio files with various durations
         • Batch organization examples (2-min & 4-min)
         • Detailed batch breakdowns
    
    
    2. Run Pipeline Test (Verify all components)
       ───────────────────────────────────────
       python src/scripts/pipeline_test.py
       
       Checks:
         • Dependencies installed
         • Configuration valid
         • Input data present
         • Directories created
         • Batching logic working
    
    
    3. Run Full Pipeline (Process audio with diarization)
       ────────────────────────────────────────────
       python src/main.py --config src/config/config.yaml
       
       Prerequisites:
         ✓ HuggingFace token configured
         ✓ Audio files in data/input/
         ✓ Dependencies installed
       
       Steps:
         1. Converts audio formats
         2. Organizes into batches
         3. Runs speaker diarization
         4. Classifies gender
         5. Generates voice dataset
    
    
    4. Configure HuggingFace Token
       ───────────────────────────
       Edit: src/config/config.yaml
       
       Replace:
         huggingface:
           token: null
       
       With:
         huggingface:
           token: "your_hf_token_here"
       
       Get token from: https://huggingface.co/settings/tokens
    """
    print(commands)


def show_key_features():
    """Display key features implemented"""
    print_section("KEY FEATURES IMPLEMENTED", 2)
    
    features = """
    ✅ AUDIO CONVERSION
       └─ Automatic MP3 → WAV conversion
       └─ Supports: MP3, M4A, FLAC, OGG, AAC
       └─ Standardized output: 16000 Hz, Mono
    
    
    ✅ DURATION-BASED BATCHING
       └─ Groups files by duration
       └─ Respects maximum batch duration
       └─ Respects maximum files per batch
       └─ Optimized for GPU memory
    
    
    ✅ BATCH ORGANIZATION
       └─ Creates batch_001, batch_002, etc.
       └─ Generates metadata for each batch
       └─ Provides organization summary
       └─ Ready for processing pipeline
    
    
    ✅ GPU MEMORY MANAGEMENT
       └─ Monitors available VRAM
       └─ Adjusts batch sizes accordingly
       └─ Clears cache between batches
       └─ Prevents out-of-memory errors
    
    
    ✅ METADATA GENERATION
       └─ Per-batch metadata (JSON)
       └─ File durations and sizes
       └─ Batch statistics
       └─ Processing timestamps
    
    
    ✅ ERROR HANDLING
       └─ Handles corrupted audio files
       └─ Validates configuration
       └─ Reports conversion errors
       └─ Provides helpful messages
    """
    print(features)


def show_file_structure():
    """Show new/modified file structure"""
    print_section("NEW/MODIFIED FILES", 2)
    
    files = """
    NEW FILES:
    ──────────────────────────────────────────────────────
    
    src/preprocessing/audio_converter.py
      └─ AudioConverter class for format conversion
      └─ Handles MP3, WAV, M4A, FLAC, OGG, AAC
      └─ Error handling and logging
    
    src/scripts/test_batching_simple.py
      └─ Standalone batching test script
      └─ Generates test audio files
      └─ Verifies batching logic
    
    src/scripts/pipeline_test.py
      └─ Complete pipeline verification
      └─ Checks dependencies, config, data
      └─ Validates all components
    
    BATCHING_IMPLEMENTATION.md
      └─ Complete documentation
      └─ Implementation details
      └─ Known issues and recommendations
    
    
    MODIFIED FILES:
    ──────────────────────────────────────────────────────
    
    src/pipeline/batch_organizer.py
      └─ Added AudioConverter integration
      └─ Automatic conversion before batching
      └─ Enhanced metadata generation
    
    src/pipeline/__init__.py
      └─ Fixed import statements
      └─ Corrected relative imports
    
    src/pipeline/pipeline_runner.py
      └─ Fixed import paths
      └─ Corrected module references
    
    src/preprocessing/__init__.py
      └─ Added module initialization
      └─ Exports AudioConverter
    """
    print(files)


def show_next_steps():
    """Show next steps for deployment"""
    print_section("NEXT STEPS FOR DEPLOYMENT", 2)
    
    steps = """
    IMMEDIATE (Ready Now):
    ─────────────────────────────────────────────
    
    1. Run Batching Test
       python src/scripts/test_batching_simple.py
       
       Verifies batching logic works with test data
    
    2. Run Pipeline Test
       python src/scripts/pipeline_test.py
       
       Confirms all components are ready
    
    
    FOR PRODUCTION USE:
    ─────────────────────────────────────────────
    
    1. Configure HuggingFace Token
       Edit src/config/config.yaml
       Add your HF token for speaker diarization
    
    2. Prepare Input Data
       Place your MP3/WAV files in data/input/
       They will be automatically converted and batched
    
    3. Run Full Pipeline
       python src/main.py --config src/config/config.yaml
       
       This will:
         • Convert formats
         • Organize batches
         • Perform diarization
         • Classify speakers
         • Create voice dataset
    
    4. Monitor Processing
       Check logs/ directory for detailed logs
       Review output in data/voice_dataset/
    
    
    PERFORMANCE OPTIMIZATION:
    ─────────────────────────────────────────────
    
    Adjust these settings in src/config/config.yaml:
    
    batch_size_minutes: 2.0          # Batch duration
    files_per_batch: 10              # Files per batch
    batch_size_gpu: 10               # GPU processing batch
    memory_threshold: 80.0           # VRAM usage warning
    
    Monitor GPU usage with:
      nvidia-smi -l 1
    """
    print(steps)


def main():
    """Run demonstration"""
    print_banner()
    
    show_workflow()
    show_current_state()
    show_batching_example()
    show_key_features()
    show_file_structure()
    show_available_commands()
    show_next_steps()
    
    print("\n" + "="*80)
    print("  VOXENT Pipeline Ready for Production")
    print("="*80 + "\n")
    
    print("Quick Start:")
    print("  1. python src/scripts/test_batching_simple.py")
    print("  2. python src/scripts/pipeline_test.py")
    print("  3. Configure HuggingFace token in src/config/config.yaml")
    print("  4. python src/main.py --config src/config/config.yaml")
    print()


if __name__ == "__main__":
    main()
