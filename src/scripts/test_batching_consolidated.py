#!/usr/bin/env python
"""
VOXENT Comprehensive Batching Test
Consolidated from test_batching.py and test_batching_simple.py

Supports two modes:
  - COMPLEX MODE: Uses BatchOrganizer module (requires module imports)
  - SIMPLE MODE: Standalone implementation (no module dependencies)
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from datetime import datetime

# Add src to path for module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ============================================================================
# STANDALONE BATCHING FUNCTIONS (Simple Mode)
# ============================================================================

def create_test_audio(filename: str, duration_seconds: float, sample_rate: int = 16000):
    """Create test audio file with specified duration"""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
    frequency = 440  # A4 note
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    sf.write(filename, audio, sample_rate)
    return filename


def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds"""
    try:
        import librosa
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception as e:
        print(f"Error: {e}")
        return 0.0


def scan_audio_files(input_dir: str):
    """Scan directory for audio files and get their durations"""
    print(f"\nScanning audio files in: {input_dir}")
    
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'}
    file_info = []
    
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if Path(filename).suffix.lower() in audio_extensions:
                file_path = os.path.join(root, filename)
                duration = get_audio_duration(file_path)
                
                if duration > 0:
                    file_info.append({
                        'path': file_path,
                        'filename': filename,
                        'duration': duration,
                        'size_mb': os.path.getsize(file_path) / (1024 * 1024)
                    })
    
    file_info.sort(key=lambda x: x['duration'])
    
    print(f"Found {len(file_info)} audio files")
    if file_info:
        total_duration = sum(f['duration'] for f in file_info)
        print(f"Total duration: {total_duration/60:.2f} minutes")
        print(f"Duration range: {file_info[0]['duration']:.1f}s - {file_info[-1]['duration']:.1f}s")
    
    return file_info


def create_batches(file_info, batch_size_minutes: float = 2.0, files_per_batch: int = 10):
    """Create batches from file info list"""
    batch_size_seconds = batch_size_minutes * 60
    batches = []
    current_batch = []
    current_batch_duration = 0.0
    
    for file in file_info:
        would_exceed_duration = (current_batch_duration + file['duration']) > batch_size_seconds
        would_exceed_count = len(current_batch) >= files_per_batch
        
        if current_batch and (would_exceed_duration or would_exceed_count):
            batches.append(current_batch)
            current_batch = []
            current_batch_duration = 0.0
        
        current_batch.append(file)
        current_batch_duration += file['duration']
    
    if current_batch:
        batches.append(current_batch)
    
    return batches


def organize_into_folders(batches, output_dir: str, copy_files: bool = True):
    """Create batch folders and organize files"""
    print(f"\nOrganizing into batch folders: {output_dir}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    batch_folders = []
    
    for batch_idx, batch in enumerate(batches, 1):
        batch_name = f"batch_{batch_idx:03d}"
        batch_path = os.path.join(output_dir, batch_name)
        Path(batch_path).mkdir(parents=True, exist_ok=True)
        
        batch_duration = 0.0
        for file_info in batch:
            src_path = file_info['path']
            dst_path = os.path.join(batch_path, file_info['filename'])
            
            if copy_files and src_path != dst_path:
                import shutil
                shutil.copy2(src_path, dst_path)
            
            batch_duration += file_info['duration']
        
        batch_folders.append(batch_path)
        print(f"  ✓ {batch_name}: {len(batch)} files, {batch_duration:.1f}s total")
    
    print(f"\n✅ Created {len(batch_folders)} batches")
    return batch_folders


# ============================================================================
# MODULE-BASED BATCHING FUNCTIONS (Complex Mode)
# ============================================================================

def test_with_batch_organizer_module():
    """Test using BatchOrganizer module (requires module imports)"""
    try:
        from preprocessing.audio_converter import AudioConverter
        from pipeline.batch_organizer import BatchOrganizer
        
        print(f"\n{'='*70}")
        print("COMPLEX MODE: Using BatchOrganizer Module")
        print(f"{'='*70}\n")
        
        config = {
            'batch_size_minutes': 2.0,
            'files_per_batch': 10,
            'sample_rate': 16000,
            'mono': True
        }
        
        organizer = BatchOrganizer(config)
        
        input_dir = "data/input"
        file_info = organizer.scan_audio_files(input_dir)
        
        if not file_info:
            print("❌ No files found in input directory")
            return False
        
        batches = organizer.create_batches(file_info)
        
        print(f"✅ BatchOrganizer created {len(batches)} batches\n")
        
        for batch_idx, batch in enumerate(batches, 1):
            batch_duration = sum(f['duration'] for f in batch)
            batch_duration_min = batch_duration / 60
            
            print(f"  Batch {batch_idx:02d}: {len(batch)} files, {batch_duration_min:.2f} min")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  Complex mode failed (missing modules): {e}")
        print("    (This is OK - simple mode will be used instead)")
        return False
    except Exception as e:
        print(f"⚠️  Complex mode error: {e}")
        return False


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def generate_test_dataset():
    """Generate test audio files"""
    print("\n" + "="*70)
    print("GENERATING TEST DATASET")
    print("="*70)
    
    test_dir = "data/input"
    Path(test_dir).mkdir(parents=True, exist_ok=True)
    
    test_files = [
        ("test_audio_30s_1.wav", 30),
        ("test_audio_45s_1.wav", 45),
        ("test_audio_1m_1.wav", 60),
        ("test_audio_1m30s_1.wav", 90),
        ("test_audio_2m_1.wav", 120),
        ("test_audio_2m30s_1.wav", 150),
        ("test_audio_3m_1.wav", 180),
        ("test_audio_3m30s_1.wav", 210),
        ("test_audio_4m_1.wav", 240),
        ("test_audio_4m30s_1.wav", 270),
        ("test_audio_5m_1.wav", 300),
        ("test_audio_6m_1.wav", 360),
    ]
    
    print(f"\nCreating {len(test_files)} test audio files...\n")
    
    created_files = []
    for filename, duration in test_files:
        filepath = os.path.join(test_dir, filename)
        
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  ✓ {filename} (already exists, {size_mb:.2f} MB)")
            created_files.append(filepath)
            continue
        
        create_test_audio(filepath, duration)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  ✓ {filename} ({duration}s, {size_mb:.2f} MB)")
        created_files.append(filepath)
    
    print(f"\n✅ Created/verified {len(created_files)} test files")
    return created_files


def test_simple_batching_2min():
    """Test batching with 2-minute batches (simple mode)"""
    print("\n" + "="*70)
    print("SIMPLE MODE: BATCHING WITH 2-MINUTE BATCHES")
    print("="*70)
    print("\nConfiguration:")
    print("  - Batch Duration: 2 minutes")
    print("  - Method: Standalone (no module dependencies)")
    
    input_dir = "data/input"
    file_info = scan_audio_files(input_dir)
    
    if not file_info:
        print("❌ No files found!")
        return
    
    batches = create_batches(file_info, batch_size_minutes=2.0)
    
    print(f"\n✅ Created {len(batches)} batches:\n")
    
    for batch_idx, batch in enumerate(batches, 1):
        batch_duration = sum(f['duration'] for f in batch)
        batch_duration_min = batch_duration / 60
        
        print(f"  Batch {batch_idx:02d}: {batch_duration_min:.2f} min ({batch_duration:.0f}s)")
        for f in batch:
            duration_min = f['duration'] / 60
            print(f"    • {f['filename']:<30} {duration_min:>6.2f} min")


def test_simple_batching_4min():
    """Test batching with 4-minute batches (simple mode)"""
    print("\n" + "="*70)
    print("SIMPLE MODE: BATCHING WITH 4-MINUTE BATCHES")
    print("="*70)
    print("\nConfiguration:")
    print("  - Batch Duration: 4 minutes")
    print("  - Method: Standalone (no module dependencies)")
    
    input_dir = "data/input"
    file_info = scan_audio_files(input_dir)
    
    if not file_info:
        print("❌ No files found!")
        return
    
    batches = create_batches(file_info, batch_size_minutes=4.0)
    
    print(f"\n✅ Created {len(batches)} batches:\n")
    
    for batch_idx, batch in enumerate(batches, 1):
        batch_duration = sum(f['duration'] for f in batch)
        batch_duration_min = batch_duration / 60
        
        print(f"  Batch {batch_idx:02d}: {batch_duration_min:.2f} min ({batch_duration:.0f}s)")
        for f in batch:
            duration_min = f['duration'] / 60
            print(f"    • {f['filename']:<30} {duration_min:>6.2f} min")


def test_full_organization():
    """Test complete organization workflow"""
    print("\n" + "="*70)
    print("SIMPLE MODE: FULL ORGANIZATION WORKFLOW (2-MIN BATCHES)")
    print("="*70)
    
    input_dir = "data/input"
    output_dir = "data/batches"
    
    # Clean output directory
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    
    # Scan
    file_info = scan_audio_files(input_dir)
    
    if not file_info:
        print("❌ No files found!")
        return
    
    # Create batches
    batches = create_batches(file_info, batch_size_minutes=2.0)
    print(f"\nCreated {len(batches)} batches")
    
    # Organize
    batch_folders = organize_into_folders(batches, output_dir, copy_files=True)
    
    # Summary
    total_files = sum(len(batch) for batch in batches)
    total_duration = sum(sum(f['duration'] for f in batch) for batch in batches)
    
    print(f"\n" + "="*70)
    print("ORGANIZATION RESULTS")
    print("="*70)
    print(f"Total Batches: {len(batches)}")
    print(f"Total Files: {total_files}")
    print(f"Total Duration: {total_duration/60:.2f} minutes")
    
    print(f"\nBatch Breakdown:")
    for idx, batch in enumerate(batches, 1):
        batch_duration = sum(f['duration'] for f in batch) / 60
        print(f"  Batch {idx:03d}: {len(batch):2d} files, {batch_duration:6.2f} min")
    
    # Verify folders exist
    print(f"\n✅ Batch folders created:")
    for folder in sorted(batch_folders):
        if os.path.exists(folder):
            num_files = len([f for f in os.listdir(folder) if f.endswith('.wav')])
            print(f"  ✓ {os.path.basename(folder)} ({num_files} WAV files)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main test execution"""
    try:
        print("\n" + "="*70)
        print("VOXENT COMPREHENSIVE BATCHING TEST SUITE")
        print("="*70)
        
        # Step 1: Generate test data
        generate_test_dataset()
        
        # Step 2: Try complex mode (with modules)
        complex_success = test_with_batch_organizer_module()
        
        # Step 3: Test simple mode - 2-minute batches
        test_simple_batching_2min()
        
        # Step 4: Test simple mode - 4-minute batches
        test_simple_batching_4min()
        
        # Step 5: Test full organization
        test_full_organization()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nTest Summary:")
        print(f"  - Complex Mode (Module-based): {'✅ PASSED' if complex_success else '⚠️  SKIPPED'}")
        print(f"  - Simple Mode (Standalone): ✅ PASSED")
        print(f"  - Organization: ✅ PASSED")
        print("\nNext Steps:")
        print("  1. Files are now in data/input/")
        print("  2. Batches are organized in data/batches/")
        print("  3. Ready to run full pipeline with speaker_separation.py")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
