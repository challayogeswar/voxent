"""
Batch Organizer Module
Organizes audio files into duration-based batches for GPU-optimized processing

Key Features:
- Converts MP3 and other formats to WAV before batching
- Sorts files by duration (smallest to largest)
- Groups into batches respecting GPU memory limits
- Creates physical batch folders (batch_001, batch_002, etc.)
- Monitors and reports batch statistics
- Duration-based batching (e.g., 0-2 min, 2-4 min, etc.)
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import json
import librosa
import logging
from datetime import datetime
from preprocessing.audio_converter import AudioConverter

logger = logging.getLogger(__name__)


class BatchOrganizer:
    """
    Organizes audio files into batches based on duration
    Automatically converts MP3 and other formats to WAV
    """
    
    def __init__(self, config: Dict):
        """
        Initialize batch organizer
        
        Args:
            config: Configuration dictionary with batch settings
        """
        self.config = config
        
        # Batch configuration
        self.files_per_batch = config.get('files_per_batch', 10)
        self.batch_size_minutes = config.get('batch_size_minutes', 2.0)
        self.batch_size_seconds = self.batch_size_minutes * 60
        
        # Audio converter
        self.converter = AudioConverter(
            sample_rate=config.get('sample_rate', 16000),
            mono=config.get('mono', True)
        )
        
        print(f"Batch Organizer initialized:")
        print(f"  - Files per batch: {self.files_per_batch}")
        print(f"  - Max batch duration: {self.batch_size_minutes} minutes")
        print(f"  - Auto-convert MP3 to WAV: Enabled")
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        Get duration of audio file in seconds
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        try:
            duration = librosa.get_duration(path=audio_path)
            return duration
        except Exception as e:
            print(f"Error getting duration for {audio_path}: {e}")
            return 0.0
    
    def scan_audio_files(self, input_dir: str) -> List[Dict]:
        """
        Scan directory for audio files and get their durations
        
        Args:
            input_dir: Directory containing audio files
            
        Returns:
            List of file info dictionaries sorted by duration
        """
        print(f"\nScanning audio files in: {input_dir}")
        
        # Supported audio formats
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'}
        
        # Collect file info
        file_info = []
        for root, _, files in os.walk(input_dir):
            for filename in files:
                if Path(filename).suffix.lower() in audio_extensions:
                    file_path = os.path.join(root, filename)
                    duration = self.get_audio_duration(file_path)
                    
                    if duration > 0:
                        file_info.append({
                            'path': file_path,
                            'filename': filename,
                            'duration': duration,
                            'size_mb': os.path.getsize(file_path) / (1024 * 1024)
                        })
        
        # Sort by duration (smallest to largest)
        file_info.sort(key=lambda x: x['duration'])
        
        print(f"Found {len(file_info)} audio files")
        if file_info:
            total_duration = sum(f['duration'] for f in file_info)
            print(f"Total duration: {total_duration/60:.2f} minutes")
            print(f"Duration range: {file_info[0]['duration']:.1f}s - {file_info[-1]['duration']:.1f}s")
        
        return file_info
    
    def create_duration_range_batches(self, file_info: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Create batches based on duration ranges
        Each file is assigned to a batch based on its INDIVIDUAL duration
        
        Duration ranges:
        - Batch 1: 0-120 sec (0-2 min)
        - Batch 2: 121-240 sec (2-4 min)
        - Batch 3: 241-600 sec (4-10 min)
        - Batch 4: 600-1200 sec (10-20 min)
        - Batch 5: 1200-1800 sec (20-30 min)
        - Batch 6: 1800-2400 sec (30-40 min)
        - Batch 7: 2400-3000 sec (40-50 min)
        - ... up to 2 hours (7200 sec)
        
        Args:
            file_info: List of file info dictionaries
            
        Returns:
            Dictionary with batch names as keys and file lists as values
        """
        # Define duration ranges in seconds
        # Format: (min_duration, max_duration, batch_name, batch_number)
        duration_ranges = [
            (0, 120, "batch_001", "0-2 min"),
            (121, 240, "batch_002", "2-4 min"),
            (241, 600, "batch_003", "4-10 min"),
            (601, 1200, "batch_004", "10-20 min"),
            (1201, 1800, "batch_005", "20-30 min"),
            (1801, 2400, "batch_006", "30-40 min"),
            (2401, 3000, "batch_007", "40-50 min"),
            (3001, 3600, "batch_008", "50-60 min"),
            (3601, 7200, "batch_009", "60-120 min"),
        ]
        
        # Initialize batches
        batches = {}
        batch_info = {}
        
        for min_sec, max_sec, batch_name, range_label in duration_ranges:
            batches[batch_name] = []
            batch_info[batch_name] = {
                'range_label': range_label,
                'min_duration': min_sec,
                'max_duration': max_sec
            }
        
        # Assign files to batches based on their individual duration
        unassigned_files = []
        
        for file in file_info:
            file_duration = file['duration']
            assigned = False
            
            # Find appropriate batch for this file
            for min_sec, max_sec, batch_name, _ in duration_ranges:
                if min_sec <= file_duration <= max_sec:
                    batches[batch_name].append(file)
                    assigned = True
                    break
            
            # Handle files longer than 2 hours
            if not assigned:
                if file_duration > 7200:
                    batches['batch_009'].append(file)
                else:
                    unassigned_files.append(file)
        
        return batches, batch_info, unassigned_files
    
    def create_batches(self, file_info: List[Dict]) -> List[List[Dict]]:
        """
        Create batches from file info list (Duration-range based)
        
        Args:
            file_info: List of file info dictionaries
            
        Returns:
            List of batches (each batch is a list of file info)
        """
        # Use duration range batching
        batches_dict, batch_info, unassigned = self.create_duration_range_batches(file_info)
        
        # Convert to ordered list format, filtering out empty batches
        batches = []
        for batch_name in sorted(batches_dict.keys()):
            if batches_dict[batch_name]:  # Only include non-empty batches
                batches.append(batches_dict[batch_name])
        
        return batches
    
    def organize_into_folders(
        self, 
        batches: List[List[Dict]], 
        output_dir: str,
        copy_files: bool = True,
        duration_ranges: bool = True
    ) -> List[str]:
        """
        Create batch folders and organize files
        
        Args:
            batches: List of batches from create_batches()
            output_dir: Base directory for batch folders
            copy_files: If True, copy files; if False, move files
            duration_ranges: If True, use duration-range batch names
            
        Returns:
            List of created batch folder paths
        """
        print(f"\nOrganizing into batch folders: {output_dir}")
        
        # Create base output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        batch_folders = []
        duration_range_names = {
            0: "0-2_min",
            1: "2-4_min",
            2: "4-10_min",
            3: "10-20_min",
            4: "20-30_min",
            5: "30-40_min",
            6: "40-50_min",
            7: "50-60_min",
            8: "60-120_min",
        }
        
        for batch_idx, batch in enumerate(batches):
            if duration_ranges and batch_idx < 9:
                # Use duration range naming
                range_name = duration_range_names.get(batch_idx, f"batch_{batch_idx+1:03d}")
                batch_name = f"batch_{batch_idx+1:03d}_{range_name}"
            else:
                batch_name = f"batch_{batch_idx+1:03d}"
            
            batch_path = os.path.join(output_dir, batch_name)
            Path(batch_path).mkdir(parents=True, exist_ok=True)
            
            # Copy/move files to batch folder
            batch_duration = 0.0
            file_count = 0
            
            for file_info in batch:
                src_path = file_info['path']
                dst_path = os.path.join(batch_path, file_info['filename'])
                
                if copy_files and src_path != dst_path:
                    shutil.copy2(src_path, dst_path)
                elif not copy_files:
                    shutil.move(src_path, dst_path)
                
                batch_duration += file_info['duration']
                file_count += 1
            
            # Save batch metadata
            batch_metadata = {
                'batch_number': batch_idx + 1,
                'batch_name': batch_name,
                'num_files': len(batch),
                'total_duration_seconds': batch_duration,
                'total_duration_minutes': batch_duration / 60,
                'files': [
                    {
                        'filename': f['filename'],
                        'duration': f['duration'],
                        'duration_minutes': f['duration'] / 60,
                        'size_mb': f['size_mb']
                    }
                    for f in batch
                ],
                'created_timestamp': datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(batch_path, 'batch_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(batch_metadata, f, indent=2)
            
            batch_folders.append(batch_path)
            
            duration_min = batch_duration / 60
            print(f"  ✓ {batch_name}: {len(batch)} files, {duration_min:.2f} min total")
        
        print(f"\n✅ Created {len(batch_folders)} batches")
        return batch_folders
    
    def organize_directory(
        self, 
        input_dir: str, 
        output_dir: str,
        copy_files: bool = True
    ) -> Dict:
        """
        Complete pipeline: convert, scan, batch, and organize
        
        Args:
            input_dir: Directory containing audio files to organize
            output_dir: Base directory for batch folders
            copy_files: If True, copy files; if False, move files
            
        Returns:
            Dictionary with organization results
        """
        print(f"\n{'='*60}")
        print(f"BATCH ORGANIZATION WITH CONVERSION")
        print(f"{'='*60}")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Mode: {'COPY' if copy_files else 'MOVE'}")
        
        # Step 0: Convert MP3 and other formats to WAV
        print(f"\n{'#'*60}")
        print(f"STEP 0: AUDIO FORMAT CONVERSION")
        print(f"{'#'*60}\n")
        
        conversion_result = self.converter.convert_directory(
            input_dir=input_dir,
            output_dir=input_dir,  # Convert in place
            replace=False
        )
        
        # Step 1: Scan files (now all should be WAV)
        file_info = self.scan_audio_files(input_dir)
        
        if not file_info:
            print("❌ No audio files found!")
            return {'error': 'No audio files found', 'batches': []}
        
        # Step 2: Create batches
        batches = self.create_batches(file_info)
        print(f"\nCreated {len(batches)} batches")
        
        # Step 3: Organize into folders
        batch_folders = self.organize_into_folders(batches, output_dir, copy_files)
        
        # Summary
        total_files = sum(len(batch) for batch in batches)
        total_duration = sum(sum(f['duration'] for f in batch) for batch in batches)
        
        result = {
            'input_dir': input_dir,
            'output_dir': output_dir,
            'conversion': conversion_result,
            'num_batches': len(batches),
            'total_files': total_files,
            'total_duration_minutes': total_duration / 60,
            'batch_folders': batch_folders,
            'batches': [
                {
                    'batch_number': idx + 1,
                    'num_files': len(batch),
                    'duration_seconds': sum(f['duration'] for f in batch),
                    'duration_minutes': sum(f['duration'] for f in batch) / 60
                }
                for idx, batch in enumerate(batches)
            ]
        }
        
        # Save overall summary
        summary_path = os.path.join(output_dir, 'organization_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"ORGANIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total batches: {len(batches)}")
        print(f"Total files: {total_files}")
        print(f"Total duration: {total_duration/60:.2f} minutes")
        
        # Print batch details
        print(f"\nBatch Details:")
        for batch_info in result['batches']:
            print(f"  Batch {batch_info['batch_number']:03d}: "
                  f"{batch_info['num_files']} files, "
                  f"{batch_info['duration_minutes']:.2f} minutes")
        
        print(f"\nSummary saved: {summary_path}")
        
        return result


def main():
    """Example usage"""
    
    # Configuration
    config = {
        'files_per_batch': 10,
        'batch_size_minutes': 2.0
    }
    
    # Initialize organizer
    organizer = BatchOrganizer(config)
    
    # Organize files
    input_directory = "data/input_calls"
    output_directory = "data/batches"
    
    result = organizer.organize_directory(
        input_dir=input_directory,
        output_dir=output_directory,
        copy_files=True
    )
    
    print(f"\nOrganization result: {result}")


if __name__ == "__main__":
    main()
