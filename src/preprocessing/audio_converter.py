"""
Audio Conversion Module
Handles conversion of various audio formats to WAV
Supports MP3, M4A, FLAC, OGG, AAC formats
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Tuple
import json
from datetime import datetime
import librosa
import soundfile as sf
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class AudioConverter:
    """
    Handles conversion of audio files to WAV format
    """
    
    def __init__(self, sample_rate: int = 16000, mono: bool = True):
        """
        Initialize audio converter
        
        Args:
            sample_rate: Target sample rate (default: 16000 Hz)
            mono: Convert to mono (default: True)
        """
        self.sample_rate = sample_rate
        self.mono = mono
        self.supported_formats = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
        
        print(f"AudioConverter initialized:")
        print(f"  - Target sample rate: {self.sample_rate} Hz")
        print(f"  - Mono conversion: {self.mono}")
        print(f"  - Supported formats: {', '.join(self.supported_formats)}")
    
    def convert_file(self, input_path: str, output_path: str) -> Tuple[bool, str]:
        """
        Convert a single audio file to WAV format
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save output WAV file
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Load audio using librosa
            audio, sr = librosa.load(
                input_path,
                sr=self.sample_rate,
                mono=self.mono
            )
            
            # Save as WAV
            sf.write(output_path, audio, self.sample_rate)
            
            return True, f"✓ Converted: {Path(input_path).name}"
        
        except Exception as e:
            error_msg = f"✗ Error converting {Path(input_path).name}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def scan_audio_files(self, input_dir: str) -> Tuple[List[str], List[str]]:
        """
        Scan directory for audio files
        
        Args:
            input_dir: Directory to scan
            
        Returns:
            Tuple of (wav_files: List[str], other_format_files: List[str])
        """
        wav_files = []
        other_files = []
        
        for root, _, files in os.walk(input_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                suffix = Path(filename).suffix.lower()
                
                if suffix in self.supported_formats:
                    if suffix == '.wav':
                        wav_files.append(file_path)
                    else:
                        other_files.append(file_path)
        
        return wav_files, other_files
    
    def convert_directory(
        self,
        input_dir: str,
        output_dir: str = None,
        replace: bool = False
    ) -> Dict:
        """
        Convert all non-WAV audio files in directory to WAV
        
        Args:
            input_dir: Directory containing audio files
            output_dir: Directory to save converted files (default: same as input)
            replace: Replace converted files in input directory (default: False)
            
        Returns:
            Dictionary with conversion results
        """
        if output_dir is None:
            output_dir = input_dir
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"\nScanning audio files in: {input_dir}")
        
        # Scan for files
        wav_files, other_files = self.scan_audio_files(input_dir)
        
        print(f"  • WAV files: {len(wav_files)}")
        print(f"  • Other formats: {len(other_files)}")
        
        if not other_files:
            print("\n✅ No conversion needed - all files are WAV format")
            return {
                'input_dir': input_dir,
                'output_dir': output_dir,
                'wav_files_found': len(wav_files),
                'files_converted': 0,
                'files_skipped': 0,
                'successful': 0,
                'failed': 0,
                'conversion_details': []
            }
        
        print(f"\n{'='*70}")
        print(f"AUDIO FORMAT CONVERSION")
        print(f"{'='*70}")
        print(f"Converting {len(other_files)} files to WAV format...\n")
        
        # Convert files
        successful = 0
        failed = 0
        conversion_details = []
        
        for file_path in tqdm(other_files, desc="Converting files"):
            filename = Path(file_path).name
            base_name = Path(filename).stem
            output_path = os.path.join(output_dir, f"{base_name}.wav")
            
            # Skip if output already exists (unless replacing)
            if os.path.exists(output_path) and not replace:
                conversion_details.append({
                    'input': filename,
                    'status': 'skipped',
                    'reason': 'Output file already exists'
                })
                continue
            
            # Convert file
            success, message = self.convert_file(file_path, output_path)
            
            if success:
                successful += 1
                conversion_details.append({
                    'input': filename,
                    'output': f"{base_name}.wav",
                    'status': 'success'
                })
            else:
                failed += 1
                conversion_details.append({
                    'input': filename,
                    'status': 'failed',
                    'reason': message
                })
        
        # Summary
        print(f"\n{'='*70}")
        print(f"CONVERSION COMPLETE")
        print(f"{'='*70}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total WAV files available: {len(wav_files) + successful}")
        
        result = {
            'input_dir': input_dir,
            'output_dir': output_dir,
            'wav_files_found': len(wav_files),
            'files_converted': successful,
            'files_skipped': len(other_files) - successful - failed,
            'successful': successful,
            'failed': failed,
            'total_wav_available': len(wav_files) + successful,
            'conversion_details': conversion_details,
            'timestamp': datetime.now().isoformat()
        }
        
        return result


def main():
    """Example usage"""
    
    converter = AudioConverter(sample_rate=16000, mono=True)
    
    # Convert files in input directory
    input_dir = "data/input"
    output_dir = "data/input"  # Save converted files in same directory
    
    result = converter.convert_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        replace=False
    )
    
    print(f"\nConversion result: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
