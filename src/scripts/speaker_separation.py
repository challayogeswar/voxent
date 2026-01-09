#!/usr/bin/env python
"""
VOXENT Speaker Diarization & Voice Separation Pipeline
Separates individual speakers from audio files and classifies by gender
Labels output as: male_1, male_2, female_1, female_2, etc.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# Setup path
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/../..')
sys.path.insert(0, 'src')


def convert_to_wav(input_file, output_dir):
    """Convert audio file to WAV format"""
    try:
        y, sr = librosa.load(str(input_file), sr=16000, mono=True)
        output_path = os.path.join(output_dir, Path(input_file).stem + '.wav')
        sf.write(output_path, y, sr)
        return output_path, y, sr
    except Exception as e:
        print(f"  ❌ Error converting {input_file.name}: {e}")
        return None, None, None


def simple_voice_segmentation(y, sr, min_duration=0.5):
    """
    Simple voice activity detection to segment audio
    Returns list of (start_time, end_time, segment_audio)
    """
    # Trim silence
    y_trimmed, index = librosa.effects.trim(y, top_db=40, frame_length=2048, hop_length=512)
    
    # Use energy-based segmentation
    frame_length = 2048
    hop_length = 512
    
    # Calculate frame energies
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    energy = np.mean(S_db, axis=0)
    
    # Normalize energy
    energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-9)
    
    # Threshold for voice activity
    threshold = 0.3
    active_frames = energy > threshold
    
    # Find segments
    segments = []
    in_segment = False
    segment_start = 0
    
    for i, is_active in enumerate(active_frames):
        time = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
        
        if is_active and not in_segment:
            segment_start = time
            in_segment = True
        elif not is_active and in_segment:
            segment_end = time
            duration = segment_end - segment_start
            
            if duration >= min_duration:
                # Extract segment
                start_sample = int(segment_start * sr)
                end_sample = int(segment_end * sr)
                segment_audio = y[start_sample:end_sample]
                
                segments.append({
                    'start': segment_start,
                    'end': segment_end,
                    'duration': duration,
                    'audio': segment_audio
                })
            
            in_segment = False
    
    return segments


def classify_segment_gender(segment_audio, sr):
    """Classify gender of an audio segment"""
    try:
        y = segment_audio
        
        if len(y) < sr * 0.3:  # Less than 300ms
            return 'Unknown', 0
        
        # Extract features
        features = {}
        
        # 1. Pitch
        try:
            f0_values = librosa.yin(y, 80, 400, sr=sr)
            f0_values = f0_values[f0_values > 0]
            if len(f0_values) > 0:
                features['pitch'] = np.median(f0_values)
            else:
                features['pitch'] = 150
        except:
            features['pitch'] = 150
        
        # 2. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['centroid'] = np.mean(centroid)
        
        # 3. MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc'] = np.mean(mfcc[0])
        
        # 4. ZCR
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr'] = np.mean(zcr)
        
        # 5. Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['rolloff'] = np.mean(rolloff)
        
        # Classification
        male_score = 0
        female_score = 0
        
        # Pitch scoring
        pitch = features['pitch']
        if pitch < 120:
            male_score += 3
        elif pitch < 160:
            male_score += 1.5
        elif pitch < 200:
            female_score += 1
        else:
            female_score += 3
        
        # Spectral Centroid
        centroid = features['centroid']
        if centroid < 2000:
            male_score += 1.5
        elif centroid < 2500:
            male_score += 0.5
        elif centroid < 3500:
            female_score += 0.5
        else:
            female_score += 2
        
        # MFCC
        if features['mfcc'] < -500:
            male_score += 1.5
        else:
            female_score += 1
        
        # ZCR
        if features['zcr'] > 0.15:
            female_score += 1.5
        else:
            male_score += 1
        
        # Rolloff
        if features['rolloff'] > 5000:
            female_score += 2
        elif features['rolloff'] < 3000:
            male_score += 2
        
        # Determine gender
        total = male_score + female_score
        if total == 0:
            return 'Unknown', 0
        
        male_prob = male_score / total
        female_prob = female_score / total
        max_prob = max(male_prob, female_prob)
        
        if max_prob < 0.55:
            return 'Uncertain', max_prob
        
        gender = 'Male' if male_prob > female_prob else 'Female'
        return gender, max_prob
        
    except Exception as e:
        return 'Unknown', 0


def main():
    """Main processing pipeline"""
    
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "VOXENT SPEAKER DIARIZATION & SEPARATION".center(78) + "█")
    print("█" + "Extract & Classify Individual Speakers from Mixed Audio".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80 + "\n")
    
    input_dir = 'data/input'
    output_dir = 'data/voice_dataset'
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'input_files': 0,
        'extracted_segments': 0,
        'male_segments': 0,
        'female_segments': 0,
        'uncertain_segments': 0,
        'files': {}
    }
    
    # ========================
    # STEP 1: Scan input files
    # ========================
    print("STEP 1: SCANNING INPUT FILES")
    print("-" * 80 + "\n")
    
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'}
    input_files = sorted([f for f in Path(input_dir).iterdir() 
                         if f.is_file() and f.suffix.lower() in audio_extensions])
    
    print(f"✓ Found {len(input_files)} audio files\n")
    for f in input_files:
        print(f"  • {f.name}")
    
    results['input_files'] = len(input_files)
    
    # ========================
    # STEP 2: Process files
    # ========================
    print(f"\n\nSTEP 2: SPEAKER SEGMENTATION & CLASSIFICATION")
    print("-" * 80 + "\n")
    
    male_counter = 1
    female_counter = 1
    uncertain_counter = 1
    
    # Create output subdirectories
    Path(os.path.join(output_dir, 'male')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, 'female')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, 'uncertain')).mkdir(parents=True, exist_ok=True)
    
    for input_file in input_files:
        print(f"\n📁 Processing: {input_file.name}")
        print("-" * 60)
        
        # Convert to WAV
        wav_path, y, sr = convert_to_wav(input_file, input_dir)
        if wav_path is None:
            results['files'][input_file.name] = {'error': 'Failed to convert'}
            continue
        
        # Get duration
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"  Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        # Segment audio
        segments = simple_voice_segmentation(y, sr, min_duration=0.3)
        print(f"  Segments detected: {len(segments)}")
        
        if not segments:
            print(f"  ⚠️  No segments detected in file")
            results['files'][input_file.name] = {'error': 'No segments detected', 'segments': 0}
            continue
        
        file_result = {
            'original_duration': duration,
            'segments': []
        }
        
        # Classify each segment
        file_male = 0
        file_female = 0
        file_uncertain = 0
        
        for idx, segment in enumerate(segments):
            segment_audio = segment['audio']
            segment_duration = segment['duration']
            
            # Classify gender
            gender, confidence = classify_segment_gender(segment_audio, sr)
            
            # Create output filename
            if gender == 'Male':
                label = f'male_{male_counter}'
                output_path = os.path.join(output_dir, 'male', f'{label}.wav')
                male_counter += 1
                results['male_segments'] += 1
                file_male += 1
                
            elif gender == 'Female':
                label = f'female_{female_counter}'
                output_path = os.path.join(output_dir, 'female', f'{label}.wav')
                female_counter += 1
                results['female_segments'] += 1
                file_female += 1
                
            else:
                label = f'uncertain_{uncertain_counter}'
                output_path = os.path.join(output_dir, 'uncertain', f'{label}.wav')
                uncertain_counter += 1
                results['uncertain_segments'] += 1
                file_uncertain += 1
            
            # Save segment
            sf.write(output_path, segment_audio, sr)
            
            file_result['segments'].append({
                'segment': idx + 1,
                'gender': gender,
                'confidence': float(confidence),
                'duration': float(segment_duration),
                'output': label,
                'file': f'{label}.wav'
            })
            
            results['extracted_segments'] += 1
        
        print(f"\n  ✓ Segments classified:")
        print(f"    Male:     {file_male} segments")
        print(f"    Female:   {file_female} segments")
        if file_uncertain > 0:
            print(f"    Uncertain: {file_uncertain} segments")
        
        results['files'][input_file.name] = file_result
    
    # ========================
    # SUMMARY
    # ========================
    print("\n\n" + "="*80)
    print("PROCESSING COMPLETE - SUMMARY")
    print("="*80 + "\n")
    
    print("📊 STATISTICS:")
    print(f"  Input Files: {results['input_files']}")
    print(f"  Total Segments Extracted: {results['extracted_segments']}")
    
    print("\n🎤 SEGMENT CLASSIFICATION:")
    print(f"  Male Segments:     {results['male_segments']}")
    print(f"  Female Segments:   {results['female_segments']}")
    if results['uncertain_segments'] > 0:
        print(f"  Uncertain:         {results['uncertain_segments']}")
    
    print("\n📂 OUTPUT STRUCTURE:")
    print(f"  data/voice_dataset/")
    print(f"    ├── male/ ({results['male_segments']} files)")
    print(f"    ├── female/ ({results['female_segments']} files)")
    if results['uncertain_segments'] > 0:
        print(f"    └── uncertain/ ({results['uncertain_segments']} files)")
    
    print("\n📁 SEGMENT DETAILS:")
    for filename, file_info in results['files'].items():
        if 'error' in file_info:
            print(f"\n  {filename}: ERROR - {file_info['error']}")
        else:
            print(f"\n  {filename}:")
            for seg in file_info['segments']:
                print(f"    • Segment {seg['segment']}: {seg['gender']} ({seg['confidence']:.1%}) → {seg['file']}")
    
    # Save results
    with open('SPEAKER_SEPARATION_RESULTS.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to SPEAKER_SEPARATION_RESULTS.json")
    print("\n" + "="*80)
    print("✅ SPEAKER DIARIZATION COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
