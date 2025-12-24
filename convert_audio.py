import os
import glob
from pydub import AudioSegment
import argparse
from tqdm import tqdm

def convert_mp3_to_wav(input_dir, output_dir=None, sample_rate=16000):
    """
    Convert all MP3 files in input directory to WAV format.

    Args:
        input_dir (str): Directory containing MP3 files
        output_dir (str): Directory to save WAV files (default: same as input)
        sample_rate (int): Target sample rate for WAV files
    """
    if output_dir is None:
        output_dir = input_dir

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all MP3 files
    mp3_files = glob.glob(os.path.join(input_dir, '*.mp3'))

    if not mp3_files:
        print(f"No MP3 files found in {input_dir}")
        return

    print(f"Found {len(mp3_files)} MP3 files to convert")

    converted_count = 0
    for mp3_file in tqdm(mp3_files, desc="Converting files"):
        try:
            # Load MP3 file
            audio = AudioSegment.from_mp3(mp3_file)

            # Convert to mono and set sample rate
            audio = audio.set_channels(1).set_frame_rate(sample_rate)

            # Generate output filename
            base_name = os.path.splitext(os.path.basename(mp3_file))[0]
            wav_file = os.path.join(output_dir, f"{base_name}.wav")

            # Export as WAV
            audio.export(wav_file, format='wav')

            converted_count += 1

        except Exception as e:
            print(f"Error converting {mp3_file}: {e}")
            continue

    print(f"Successfully converted {converted_count}/{len(mp3_files)} files")

def main():
    parser = argparse.ArgumentParser(description='Convert MP3 files to WAV format')
    parser.add_argument('--input', '-i', default='data',
                       help='Input directory containing MP3 files (default: data)')
    parser.add_argument('--output', '-o', default='data/input_calls',
                       help='Output directory for WAV files (default: data/input_calls)')
    parser.add_argument('--sample-rate', '-r', type=int, default=16000,
                       help='Target sample rate (default: 16000)')

    args = parser.parse_args()

    convert_mp3_to_wav(args.input, args.output, args.sample_rate)

if __name__ == "__main__":
    main()