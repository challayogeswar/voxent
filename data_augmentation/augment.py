import os
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import random
from tqdm import tqdm

class AudioAugmenter:
    """Audio data augmentation class for voice dataset enhancement."""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def add_noise(self, audio, noise_factor=0.005):
        """Add random noise to audio."""
        noise = np.random.normal(0, noise_factor, len(audio))
        return audio + noise

    def change_pitch(self, audio, pitch_factor_range=(-0.1, 0.1)):
        """Change pitch by resampling."""
        pitch_factor = random.uniform(*pitch_factor_range)
        new_rate = int(self.sample_rate * (1 + pitch_factor))
        # Simple pitch change by resampling (not perfect but works for augmentation)
        return np.interp(
            np.linspace(0, len(audio), int(len(audio) * self.sample_rate / new_rate)),
            np.arange(len(audio)),
            audio
        )

    def change_speed(self, audio, speed_factor_range=(0.9, 1.1)):
        """Change playback speed."""
        speed_factor = random.uniform(*speed_factor_range)
        new_length = int(len(audio) / speed_factor)
        return np.interp(
            np.linspace(0, len(audio) - 1, new_length),
            np.arange(len(audio)),
            audio
        )

    def change_volume(self, audio, volume_factor_range=(0.8, 1.2)):
        """Change volume."""
        volume_factor = random.uniform(*volume_factor_range)
        return audio * volume_factor

    def apply_augmentation(self, audio, methods=['add_noise', 'change_pitch', 'change_speed', 'change_volume']):
        """Apply random augmentations to audio."""
        augmented = audio.copy()

        for method in methods:
            if random.random() < 0.7:  # 70% chance to apply each augmentation
                if method == 'add_noise':
                    augmented = self.add_noise(augmented)
                elif method == 'change_pitch':
                    augmented = self.change_pitch(augmented)
                elif method == 'change_speed':
                    augmented = self.change_speed(augmented)
                elif method == 'change_volume':
                    augmented = self.change_volume(augmented)

        # Normalize to prevent clipping
        max_val = np.max(np.abs(augmented))
        if max_val > 1.0:
            augmented = augmented / max_val

        return augmented

def balance_dataset(dataset_dir, cfg):
    """
    Balance dataset by augmenting underrepresented classes.

    Args:
        dataset_dir (str): Path to voice_dataset directory
        cfg (dict): Configuration dictionary
    """
    print("Starting dataset balancing with augmentation...")

    # Count files in each category
    categories = ['male', 'female', 'uncertain']
    category_counts = {}

    for category in categories:
        category_dir = os.path.join(dataset_dir, category)
        if os.path.exists(category_dir):
            files = [f for f in os.listdir(category_dir) if f.endswith('.wav')]
            category_counts[category] = len(files)
        else:
            category_counts[category] = 0

    print(f"Current distribution: {category_counts}")

    # Find the category with maximum files
    max_count = max(category_counts.values())
    target_count = max_count  # Balance to the largest category

    augmenter = AudioAugmenter(cfg.get('sample_rate', 16000))
    max_augmentations = cfg.get('max_augmentations_per_sample', 3)
    intensity = cfg.get('augmentation_intensity', 0.1)

    total_augmentations = 0

    for category in categories:
        category_dir = os.path.join(dataset_dir, category)
        if not os.path.exists(category_dir):
            continue

        current_count = category_counts[category]
        augmentations_needed = target_count - current_count

        if augmentations_needed <= 0:
            continue

        print(f"Augmenting {category} category: {current_count} -> {target_count}")

        # Get list of files to augment
        files = [f for f in os.listdir(category_dir) if f.endswith('.wav')]

        # Augment files
        augmentations_per_file = min(max_augmentations, max(1, augmentations_needed // len(files)))
        augmentations_created = 0

        for file in tqdm(files, desc=f"Augmenting {category}"):
            if augmentations_created >= augmentations_needed:
                break

            file_path = os.path.join(category_dir, file)

            try:
                # Load audio
                audio, sr = sf.read(file_path)

                # Create augmentations
                for i in range(augmentations_per_file):
                    if augmentations_created >= augmentations_needed:
                        break

                    # Apply augmentation
                    augmented = augmenter.apply_augmentation(audio)

                    # Generate new filename
                    base_name = os.path.splitext(file)[0]
                    new_filename = f"{base_name}_aug{i+1}.wav"
                    new_path = os.path.join(category_dir, new_filename)

                    # Save augmented file
                    sf.write(new_path, augmented, sr)

                    augmentations_created += 1
                    total_augmentations += 1

            except Exception as e:
                print(f"Error augmenting {file}: {e}")
                continue

    print(f"Dataset balancing completed. Created {total_augmentations} augmented samples.")
    print(f"Final distribution: {get_category_counts(dataset_dir)}")

def get_category_counts(dataset_dir):
    """Get current file counts for each category."""
    categories = ['male', 'female', 'uncertain']
    counts = {}

    for category in categories:
        category_dir = os.path.join(dataset_dir, category)
        if os.path.exists(category_dir):
            files = [f for f in os.listdir(category_dir) if f.endswith('.wav')]
            counts[category] = len(files)
        else:
            counts[category] = 0

    return counts