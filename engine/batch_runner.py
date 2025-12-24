
import os
import yaml
import logging
import psutil
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from preprocessing.audio_loader import load_audio
from preprocessing.normalize import normalize
from preprocessing.vad import remove_silence
from dIarization.diarizer import diarize
from dIarization.segments import extract_segment
from classification import get_classifier
from dataset.organizer import save_sample
from dataset.metadata import append_metadata
from quality_assurance.metrics import QualityMetrics
from data_augmentation.augment import balance_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_config(cfg):
    """Validate configuration parameters."""
    required_keys = ['sample_rate', 'min_segment_duration']
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Missing required config key: {key}")

    if cfg['sample_rate'] <= 0:
        raise ValueError("sample_rate must be positive")
    if cfg['min_segment_duration'] <= 0:
        raise ValueError("min_segment_duration must be positive")

    # Validate classification settings
    if 'classification' in cfg:
        class_cfg = cfg['classification']
        if 'pitch_male_threshold' in class_cfg and 'pitch_female_threshold' in class_cfg:
            if class_cfg['pitch_male_threshold'] >= class_cfg['pitch_female_threshold']:
                raise ValueError("pitch_male_threshold must be less than pitch_female_threshold")

    # Legacy validation for backward compatibility
    if 'male_pitch_threshold' in cfg and 'female_pitch_threshold' in cfg:
        if cfg['male_pitch_threshold'] >= cfg['female_pitch_threshold']:
            raise ValueError("male_pitch_threshold must be less than female_pitch_threshold")

    logger.info("Configuration validation passed")
    return True

def monitor_performance():
    """Monitor system performance during processing."""
    memory = psutil.virtual_memory()
    return {
        "memory_percent": memory.percent,
        "memory_used_gb": memory.used / (1024**3),
        "cpu_percent": psutil.cpu_percent(interval=1)
    }

def process_file(file_path, cfg):
    """Process a single audio file."""
    start_time = time.time()

    try:
        logger.info(f"Processing file: {os.path.basename(file_path)}")

        # Load and validate configuration
        validate_config(cfg)

        # Initialize integrated classifier
        classifier = get_classifier(cfg)

        # Load and preprocess audio
        audio = load_audio(file_path, cfg["sample_rate"])
        audio = normalize(audio)

        # Monitor memory before diarization
        mem_before = monitor_performance()

        # Speaker diarization
        segments = diarize(file_path)

        # Process each segment
        processed_segments = 0
        for i, seg in enumerate(segments):
            try:
                clip = extract_segment(audio, cfg["sample_rate"], seg["start"], seg["end"])

                # Skip segments that are too short
                if len(clip) / cfg["sample_rate"] < cfg["min_segment_duration"]:
                    continue

                # Classify gender using integrated classifier
                label, conf = classifier.classify(clip, cfg["sample_rate"])

                # Estimate pitch for metadata (legacy compatibility)
                from classification.pitch_gender import PitchGenderClassifier
                pitch_estimator = PitchGenderClassifier()
                pitch = pitch_estimator.estimate_pitch(clip, cfg["sample_rate"])

                name = f"{os.path.basename(file_path).replace('.wav', '')}_spk{i}_conf{int(conf)}.wav"
                saved_path = save_sample(clip, cfg["sample_rate"], label, name, "data/voice_dataset")

                # Assess audio quality
                quality_metrics = QualityMetrics(cfg["sample_rate"]).assess_audio_quality(saved_path)

                # Append metadata with quality metrics
                metadata_entry = {
                    "file": name,
                    "source": os.path.basename(file_path),
                    "speaker": seg["speaker"],
                    "pitch": pitch,
                    "label": label,
                    "confidence": conf,
                    "duration": len(clip)/cfg["sample_rate"],
                    "quality_score": quality_metrics["quality_score"],
                    "snr": quality_metrics["snr"],
                    "clipping_ratio": quality_metrics["clipping_ratio"],
                    "silence_ratio": quality_metrics["silence_ratio"]
                }

                append_metadata("data/voice_dataset/metadata.csv", metadata_entry)

                processed_segments += 1

            except Exception as e:
                logger.error(f"Error processing segment {i} of {file_path}: {e}")
                continue

        processing_time = time.time() - start_time
        mem_after = monitor_performance()

        logger.info(f"Successfully processed {file_path}: {processed_segments} segments in {processing_time:.2f}s")

        return {
            "file": os.path.basename(file_path),
            "segments_processed": processed_segments,
            "processing_time": processing_time,
            "memory_delta": mem_after["memory_used_gb"] - mem_before["memory_used_gb"]
        }

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return {
            "file": os.path.basename(file_path),
            "error": str(e),
            "processing_time": time.time() - start_time
        }

def run_parallel(files, cfg, max_workers=2):
    """Run processing in parallel."""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_file, file_path, cfg): file_path for file_path in files}

        for future in tqdm(as_completed(future_to_file), total=len(files), desc="Processing files"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Thread execution error: {e}")
                results.append({"error": str(e)})

    return results

def run(config_path):
    """Main processing function."""
    try:
        # Load and validate configuration
        cfg = yaml.safe_load(open(config_path))
        validate_config(cfg)

        # Set up directories
        input_dir = "data/input_calls"
        dataset_dir = "data/voice_dataset"

        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "male"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "female"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "uncertain"), exist_ok=True)

        # Get files to process
        files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".wav")]

        if not files:
            logger.warning("No WAV files found in input directory")
            return {"status": "no_files", "message": "No WAV files found"}

        logger.info(f"Starting batch processing of {len(files)} files")

        # Choose processing method based on file count
        if len(files) <= 5:
            # Sequential processing for small batches
            results = []
            for file_path in tqdm(files, desc="Processing files"):
                result = process_file(file_path, cfg)
                results.append(result)
        else:
            # Parallel processing for larger batches
            max_workers = min(cfg.get("parallel_workers", 2), len(files))
            results = run_parallel(files, cfg, max_workers)

        # Data augmentation if enabled
        if cfg.get("enable_augmentation", False):
            logger.info("Running data augmentation...")
            try:
                balance_dataset(dataset_dir, cfg)
                logger.info("Data augmentation completed")
            except Exception as e:
                logger.error(f"Data augmentation failed: {e}")

        # Summary
        successful = len([r for r in results if "error" not in r])
        failed = len([r for r in results if "error" in r])

        logger.info(f"Batch processing completed: {successful} successful, {failed} failed")

        return {
            "status": "completed",
            "total_files": len(files),
            "successful": successful,
            "failed": failed,
            "results": results
        }

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python batch_runner.py <config_path>")
        sys.exit(1)

    result = run(sys.argv[1])
    print(f"Processing result: {result}")
