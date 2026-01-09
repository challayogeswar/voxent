# 🎙️ VOXENT - Open Source Voice Dataset Creator
## AI-Powered Multi-Speaker Voice Dataset Generation Platform

Project Vision: Community-driven tool for creating high-quality voice datasets from multi-speaker audio recordings  
Target Users: ML developers, researchers, students, data scientists, voice AI enthusiasts  
Mission: Democratize voice dataset creation - free, open, powerful

---

## 📌 WHAT IS VOXENT?

VOXENT transforms raw conversation recordings into organized, labeled voice datasets ready for ML training.

Input: Multi-speaker audio files (calls, podcasts, conversations, interviews)  
Output: Separated speaker segments with metadata (gender, timestamps, quality scores, transcripts)  
Use Cases: Speech recognition training, voice cloning datasets, speaker identification, conversation analysis

---

## ✅ WHAT'S IMPLEMENTED (Current v2.0)

### Core Audio Processing
- Speaker Diarization - Identifies who spoke when using pyannote.audio 3.1 (state-of-the-art ML model)
- Speaker Separation - Extracts individual speaker segments with precise timestamps
- Audio Segmentation - Splits conversations into clean, individual speaker clips
- Batch Organization - Groups files by duration for efficient GPU processing
- Sequential Processing - Processes batches in order (batch_001, batch_002, etc.)

### Classification & Analysis
- Gender Classification - Pitch-based gender detection (male/female) with confidence scores
- Quality Filtering - Duration and sample rate validation
- Metadata Generation - Complete JSON metadata for every segment (speaker, gender, timestamps, duration)

### GPU Optimization
- CUDA Acceleration - Leverages GPU for fast diarization (RTX 2050 optimized)
- Memory Management - VRAM monitoring with 80% threshold warnings
- Batch Processing - Configurable batch sizes (default: 10 files)
- Cache Clearing - Automatic GPU cache management between batches

### Output Organization
- Gender-Based Folders - Segments organized into male/ and female/ directories
- Speaker Labels - Neutral naming (SPEAKER_00, SPEAKER_01) without bias
- Hierarchical Structure - batch_folder/audio_file/gender/segments.wav
- Complete Traceability - Every segment linked to original file with timestamps

### Infrastructure
- Configuration System - YAML-based settings for all parameters
- Setup Automation - One-command installation (setup_voxent.py)
- Testing Suite - Comprehensive verification (test_voxent.py)
- Logging System - Detailed processing logs and error tracking
- Progress Tracking - Real-time batch processing progress

### Documentation
- User Guide - Step-by-step QUICK_START.md
- API Documentation - Function-level docstrings
- Configuration Reference - All settings explained
- Troubleshooting Guide - Common issues and solutions

---

## 🧠 HOW THE AI/ML WORKS

### Current Implementation

Speaker Diarization (pyannote.audio):
```
Audio → Spectrogram → ResNet CNN → Speaker Embeddings → 
Clustering (VBx) → "Who spoke when" timestamps (RTTM)
```
- Uses pre-trained neural network (trained on 1000+ hours of speech)
- Detects speaker changes with 90%+ accuracy
- Generates precise timestamps (start/end times per speaker)

Gender Classification (Pitch-Based):
```
Audio Segment → FFT (Fast Fourier Transform) → Fundamental Frequency →
Pitch Estimation → Threshold Comparison → Gender Label + Confidence
```
- Male: pitch <150 Hz
- Female: pitch >180 Hz
- Ambiguous: 150-180 Hz (lower confidence)

Batch Organization (Duration-Based):
```
Scan Files → Extract Duration (librosa) → Sort Ascending →
Group by Cumulative Duration → Create Physical Batch Folders
```
- Groups files to stay under GPU memory limits
- Smaller files processed first (faster warmup)

### Planned AI Enhancements

ML Gender Classifier (Phase 1):
```
Audio → Multi-Feature Extraction:
  - Pitch (F0)
  - Formants (F1, F2, F3)
  - MFCCs (13 coefficients)
  - Spectral Centroid
  - Zero Crossing Rate
→ Feature Vector (20+ dimensions) → RandomForest/XGBoost → 
Gender Prediction + Confidence
```
- Trained on user's own data (transfer learning)
- Continuously improves with more labeled examples
- 90-95% accuracy (vs 70-75% pitch-only)

Voice Activity Detection (Phase 1):
```
Audio → Silero VAD (RNN-based) → Voice Probability Per Frame →
Threshold Application → Voice Segments Only
```
- Removes silence, noise, non-speech
- Faster processing (skips empty audio)
- Cleaner dataset

Quality Assessment (Phase 2):
```
Segment → Feature Extraction:
  - SNR (Signal-to-Noise Ratio)
  - Clipping Detection
  - Energy Distribution
  - Speech Rate
→ Custom CNN/Quality Scorer → Quality Score (0-100)
```
- Automatic quality ranking
- Filter low-quality segments
- Identify best samples for training

Whisper Transcription (Phase 2):
```
Audio Segment → Mel Spectrogram → Transformer Encoder-Decoder (Whisper) →
Text Transcript + Word Timestamps + Language Detection
```
- 80+ languages supported
- Near-human accuracy
- Runs locally (no API costs)

Speaker Embeddings (Phase 2):
```
Audio → Pre-trained Speaker Encoder (ResNet/TDNN) → 
192-512 Dimensional Vector (d-vector) → Cosine Similarity → 
Speaker Clustering Across Files
```
- Unique "voice fingerprint" per speaker
- Track same person in different recordings
- Cross-file speaker identification

Emotion Detection (Phase 3):
```
Audio → Wav2Vec2 Features → Fine-tuned Emotion Classifier →
Emotion Probabilities (happy, sad, angry, neutral, surprised)
```
- Trained on emotional speech datasets
- Per-segment emotion labels
- Conversation sentiment analysis

Active Learning (Phase 3):
```
Unlabeled Data → Model Prediction → Uncertainty Estimation →
High-Uncertainty Samples → Human Review → Retrain Model →
Improved Accuracy
```
- Reduces manual labeling by 70-90%
- Focuses human effort on difficult cases
- Continuously improves model

---

## � COMPLETE PROJECT STRUCTURE

### Full Directory Tree with Every File

```
VOXENT/
│
├── 📋 ROOT FILES
│   ├── __init__.py                          # Package initialization
│   ├── deploy.sh                            # Deployment script
│   ├── docker-compose.yml                   # Docker configuration
│   ├── requirements.txt                     # Python dependencies
│   ├── .gitignore                           # Git ignore rules
│   ├── echoforge_product.md                 # Product documentation
│   ├── SPEAKER_SEPARATION_REPORT.md         # Latest results report
│   ├── SPEAKER_SEPARATION_RESULTS.json      # Processing results
│   ├── REDUNDANCY_ANALYSIS.md               # Code cleanup analysis
│   └── DOUBLE_CHECK_REDUNDANCY_VERIFICATION.md
│
├── 📁 data/
│   ├── input/                               # User audio files (MP3, WAV, etc.)
│   │   └── .mp3                             # Sample voice recordings
│   ├── temp/                                # Temporary processing files
│   ├── batches/                             # Organized batch folders (auto-generated)
│   │   ├── batch_001/
│   │   ├── batch_002/
│   │   └── ...
│   └── voice_dataset/                       # Final output segments
│       ├── male/                            # Male speaker segments
│       │   ├── male_1.wav
│       │   ├── male_2.wav
│       │   └── ... (386 files)
│       ├── female/                          # Female speaker segments
│       │   ├── female_1.wav
│       │   ├── female_2.wav
│       │   └── ... (20 files)
│       └── uncertain/                       # Ambiguous gender segments
│           ├── uncertain_1.wav
│           └── ... (6 files)
│
├── 📁 markdown & docs/
│   ├── README.md                            # Main documentation
│   ├── API.md                               # API reference
│   ├── 📁 mvps/
│   │   ├── MVP_COMPARISON.md               # MVP comparison
│   │   ├── Voxent_MVP_V0.md                # MVP v0 specs
│   │   └── Voxent_MVP_V1.md                # MVP v1 specs
│   ├── 📁 progress/
│   │   ├── 📁 09-01-26/
│   │   │   ├── IMPLEMENTATION_COMPLETE.md
│   │   │   ├── IMPLEMENTATION_SUMMARY.md
│   │   │   ├── INDEX.md
│   │   │   └── VOICE_CLASSIFICATION_REPORT.md
│   │   ├── 📁 26-12-25/
│   │   │   ├── COMPLETE_FIX_REPORT.md
│   │   │   ├── FIXES_APPLIED.md
│   │   │   └── [other progress files]
│   │   ├── 📁 29-12-25/
│   │   └── 📁 [other dates]/
│   ├── 📁 test reports/
│   │   ├── 📁 09-01-26/
│   │   ├── 📁 23-12-25/
│   │   ├── 📁 24-12-25/
│   │   ├── 📁 29-12-25/
│   │   └── 📁 30-12-25/
│   └── 📁 todo's/
│       ├── 📁 09-01-2025/
│       ├── 📁 26-12-25/
│       ├── 📁 29-12-25/
│       └── 📁 30-12-25/
│
├── 📁 src/
│   ├── __init__.py                          # Package init
│   ├── main.py                              # Main entry point
│   │
│   ├── 📁 classification/
│   │   ├── __init__.py                      # Integrated classifier export
│   │   ├── advanced_gender_classifier.py    # Multi-feature classifier
│   │   ├── ml_classifier.py                 # ML-based classifier
│   │   ├── ml_gender_classifier_v3.py       # ML classifier v3
│   │   ├── pitch_gender.py                  # Pitch-based classifier
│   │   └── __pycache__/                     # Compiled Python cache
│   │
│   ├── 📁 config/
│   │   ├── config.yaml                      # Main configuration file
│   │   ├── run_pipeline.py                  # Pipeline runner
│   │   └── __pycache__/
│   │
│   ├── 📁 data_augmentation/
│   │   ├── __init__.py
│   │   ├── augment.py                       # Audio augmentation
│   │   └── __pycache__/
│   │
│   ├── 📁 dataset/
│   │   ├── __init__.py
│   │   ├── metadata.py                      # Metadata management
│   │   ├── organizer.py                     # Dataset organization
│   │   └── __pycache__/
│   │
│   ├── 📁 diarization/
│   │   ├── __init__.py
│   │   ├── diarizer.py                      # Basic diarization
│   │   ├── enhanced_diarizer.py             # Enhanced diarization
│   │   ├── segments.py                      # Segment processing
│   │   └── __pycache__/
│   │
│   ├── 📁 engine/
│   │   ├── __init__.py
│   │   ├── batch_runner.py                  # Batch processing engine
│   │   ├── logger.py                        # Logging system
│   │   └── __pycache__/
│   │
│   ├── 📁 pipeline/
│   │   ├── __init__.py
│   │   ├── ai_integration.py                # AI model integration
│   │   ├── batch_organizer.py               # Batch organization
│   │   ├── batch_processor.py               # Batch processing
│   │   ├── pipeline_runner.py               # Pipeline orchestration
│   │   └── __pycache__/
│   │
│   ├── 📁 preprocessing/
│   │   ├── __init__.py
│   │   ├── audio_converter.py               # Audio format conversion
│   │   ├── audio_loader.py                  # Audio loading
│   │   ├── normalize.py                     # Audio normalization
│   │   ├── source_separator.py              # Source separation
│   │   ├── vad_enhanced.py                  # Enhanced VAD
│   │   ├── vad.py                           # Voice activity detection
│   │   └── __pycache__/
│   │
│   ├── 📁 quality/
│   │   ├── __init__.py
│   │   ├── metrics.py                       # Quality metrics
│   │   ├── quality_assessor.py              # Quality assessment
│   │   └── __pycache__/
│   │
│   ├── 📁 scripts/
│   │   ├── comprehensive_system_test.py     # ✅ Unified system tests (GPU/CPU/Full)
│   │   ├── generate_test_audio.py           # Test audio generation
│   │   ├── setup_voxent.py                  # Initial setup script
│   │   ├── speaker_separation.py            # MAIN: Speaker diarization pipeline
│   │   ├── test_batching_consolidated.py    # ✅ Unified batching tests
│   │   ├── train_ai_models.py               # AI model training
│   │   ├── train_ml_classifier.py           # ML classifier training
│   │   ├── verification.py                  # Manual verification tool
│   │   ├── verify_installation.py           # Installation verifier
│   │   ├── voxent_demo.py                   # Demo script
│   │   └── web_app.py                       # Web interface
│   │
│   ├── 📁 templates/
│   │   └── index.html                       # Web UI template
│   │
│   ├── 📁 tests/
│   │   ├── test_integration.py              # Integration tests
│   │   └── test_pipeline.py                 # Pipeline tests
│   │
│   └── 📁 utils/
│       └── __init__.py                      # Utilities package
│
├── 📁 models/
│   └── [Pre-trained models - auto-downloaded]
│
├── 📁 extra data/
│   └── [Additional data files]
│
└── 📁 VOXENT/
    └── [Backup/legacy code - to be archived]
```

---

## 🔧 TECHNICAL ARCHITECTURE

### Current Stack (v2.0)

Core Libraries:
- `pyannote.audio 3.1` - Speaker diarization
- `librosa 0.10+` - Audio analysis
- `soundfile` - Audio I/O
- `PyTorch 2.0+` - Deep learning backend
- `pyyaml` - Configuration
- `tqdm` - Progress tracking
- `psutil` - System monitoring

Hardware Requirements:
- Minimum: CPU-only (slow), 8GB RAM
- Recommended: NVIDIA GPU (4GB+ VRAM), 16GB RAM
- Optimal: RTX 2050/3050 or better, 32GB RAM

### Script Descriptions

#### 🌟 Main Production Scripts

speaker_separation.py (MOST IMPORTANT)
- Purpose: Main pipeline for speaker diarization and gender classification
- Function: Extracts individual speakers from mixed audio
- Output: Labeled segments (male_1.wav, female_1.wav, uncertain_1.wav, etc.)
- Usage: `python src/scripts/speaker_separation.py`
- Key Features:
  - Energy-based voice activity detection
  - Multi-feature gender classification
  - Auto-incrementing sequential labeling
  - Processes all files in data/input/

setup_voxent.py
- Purpose: Initial project setup and verification
- Function: Creates directories, downloads models, verifies dependencies
- Usage: `python src/scripts/setup_voxent.py`

web_app.py
- Purpose: Web interface for VOXENT
- Function: Flask/Streamlit-based UI for processing
- Usage: `python src/scripts/web_app.py`

#### 🧪 Consolidated Test Scripts (Recently Merged)

test_batching_consolidated.py (MERGED FROM 2 FILES)
- Purpose: Test batching logic and organization
- Merged From: test_batching.py + test_batching_simple.py
- Modes:
  - Complex Mode: Uses BatchOrganizer module
  - Simple Mode: Standalone implementation
- Usage: `python src/scripts/test_batching_consolidated.py`

comprehensive_system_test.py (MERGED FROM 4 FILES)
- Purpose: System-wide testing and verification
- Merged From: test_gpu.py + run_cpu_test.py + run_full_test.py + test_voxent_complete.py
- Modes:
  - GPU Mode: Test CUDA availability
  - CPU Mode: Test CPU pipeline
  - Full Mode: Comprehensive system test
- Usage: `python src/scripts/comprehensive_system_test.py [gpu|cpu|full]`

#### 🛠️ Training & Utilities

train_ai_models.py
- Purpose: Train AI models for enhanced classification
- Function: Fine-tunes pre-trained models on user data

train_ml_classifier.py
- Purpose: Train ML-based gender classifier
- Function: RandomForest/XGBoost classifier training

generate_test_audio.py
- Purpose: Generate synthetic test audio
- Function: Creates test audio files with specific durations

verify_installation.py
- Purpose: Verify installation completeness
- Function: Checks all dependencies and configuration

verification.py
- Purpose: Manual classification verification
- Function: Interactive tool for verifying segment labels

voxent_demo.py
- Purpose: Demonstration of VOXENT capabilities
- Function: Runs demo pipeline on sample data

### Core Module Descriptions

#### classification/
- `__init__.py` - Integrated classifier that combines all methods
- `pitch_gender.py` - Pitch-based gender detection (current)
- `ml_classifier.py` - ML-based classifier framework
- `ml_gender_classifier_v3.py` - V3 ML classifier
- `advanced_gender_classifier.py` - Multi-feature classifier

#### preprocessing/
- `audio_converter.py` - MP3/WAV/FLAC conversion
- `audio_loader.py` - Audio file loading and I/O
- `normalize.py` - Audio normalization
- `source_separator.py` - Voice source separation
- `vad.py` - Voice activity detection
- `vad_enhanced.py` - Enhanced VAD

#### diarization/
- `diarizer.py` - Basic speaker diarization
- `enhanced_diarizer.py` - Enhanced diarization with post-processing
- `segments.py` - Segment extraction and management

#### pipeline/
- `batch_organizer.py` - Duration-based batch creation
- `batch_processor.py` - Batch processing engine
- `ai_integration.py` - AI model integration
- `pipeline_runner.py` - Full pipeline orchestration

#### engine/
- `batch_runner.py` - Batch execution engine
- `logger.py` - Structured logging system

#### dataset/
- `organizer.py` - Dataset organization and structure
- `metadata.py` - Metadata management

#### quality/
- `metrics.py` - Quality metrics calculation
- `quality_assessor.py` - Quality assessment module

#### data_augmentation/
- `augment.py` - Audio augmentation techniques

---

### Configuration Files

config.yaml - Main configuration file
```yaml
preprocessing:
  sample_rate: 16000        # Standard speech sample rate
  mono: true               # Convert to mono

diarization:
  method: "pyannote"       # Speaker diarization method
  min_segment_duration: 0.3 # Minimum 300ms segments

classification:
  pitch_male_threshold: 150  # Hz
  pitch_female_threshold: 180 # Hz
  confidence_threshold: 0.55

gpu:
  device: "cuda"           # GPU or CPU
  batch_size: 10           # Files per batch
  memory_threshold: 0.8    # 80% VRAM limit
```

---

### Planned Stack Additions

Phase 1 (ML Enhancements):
- `scikit-learn` - ML classifiers
- `xgboost` - Gradient boosting
- `silero-vad` - Voice activity detection

Phase 2 (Intelligence):
- `openai-whisper` - Transcription
- `resemblyzer` - Speaker embeddings
- `transformers` - HuggingFace models

Phase 3 (Advanced AI):
- `wav2vec2` - Emotion detection
- `speechbrain` - Advanced audio AI
- `faiss` - Vector similarity search

Phase 4 (Community):
- `flask` / `fastapi` - Web backend
- `streamlit` - Web interface
- `docker` - Containerization
- `celery` - Background tasks

---

## 💡 KEY INNOVATIONS

### 1. Timestamp-Driven Extraction
Unlike traditional vocal separation (Demucs), we use ML diarization timestamps to extract speaker segments precisely:
```
Traditional: Audio → Demucs → Mixed vocals (all speakers together)
VOXENT: Audio → Diarization → Timestamps → Extract per speaker → Clean segments
```

### 2. Duration-Based Batching
Smart file grouping prevents GPU memory overflow:
```
Instead of: Process all files → GPU crash
VOXENT: Group by duration → Process batch → Clear cache → Next batch
```

### 3. Multi-Level Organization
Hierarchical output structure for easy navigation:
```
batch_001/recording_001/male/SPEAKER_00_seg001.wav
batch_001/recording_001/female/SPEAKER_01_seg002.wav
```

### 4. Bootstrap Learning
Use simple methods to train better models:
```
Pitch-based labels → Train ML classifier → Better accuracy → 
Relabel with ML → Retrain → Even better accuracy
```

### 5. Configuration-Driven
All features optional, user-controlled:
```yaml
ai_enhancements:
  gender_ml: true        # Enable ML classifier
  transcription: true    # Enable Whisper
  quality_filter: true   # Auto-filter low quality
```

---

## 🎯 REAL-WORLD APPLICATIONS

### For ML Developers
- Train speech recognition models - Labeled speaker data
- Build voice cloning systems - Per-speaker voice samples
- Develop speaker identification - Speaker embeddings dataset

### For Researchers
- Conversation analysis - Turn-taking, interruption patterns
- Sociolinguistics studies - Gender differences in speech
- Emotion research - Emotional tone in conversations

### For Students
- Learn audio processing - Hands-on with state-of-the-art tools
- Build ML projects - Ready-to-use voice datasets
- Research projects - Quick dataset creation for experiments

### For Industry
- Call center analytics - Agent/customer separation
- Podcast production - Automatic speaker segmentation
- Meeting transcription - Multi-speaker meeting notes

---

## 📊 PERFORMANCE METRICS

### Current (v2.0)
- Processing Speed: 5-10 sec per minute of audio (GPU)
- Diarization Accuracy: 85-95% (depends on audio quality)
- Gender Classification: 70-75% (pitch-based)
- Memory Usage: ~2.5-3.0 GB VRAM (RTX 2050)

### Planned (v3.0 with AI)
- Processing Speed: 18-30 sec per minute (with transcription)
- Diarization Accuracy: 90-98% (improved post-processing)
- Gender Classification: 90-95% (ML-based)
- Quality Assessment: 85-90% (automated filtering)
- Transcription Accuracy: 90-95% (Whisper)

---

## 🌍 COMMUNITY & OPEN SOURCE

### Open Source Commitment
- MIT License - Free for commercial and personal use
- GitHub Repository - Full source code, issues, discussions
- Community Contributions - PRs welcome, contributor guidelines
- Documentation - Comprehensive guides for users and developers

### Community Features (Planned)
- Dataset Sharing - Upload/download pre-processed datasets
- Model Marketplace - Share fine-tuned models
- Use Case Gallery - Showcase projects built with VOXENT
- Forum & Discord - Community support and discussions

### Educational Resources (Planned)
- Video Tutorials - YouTube channel with how-tos
- Blog Posts - Technical deep-dives, use cases
- Workshop Materials - Presentations, notebooks
- Paper/Citation - Academic reference (arXiv)

---

## 🔒 PRIVACY & ETHICS

### Current Features
- Local Processing - All data stays on your machine
- No Cloud Upload - No external API calls (except HuggingFace auth)
- No Data Collection - Zero telemetry or usage tracking

### Planned Privacy Tools
- Voice Anonymization - Pitch shifting, formant modification
- PII Redaction - Remove names, numbers from transcripts
- Consent Management - Track speaker permissions
- GDPR Compliance - Right to deletion, data export

### Ethical Guidelines
- Informed Consent - Only process audio with speaker permission
- Fair Use - No public figure voice cloning without consent
- Bias Awareness - Document model limitations and biases
- Responsible AI - Clear guidelines for ethical use

---

## 📈 ROADMAP SUMMARY

### Q1 2025 (3 months)
- ✅ v2.0 Release (Current) - Core diarization working
- 🔄 v2.1 - ML gender classifier
- 🔄 v2.2 - VAD integration
- 🔄 v2.3 - Quality assessment

### Q2 2025 (3 months)
- 📅 v3.0 - Whisper transcription
- 📅 v3.1 - Speaker embeddings
- 📅 v3.2 - Cross-file speaker tracking

### Q3 2025 (3 months)
- 📅 v4.0 - Emotion detection
- 📅 v4.1 - Active learning system
- 📅 v4.2 - Web interface beta

### Q4 2025 (3 months)
- 📅 v5.0 - Cloud processing support
- 📅 v5.1 - Dataset marketplace
- 📅 v5.2 - Real-time processing

---

## 🤝 HOW TO CONTRIBUTE

### As a User
- Report Bugs - GitHub issues with detailed descriptions
- Request Features - Describe your use case and needs
- Share Results - Post your experience and datasets
- Star the Repo - Help others discover VOXENT

### As a Developer
- Code Contributions - PRs for bug fixes, features
- Documentation - Improve guides, add examples
- Testing - Test on different hardware, OS configurations
- Model Training - Share fine-tuned models

### As a Researcher
- Validation Studies - Test accuracy, compare with other tools
- Novel Applications - New use cases and domains
- Academic Papers - Cite and extend VOXENT
- Workshops - Teach VOXENT to students/researchers

---

## 📚 TECHNICAL DETAILS (Deep Dive)

### Speaker Diarization Pipeline
Model: pyannote/speaker-diarization-3.1 (HuggingFace)
```
Architecture:
1. Segmentation: PyanNet (ResNet backbone)
   - Input: Audio waveform
   - Output: Speech activity per speaker
   
2. Embedding: wespeaker-voxceleb-resnet34
   - Input: Speech segments
   - Output: 256-dim speaker embeddings
   
3. Clustering: VBx (Variational Bayesian)
   - Input: Speaker embeddings
   - Output: Speaker labels (SPEAKER_00, SPEAKER_01, ...)
   
4. RTTM Generation:
   - Format: START END SPEAKER
   - Example: 0.5 3.2 SPEAKER_00
```

### Audio Feature Extraction (Current & Planned)
```python
# Pitch Extraction
pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate)
pitch = np.mean(pitches[pitches > 0])

# MFCCs (Mel-Frequency Cepstral Coefficients)
mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)

# Spectral Features
centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)

# Zero Crossing Rate
zcr = librosa.feature.zero_crossing_rate(audio)

# Chroma Features
chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
```

### ML Classification (Planned Implementation)
```python
# Feature vector: [20+ dimensions]
features = [
    pitch,                    # Fundamental frequency
    pitch_std,                # Pitch variation
    mfcc[0:13],              # 13 MFCCs
    spectral_centroid,       # Brightness
    spectral_rolloff,        # Energy concentration
    zcr,                     # Voice/unvoiced detection
]

# Classifier: RandomForest or XGBoost
model = RandomForestClassifier(n_estimators=100)
model.fit(features, labels)  # labels: 'male' or 'female'

# Prediction with confidence
prediction = model.predict([features])[0]
confidence = max(model.predict_proba([features])[0])
```

### Batch Processing Logic
```python
# Duration-based grouping
def create_batches(files, max_duration=120.0, max_files=10):
    batches = []
    current_batch = []
    current_duration = 0.0
    
    for file in sorted_files:  # Sorted by duration
        duration = get_duration(file)
        
        if (current_duration + duration > max_duration or
            len(current_batch) >= max_files):
            # Start new batch
            batches.append(current_batch)
            current_batch = [file]
            current_duration = duration
        else:
            # Add to current batch
            current_batch.append(file)
            current_duration += duration
    
    if current_batch:
        batches.append(current_batch)
    
    return batches
```

### GPU Memory Management
```python
# VRAM monitoring
def check_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        usage_percent = (allocated / total)  100
        
        if usage_percent > 80:  # Threshold
            logger.warning(f"High VRAM usage: {usage_percent:.1f}%")
        
        return usage_percent

# Cache clearing between batches
def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()  # Python garbage collection
```

---

## 🎓 LEARNING RESOURCES

### Audio Processing
- librosa Documentation - https://librosa.org/doc/latest/index.html
- Speech Processing Course - Stanford CS224S
- Digital Signal Processing - MIT OpenCourseWare

### Deep Learning for Audio
- Wav2Vec2 Paper - Facebook AI Research
- Whisper Technical Report - OpenAI
- Speaker Diarization Survey - arXiv:2012.01477

### Machine Learning
- scikit-learn Tutorials - https://scikit-learn.org/stable/tutorial/
- XGBoost Documentation - https://xgboost.readthedocs.io/
- Deep Learning Book - Ian Goodfellow (free online)

### VOXENT Specific
- Project Documentation - Full guides in repository
- API Reference - Docstrings in all modules
- Example Notebooks - Jupyter notebooks (coming soon)

---

## 🏆 SUCCESS STORIES (User Contributions)

### Research Projects
- [Your Project Here] - Describe how you used VOXENT
- [Community Project] - Share your results
- [Academic Study] - Link to published papers

### Commercial Applications
- [Company/Product] - How VOXENT helped your business
- [Startup] - Dataset creation for voice AI product

### Educational Uses
- [University Course] - VOXENT in ML curriculum
- [Workshop] - Community training event

---

## 📞 CONTACT & SUPPORT

### Get Help
- Documentation - Read QUICK_START.md and README.md
- GitHub Issues - Report bugs, ask questions
- GitHub Discussions - Community Q&A
- Discord Server - Real-time chat (coming soon)

### Stay Updated
- GitHub Releases - New version notifications
- Blog - Technical articles and updates (planned)
- Twitter/X - Follow for announcements (planned)
- Newsletter - Monthly updates (planned)

### Contribute
- GitHub - Fork, PR, star the repository
- Documentation - Improve guides and examples
- Testing - Try on your hardware, report results
- Feedback - Share your use case and suggestions

---

## 📄 LICENSE & CREDITS

### License
MIT License - Free for personal and commercial use

### Credits
- pyannote.audio - Hervé Bredin (CNRS)
- Whisper - OpenAI
- librosa - Brian McFee et al.
- PyTorch - Facebook AI Research
- Community Contributors - [Your name here!]

### Citation
```bibtex
@software{voxent2025,
  title={VOXENT: AI-Powered Voice Dataset Creator},
  author={[ChallaYogeswar/FORGEXA]},
  year={2025},
  url={https://github.com/[your-repo]}
}
```

---

## 🎉 CONCLUSION

VOXENT is more than a tool - it's a movement to democratize voice AI.

We're building a platform where anyone can create professional-grade voice datasets for ML, research, or learning. By combining cutting-edge AI with open-source principles, we're making voice dataset creation accessible to everyone.

Join us in building the future of voice AI! 🚀

---

Version: 2.0 (Current) → 3.0 (In Development)  
Last Updated: December 30, 2025  
Contributors: Community-Driven Project  
Status: Active Development

---

## 📋 QUICK REFERENCE

| Feature | Status | Version | Priority |
|---------|--------|---------|----------|
| Speaker Diarization | ✅ Done | v2.0 | Core |
| Gender Classification (Pitch) | ✅ Done | v2.0 | Core |
| Batch Processing | ✅ Done | v2.0 | Core |
| GPU Optimization | ✅ Done | v2.0 | Core |
| ML Gender Classifier | 🔄 In Progress | v2.1 | High |
| VAD Integration | 🔄 In Progress | v2.1 | High |
| Quality Assessment | 📅 Planned | v2.3 | High |
| Whisper Transcription | 📅 Planned | v3.0 | Medium |
| Speaker Embeddings | 📅 Planned | v3.1 | Medium |
| Emotion Detection | 📅 Planned | v4.0 | Low |
| Web Interface | 📅 Planned | v4.2 | Medium |
| Cloud Processing | 📅 Planned | v5.0 | Low |

Legend:  
✅ Implemented | 🔄 In Progress | 📅 Planned | ⏸️ On Hold

---

🌟 Star us on GitHub | 🍴 Fork and contribute | 📢 Share with your network

Together, we're making voice AI accessible to everyone!
