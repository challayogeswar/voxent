## 🚀 WHAT WE'RE ADDING (Planned Enhancements)

### Phase 1: Enhanced ML Classification (1-2 weeks)
- Multi-Feature Gender Classifier - RandomForest/XGBoost using pitch, formants, MFCCs, spectral features (90%+ accuracy)
- Confidence Thresholding - Adjustable confidence levels for classification
- Model Training Pipeline - Bootstrap from existing labels, continuous improvement
- Voice Activity Detection (VAD) - Silero VAD for cleaner segments, faster processing

### Phase 2: Intelligent Analysis (2-4 weeks)
- Quality Assessment System - AI-powered quality scoring (audio SNR, speech clarity, segment completeness)
- Automatic Transcription - Whisper integration for speech-to-text on all segments
- Speaker Embeddings - Deep learning voice fingerprints (192-512 dimensions)
- Cross-File Speaker Tracking - Identify same speaker across multiple recordings
- Topic Extraction - Keyword and topic modeling from transcripts

### Phase 3: Advanced AI Features (1-3 months)
- Emotion Detection - Recognize emotional tone (happy, sad, angry, neutral, surprised)
- Speaking Style Analysis - Detect speaking rate, energy, pauses, interruptions
- Active Learning System - Semi-automated labeling with confidence-based review
- Conversation Dynamics - Turn-taking patterns, interruption frequency, dominance metrics
- Multi-Language Support - Extend beyond English (Spanish, French, Hindi, etc.)

### Phase 4: Community & Scale (3-6 months)
- Web Interface - Browser-based upload and processing (Flask/FastAPI backend)
- Cloud Processing - Optional cloud GPU support (AWS/Google Cloud integration)
- Dataset Marketplace - Share/discover datasets (with privacy controls)
- Model Zoo - Pre-trained models for different languages and domains
- Collaboration Tools - Multi-user projects, version control for datasets

### Phase 5: Production Features (6+ months)
- Real-Time Processing - Live conversation separation (streaming audio)
- API Service - RESTful API for programmatic access
- Plugin System - Custom processors, exporters, analyzers
- Export Formats - HuggingFace datasets, CSV, Kaldi, Common Voice format
- Privacy Tools - Voice anonymization, PII redaction, consent management

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

File Structure:
```
voxent/
├── data/
│   ├── input_calls/          # Input audio files
│   ├── batches/              # Organized batch folders
│   └── voice_dataset/        # Final output
├── logs/                     # Processing logs
├── models/                   # AI models (future)
├── voxent_pipeline.py        # Main orchestrator
├── enhanced_diarizer.py      # Speaker separation
├── batch_organizer.py        # Batch creation
├── batch_processor.py        # GPU processing
├── config.yaml               # Configuration
└── [documentation files]
```

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