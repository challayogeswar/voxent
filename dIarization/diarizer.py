import os
from pyannote.audio import Pipeline

_pipeline = None

def get_pipeline():
    """Get or create the diarization pipeline."""
    global _pipeline
    if _pipeline is None:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN environment variable is required for pyannote/speaker-diarization. Please set it with your Hugging Face token.")
        try:
            _pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=token)
        except Exception as e:
            raise RuntimeError(f"Failed to load diarization pipeline: {e}")
    return _pipeline

def diarize(audio_path):
    """Perform speaker diarization on audio file."""
    pipeline = get_pipeline()
    diarization = pipeline(audio_path)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    return segments
