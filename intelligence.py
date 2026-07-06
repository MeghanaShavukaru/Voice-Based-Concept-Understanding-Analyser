import os
import ssl
import urllib.request
import whisper
import librosa
import numpy as np
import matplotlib.pyplot as plt
import re
from difflib import SequenceMatcher

# Some environments reject the certificate chain for Hugging Face model downloads.
# Disable certificate verification for this specific runtime so Whisper can fetch its model.
if not getattr(urllib.request, "_smartbridge_ssl_patched", False):
    _original_urlopen = urllib.request.urlopen

    def _ssl_compatible_urlopen(url, *args, **kwargs):
        if isinstance(url, str) and url.startswith("https://"):
            kwargs.setdefault("context", ssl._create_unverified_context())
        return _original_urlopen(url, *args, **kwargs)

    urllib.request.urlopen = _ssl_compatible_urlopen
    urllib.request._smartbridge_ssl_patched = True

try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer = None
    util = None

# Load models lazily to avoid failing the app startup when downloads are blocked.
stt_model = None
semantic_model = None


def _load_models():
    global stt_model, semantic_model

    if stt_model is None:
        try:
            stt_model = whisper.load_model("base")
        except Exception as exc:
            raise RuntimeError(f"Unable to load speech model: {exc}") from exc

    if semantic_model is None and SentenceTransformer is not None:
        try:
            semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            semantic_model = None


def transcribe_audio(file_path):
    """Speech-to-Text Module"""
    try:
        _load_models()
        result = stt_model.transcribe(file_path)
        return result.get("text", "").strip() or "No speech detected."
    except Exception as exc:
        return f"Transcription unavailable. {exc}"


def analyze_semantics(transcript, reference_text):
    """Semantic Understanding Module"""
    if semantic_model is not None and util is not None:
        try:
            _load_models()
            embeddings1 = semantic_model.encode(transcript, convert_to_tensor=True)
            embeddings2 = semantic_model.encode(reference_text, convert_to_tensor=True)
            cosine_scores = util.cos_sim(embeddings1, embeddings2)
            return cosine_scores.item() * 100
        except Exception:
            pass

    # Fallback heuristic when embedding models are unavailable.
    ref_words = set(re.findall(r"\w+", reference_text.lower()))
    trans_words = set(re.findall(r"\w+", transcript.lower()))
    overlap = len(ref_words & trans_words)
    total = max(len(ref_words), 1)
    token_score = (overlap / total) * 100
    sequence_score = SequenceMatcher(None, transcript.lower(), reference_text.lower()).ratio() * 100
    return round(max(token_score, sequence_score * 0.8), 2)

def analyze_audio_features(file_path, output_plot_path):
    """Audio Feature Extraction Module (Pause, RMS, Waveform)"""
    y, sr = librosa.load(file_path, sr=None)
    
    # RMS Energy
    rms = librosa.feature.rms(y=y)
    mean_rms = np.mean(rms)
    
    # Pause Ratio (Silence evaluation)
    non_mute_intervals = librosa.effects.split(y, top_db=20)
    non_mute_duration = np.sum([interval[1] - interval[0] for interval in non_mute_intervals]) / sr
    total_duration = librosa.get_duration(y=y, sr=sr)
    pause_ratio = ((total_duration - non_mute_duration) / total_duration) * 100 if total_duration > 0 else 0
    
    # Save Waveform Visualization
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    plt.title("Audio Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.close()
    
    return mean_rms, pause_ratio

def analyze_filler_words(transcript):
    """Filler Word Analysis Module"""
    filler_words = ['um', 'uh', 'like', 'you know', 'so', 'actually']
    words = transcript.lower().split()
    total_words = len(words)
    
    if total_words == 0:
        return 0, 0
        
    filler_count = sum(1 for word in words if re.sub(r'[^\w\s]', '', word) in filler_words)
    filler_ratio = (filler_count / total_words) * 100
    
    return filler_count, filler_ratio

def generate_score_and_feedback(semantic_score, pause_ratio, filler_ratio):
    """Evaluation Scoring Module"""
    # Weighted scoring logic (Customizable)
    final_score = (semantic_score * 0.7) + ((100 - pause_ratio) * 0.15) + ((100 - filler_ratio) * 0.15)
    
    if final_score >= 80:
        feedback = "Strong Understanding: Excellent grasp of the concept and confident delivery."
    elif final_score >= 50:
        feedback = "Moderate Understanding: Good attempt, but some key points or fluency could be improved."
    else:
        feedback = "Poor Understanding: Concept meaning was missed or delivery was highly hesitant."
        
    return final_score, feedback