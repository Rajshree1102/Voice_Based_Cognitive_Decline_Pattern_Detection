# cognitive_model.py

import librosa
import numpy as np
import whisper
import re
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pandas as pd
import joblib

SAMPLE_RATE = 16000
model = whisper.load_model("base")

def transcribe_audio(file_path):
    result = model.transcribe(file_path)
    return result['text']

def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    duration = librosa.get_duration(y=y, sr=sr)

    text = transcribe_audio(file_path)
    words = text.split()
    word_count = len(words)
    speech_rate = word_count / duration if duration > 0 else 0

    hesitations = re.findall(r"\b(uh+|um+)\b", text.lower())
    hesitation_count = len(hesitations)

    intervals = librosa.effects.split(y, top_db=30)
    pause_durations = [(intervals[i][0] - intervals[i-1][1]) / sr
                       for i in range(1, len(intervals))]
    pause_count = len(pause_durations)
    avg_pause_duration = np.mean(pause_durations) if pause_durations else 0

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
    pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    target_words = ["apple", "banana", "chair", "car"]
    missing_words = [word for word in target_words if word not in text.lower()]
    word_recall_errors = len(missing_words)

    features = {
        "speech_rate": speech_rate,
        "hesitation_count": hesitation_count,
        "pause_count": pause_count,
        "avg_pause_duration": avg_pause_duration,
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "word_recall_errors": word_recall_errors,
        **{f"mfcc_{i+1}": val for i, val in enumerate(mfccs_mean)}
    }

    return features

scaler = joblib.load("scaler.joblib")
anomaly_model = joblib.load("isolation_forest_model.joblib")

def detect_cognitive_decline(file_path):
    feats = extract_audio_features(file_path)
    text = transcribe_audio(file_path)
    feats_df = pd.DataFrame([feats])
    X_scaled = scaler.transform(feats_df)
    risk_score = anomaly_model.predict(X_scaled)[0] 
    return {
        "risk_score": float(risk_score),
        "features": feats,
        "audio_to_text": text
    }
