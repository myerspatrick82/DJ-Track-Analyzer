from pathlib import Path

import librosa
import numpy as np

def analyze_track(file_path: str) -> dict:
    audio, sr = librosa.load(file_path, sr=22050)

    if len(audio) == 0:
        raise ValueError("Audio File is Empty")

    rms = librosa.feature.rms(y=audio)[0]

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]

    onset_strength = librosa.onset.onset_strength(y=audio, sr=sr)

    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
    
    return {
        "filename": Path(file_path).name,
        "duration": round(len(audio) / sr, 2),
        "bpm": round(float(np.asarray(tempo).item()), 2),
        "average_rms": round(float(np.mean(rms)), 4),
        "average_centroid": round(float(np.mean(centroid)), 2),
        "average_onset_strength": round(float(np.mean(onset_strength)), 4),
        "beat_count": len(beats)
    }