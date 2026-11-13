import librosa
import pandas as pd
import numpy as np


def load_labels(file_path: str) -> pd.DataFrame:
    """
    Load labels from a CSV file and validate the data.

    Args:
        file_path (str): Path to the CSV file containing track IDs and labels.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded labels.
    """

    column_names = {"track", "start", "end", "label"}
    valid_labels = {"drop", "build", "breakdown"}

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    missing = column_names - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in the CSV file: {missing}")

    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")

    if df[["start", "end"]].isna().any().any():
        raise ValueError("Start and end must be numeric values.")

    if (df[["start", "end"]] < 0).any().any():
        raise ValueError("Start and end must be non-negative values.")

    if (df["start"] >= df["end"]).any():
        raise ValueError("Start time must be less than end time.")

    if not df["label"].isin(valid_labels).all():
        raise ValueError(f"Invalid labels found in the CSV file. Valid labels are: {valid_labels}")

    if df["track"].isna().any():
        raise ValueError("Every section must have a track ID.")

    if not np.isfinite(df[["start", "end"]]).to_numpy().all():
        raise ValueError("Start and end must be finite values.")

    return df

def extract_segment(audio, sr, start, end):
    """
    Extract a segment of audio based on start and end times.

    Args:
        audio (np.ndarray): The audio signal.
        sr (int): The sample rate.
        start (float): The start time of the segment.
        end (float): The end time of the segment.

    Returns:
        np.ndarray: The extracted audio segment.
    """
    sample_idx = int(start * sr)
    end_sample_idx = int(end * sr)
    
    if start < 0 or end < 0:
        raise ValueError("Start and end times must be non-negative.")

    if start >= end:
        raise ValueError("Start time must be less than end time.")

    if end > len(audio) / sr:
        raise ValueError("Segment exceeds audio duration.")
    
    segment = audio[sample_idx:end_sample_idx]
    
    return segment

def extract_features(segment, sr):
    """
    Extract features from an audio segment.

    Args:
        segment (np.ndarray): The audio segment.
        sr (int): The sample rate.

    Returns:
        dict: A dictionary containing extracted features.
    """

    if len(segment) == 0:
        raise ValueError("Audio segment cannot be empty.")

    rms = librosa.feature.rms(y=segment)[0]
    std_rms = np.std(rms)
    mean_rms = np.mean(rms)

    centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)[0]
    std_centroid = np.std(centroid)
    mean_centroid = np.mean(centroid)

    onset_strength = librosa.onset.onset_strength(y=segment, sr=sr)
    std_onset_strength = np.std(onset_strength)
    mean_onset_strength = np.mean(onset_strength)

    return {
        "std_rms": float(std_rms),
        "mean_rms": float(mean_rms),
        "std_centroid": float(std_centroid),
        "mean_centroid": float(mean_centroid),
        "std_onset_strength": float(std_onset_strength),
        "mean_onset_strength": float(mean_onset_strength)
    }

def build_dataset(labels_path: str, audio_dir: str) -> pd.DataFrame:
    """
    Build a machine learning dataset from labeled audio segments.
    """
    labels = load_labels(labels_path)
    train_data = []
    for track_id, track_labels in labels.groupby("track"):
        audio_path = f"{audio_dir}/{track_id}.wav"
        audio, sr = librosa.load(audio_path, sr=22050)
        for _, row in track_labels.iterrows():
            start_time = row["start"]
            end_time = row["end"]
            label = row["label"]

            segment = extract_segment(audio, sr, start_time, end_time)
            features = extract_features(segment, sr)

            train_data.append({
                "track": track_id,
                "start": start_time,
                "end": end_time,
                "label": label,
                **features
            })
    train_data = pd.DataFrame(train_data)
    return train_data

if __name__ == "__main__":
    labels = load_labels("testcsv.csv")
    print(labels)

    import numpy as np

    audio = np.arange(1000)
    sr = 100

    segment = extract_segment(audio, sr, 2, 5)

    print(segment[0])    # Expected: 200
    print(segment[-1])   # Expected: 499
    print(len(segment))  # Expected: 300

    df = build_dataset("testcsv.csv", "data/raw")

    print(df.head())