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

    segment = audio[sample_idx:end_sample_idx]
    
    return segment

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