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

    if not np.isinfinite(df[["start", "end"]]).to_numpy().all():
        raise ValueError("Start and end must be finite values.")

    return df

def extract_segment(audio, sr, start, end):
    # calc starting sample index
    
    # calc ending sample index

    # check time stams are valid

    # return sliced audio
    
    pass

if __name__ == "__main__":
    labels = load_labels("testcsv.csv")
    print(labels)