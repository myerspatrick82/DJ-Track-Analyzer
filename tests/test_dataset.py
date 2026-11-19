from dataset import extract_segment
import numpy as np
import pytest


def test_extract_segment():
    audio = np.arange(1000)
    sr = 100

    segment = extract_segment(audio, sr, 2, 5)

    assert len(segment) == 300
    assert segment[0] == 200
    assert segment[-1] == 499

def test_start_greater_than_end():
    audio = np.arange(1000)

    with pytest.raises(ValueError):
        extract_segment(audio, 100, 5, 2)

def test_start_negative():
    audio = np.arange(1000)

    with pytest.raises(ValueError):
        extract_segment(audio, 100, -1, 2)

def test_end_greater_than_duration():
    audio = np.arange(1000)

    with pytest.raises(ValueError):
        extract_segment(audio, 100, 2, 15)

def test_start_equals_end():
    audio = np.arange(1000)

    with pytest.raises(ValueError):
        extract_segment(audio, 100, 2, 2)