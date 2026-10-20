import librosa
import numpy as np
import matplotlib.pyplot as plt

# print("getting here")
audio, sr = librosa.load("test.wav") # gone for a minute tisoki

print("Sample rate:", sr)
print("Number of samples:", len(audio))
print("Duration:", len(audio) / sr)
print("Mean amplitude:", np.mean(audio))



librosa.display.waveshow(audio, sr=sr)
plt.title("Track Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)

print("Estimated BPM:", float(np.asarray(tempo).item()))
print("Detected beats:", len(beat_frames))

plt.show()