import librosa
import numpy as np
import matplotlib.pyplot as plt

audio, sr = librosa.load("test.wav") # gone for a minute tisoki

print("Sample rate:", sr)
print("Number of samples:", len(audio))
print("Duration:", len(audio) / sr)
print("Mean amplitude:", np.mean(audio))



# librosa.display.waveshow(audio, sr=sr)
# plt.title("Track Waveform")
# plt.xlabel("Time (seconds)")
# plt.ylabel("Amplitude")

tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)

print("Estimated BPM:", float(np.asarray(tempo).item()))
print("Detected beats:", len(beat_frames))

# CALCULATE RMS AMPLITUDE

rms = librosa.feature.rms(y=audio)[0] # RMS amplitude (root mean square energy)

times = librosa.times_like(rms, sr=sr) # get time stamps according to each RMS value

# plt.figure(figsize=(12, 4))
# plt.plot(times, rms)

# plt.title("RMS Energy Over Time")
# plt.xlabel("Time (seconds)")
# plt.ylabel("RMS Amplitude")

# CALCULATE SPECTRAL CENTROID
# spectral centroid gives the weighted average of frequences along the
# selected time frames 

centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]

# get corresponding timestamps
centroid_times = librosa.times_like(centroid, sr=sr)

# plot rms and spectral centroid together
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

axes[0].plot(times, rms)
axes[0].set_title("RMS Amplitude")
axes[0].set_ylabel("RMS")

axes[1].plot(centroid_times, centroid)
axes[1].set_title("Spectral Centroid")
axes[1].set_ylabel("Frequency (Hz)")
axes[1].set_xlabel("Time (seconds)")

plt.tight_layout()
plt.show()