
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
import wave
audio_path = librosa.util.example_audio_file()
y, sr = librosa.load(audio_path)
# Let's make and display a mel-scaled power (energy-squared) spectrogram
S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

# Convert to log scale (dB). We'll use the peak power (max) as reference.
log_S = librosa.power_to_db(S, ref=np.max)

mfcc        = librosa.feature.mfcc(S=log_S, n_mfcc=13)
delta_mfcc  = librosa.feature.delta(mfcc)
delta2_mfcc = librosa.feature.delta(mfcc, order=2)

print(mfcc.shape)