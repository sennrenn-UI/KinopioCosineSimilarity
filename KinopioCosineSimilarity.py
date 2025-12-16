import librosa
import numpy as np

y1, sr1 = librosa.load("kinopio_ref.m4a", sr=None, mono=True)
y2, sr2 = librosa.load("kinopio_imit.m4a", sr=None, mono=True)

sr = sr1

a1 = np.abs(np.fft.fft(y1))[:len(y1)//2]
a2 = np.abs(np.fft.fft(y2))[:len(y2)//2]

a1 /= np.max(a1)
a2 /= np.max(a2)

min_len = min(len(a1), len(a2))
a1, a2 = a1[:min_len], a2[:min_len]

cos_sim = np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2))
print(cos_sim)
