import matplotlib.pyplot as plt
import numpy as np
import librosa

y, sr = librosa.load(librosa.ex('libri1'))
# librosa.feature.mfcc(y=y, sr=sr)
# librosa.feature.mfcc(y=y, sr=sr, hop_length=1024, htk=True)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

librosa.feature.mfcc(S=librosa.power_to_db(S))
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

fig, ax = plt.subplots(nrows=2, sharex=True)
img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),  x_axis='time', y_axis='mel', fmax=8000, ax=ax[0])
fig.colorbar(img, ax=[ax[0]])
ax[0].set(title='Mel spectrogram')
ax[0].label_outer()
#plt.show()
img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1])
#fig.colorbar(img, ax=[ax[1]])
ax[1].set(title='MFCC')
plt.show()
