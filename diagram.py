
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

""" Caricare il file audio utilizzando la funzione load() di Librosa: """
#filename = librosa.example('nutcracker')

## load an audio file from local directory
filename = 'genres/pop/pop.00006.au'
signal, sr = librosa.load(filename)

""" Calcolare lo spettrogramma dell'audio utilizzando la funzione stft() di Librosa: """
stft = librosa.stft(signal)

""" Calcolare lo spettro di potenza dell'audio utilizzando la funzione power_to_db() di Librosa: """
spectrogram = librosa.power_to_db(abs(stft)**2, ref=np.max)
## change the scale of the spectrogram


""" Visualizzare lo spettro di frequenza dell'audio utilizzando la funzione specshow() di Librosa e 
la funzione show() di matplotlib:
"""

librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='hz')
#plt.colorbar()
plt.title('Spettro di frequenza')
plt.show()

"""
In questo modo, verrà visualizzato il grafico dello spettro di frequenza dell'audio caricato,
dove sull'asse X si rappresenta il tempo e sull'asse Y la frequenza in Hz.
"""