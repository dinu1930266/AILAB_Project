import matplotlib.pyplot as plt
import numpy as np
import librosa
import os
import random
from sklearn.ensemble import RandomForestClassifier


# def get_mfcc(audio_recording, genre):
#     y, sr = librosa.load(audio_recording)
#     # qui sono contenute le feature dello spectogramma
#     S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
#     librosa.feature.mfcc(S=librosa.power_to_db(S))
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
#     fig, ax = plt.subplots(nrows=2, sharex=True)
#     img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),  x_axis='time', y_axis='mel', fmax=8000, ax=ax[0])
#     fig.colorbar(img, ax=[ax[0]])
#     ax[0].set(title=f"Mel spectrogram, {audio_recording}")
#     ax[0].label_outer()
#     img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1])
#     #fig.colorbar(img, ax=[ax[1]])
#     ax[1].set(title=f"MFCC, {audio_recording}")
#     audio_recording_save = audio_recording[:-4]

#     split_audio_recording_save = audio_recording_save.split('/')
#     split_audio_recording_save_canzone = split_audio_recording_save[-1]

#     print(split_audio_recording_save_canzone)

#     plt.savefig(f'Data/images_original/{genre}/{split_audio_recording_save_canzone}.png', format='png', dpi=1200, bbox_inches='tight')
#     #plt.show()
#     plt.close()

genres = ['rock', 'pop', 'classical', 'hiphop', 'jazz', 'metal', 'reggae', 'disco', 'country', 'blues']

# for genre in genres:
#     filename = f"Data/genres_original/{genre}/"
#     for file in os.listdir(filename):
#         if file[0]!= '.':
#             file_tot = filename+file
#             get_mfcc(file_tot, genre)



# Creare un array di feature vuoto
X = np.empty((0, 140))

# Creare un array di etichette vuoto
y = np.empty((0))

# Loop attraverso i file audio
for genre in genres:
    data_dir = f"Data/genres_original/{genre}"
    genre_dir = os.path.join(data_dir, genre)
    for filename in os.listdir(data_dir):
        print(filename)
        file_path = os.path.join(data_dir, filename)
        y, sr = librosa.load(file_path)
        
        # target_length = 10 * sr  # lunghezza desiderata in campioni
        # y = librosa.util.fix_length(y, target_length) 

        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        # show chroma in a plot
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
        plt.colorbar()
        plt.title('Chromagram')
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'Data/images_original/{genre}/{filename}.png', format='png', dpi=1200, bbox_inches='tight')

        
        features = np.concatenate((mel_spectrogram, chroma), axis=0).T
        # v.stack funzione che permettere di concatenare 2 array
        X = np.vstack((X, features))
        print(X.shape)
        y = np.append(y, genre)
        
# Dividere il set di dati in set di addestramento e set di test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creare un classificatore RandomForest
rf = RandomForestClassifier(n_estimators=130, random_state=42)

# Addestrare il modello sul set di addestramento
rf.fit(X_train, y_train)

# Valutare l'accuratezza del modello sul set di test
accuracy = rf.score(X_test, y_test)
print("Accuracy:", accuracy)
