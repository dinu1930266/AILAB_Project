import matplotlib.pyplot as plt
import numpy as np
import librosa
import os


def get_mfcc(audio_recording, genre):
    y, sr = librosa.load(audio_recording)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    librosa.feature.mfcc(S=librosa.power_to_db(S))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    fig, ax = plt.subplots(nrows=2, sharex=True)
    img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),  x_axis='time', y_axis='mel', fmax=8000, ax=ax[0])
    fig.colorbar(img, ax=[ax[0]])
    ax[0].set(title=f"Mel spectrogram, {audio_recording}")
    ax[0].label_outer()
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1])
    #fig.colorbar(img, ax=[ax[1]])
    ax[1].set(title=f"MFCC, {audio_recording}")
    audio_recording_save = audio_recording[:-4]

    split_audio_recording_save = audio_recording_save.split('/')
    split_audio_recording_save_canzone = split_audio_recording_save[-1]

    print(split_audio_recording_save_canzone)

    plt.savefig(f'Data/Image/{genre}/{split_audio_recording_save_canzone}.png', format='png', dpi=1200, bbox_inches='tight')
    #plt.show()
    plt.close()

genres = ['rock', 'blues', 'classical', 'country', 'disco', 'pop', 'raggae']

for genre in genres:
    filename = f"Data/genres_original/{genre}/"
    for file in os.listdir(filename):
        if file[0]!= '.':
            file_tot = filename+file
            get_mfcc(file_tot, genre)
