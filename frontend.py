import tkinter as tk
from tkinter import filedialog
#import pyaudio
import glob
import IPython
import keras.backend as K
import librosa
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydot
import random
import scipy
from keras import layers
from keras.initializers import glorot_uniform
from keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout)
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import layer_utils, plot_model
from keras.utils.vis_utils import model_to_dot
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from numpy import asarray
from PIL import Image
from pydub import AudioSegment
import os
import shutil
import sys
import tensorflow as tf
import IPython.display as ipd # to play the Audio Files
from scipy import misc
from keras.utils import layer_utils
#from IPython.display import SVG
from keras.utils import plot_model
#from keras.preprocessing.image import img_to_array
from keras.models import load_model
import librosa.display
from matplotlib.colors import Normalize


def crea_cartelle_necessarie(genres):
    # os.listdir() mostra tutte le cartelle del path in cui ci troviamo.
    #print('listdir:', os.listdir())

    # Perciò se la cartella 'content' non è presente nel path creiamo le varie directory:
    if 'content_10sec' not in os.listdir():

        # creiamo le cartelle spectogram3sec/train, /test, e audio3sec con all'interno le cartelle con i vari generi musicali 
        for g in genres:
            ".join() utilizzato per unire gli elementi di una sequenza di stringhe in una sola stringa utilizzando un delimitatore specificato"
            path_audio = os.path.join('content_10sec/audio10sec/',f'{g}')
            os.makedirs(path_audio)


def split_audio(genres):
    # qui splittiamo gli audio in 3 parti (ogni brano originale dura 30 secondi, perciò ogni brano splittato avrà una durata di 10 secondi) 
    # e li inseriamo nella cartella audio3sec
    i = 0
    for g in genres:
        j=0
        print(f"{g}")
        for filename in os.listdir(os.path.join('Data/genres_original',f"{g}")):
            print(filename)
            #print(j)
            song  =  os.path.join(f'Data/genres_original/{g}',f'{filename}')
            if (filename[0]!= '.'):
                for w in range(0,3):
                    i = i+1
                    #print(i)
                    t1 = 10*(w)*1000
                    t2 = 10*(w+1)*1000
                    newAudio = AudioSegment.from_wav(song)
                    new = newAudio[t1:t2]
                    new.export(f'content_10sec/audio10sec/{g}/{g}.{j}.{w}.wav', format="wav")

                j = j+1


genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]


crea_cartelle_necessarie(genres)


#split_audio(genres)



def get_features(path_audio, nome_audio, genre):
    # Load the audio file
    audio_recording = nome_audio
    y, sr = librosa.load(path_audio, sr=None)

    # Extract features
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = chroma_stft.mean()
    chroma_stft_var = chroma_stft.var()

    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean()
    rms_var = rms.var()

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = spectral_centroid.mean()
    spectral_centroid_var = spectral_centroid.var()

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth_mean = spectral_bandwidth.mean()
    spectral_bandwidth_var = spectral_bandwidth.var()

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = rolloff.mean()
    rolloff_var = rolloff.var()

    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate_mean = zero_crossing_rate.mean()
    zero_crossing_rate_var = zero_crossing_rate.var()

    harmony = librosa.effects.harmonic(y)
    harmony_mean = harmony.mean()
    harmony_var = harmony.var()

    perceptr = librosa.effects.percussive(y)
    perceptr_mean = perceptr.mean()
    perceptr_var = perceptr.var()

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_var = mfcc.var(axis=1)

    mfcc1_mean = mfcc[0].mean()
    mfcc1_var = mfcc[0].var()
    mfcc2_mean = mfcc[1].mean()
    mfcc2_var = mfcc[1].var()
    mfcc3_mean = mfcc[2].mean()
    mfcc3_var = mfcc[2].var()
    mfcc4_mean = mfcc[3].mean()
    mfcc4_var = mfcc[3].var()
    mfcc5_mean = mfcc[4].mean()
    mfcc5_var = mfcc[4].var()
    mfcc6_mean = mfcc[5].mean()
    mfcc6_var = mfcc[5].var()
    mfcc7_mean = mfcc[6].mean()
    mfcc7_var = mfcc[6].var()
    mfcc8_mean = mfcc[7].mean()
    mfcc8_var = mfcc[7].var()
    mfcc9_mean = mfcc[8].mean()
    mfcc9_var = mfcc[8].var()
    mfcc10_mean = mfcc[9].mean()
    mfcc10_var = mfcc[9].var()
    mfcc11_mean = mfcc[10].mean()
    mfcc11_var = mfcc[10].var()
    mfcc12_mean = mfcc[11].mean()
    mfcc12_var = mfcc[11].var()
    mfcc13_mean = mfcc[12].mean()
    mfcc13_var = mfcc[12].var()
    mfcc14_mean = mfcc[13].mean()
    mfcc14_var = mfcc[13].var()
    mfcc15_mean = mfcc[14].mean()
    mfcc15_var = mfcc[14].var()
    mfcc16_mean = mfcc[15].mean()
    mfcc16_var = mfcc[15].var()
    mfcc17_mean = mfcc[16].mean()
    mfcc17_var = mfcc[16].var()
    mfcc18_mean = mfcc[16].mean()
    mfcc18_var = mfcc[17].var()
    mfcc19_mean = mfcc[17].mean()
    mfcc19_var = mfcc[18].var()
    mfcc20_mean = mfcc[19].mean()
    mfcc20_var = mfcc[19].var()


    # Prepare the extracted features
    length = librosa.get_duration(y=y, sr=sr)
    
    features = [
        nome_audio, length, chroma_stft_mean, chroma_stft_var, rms_mean, rms_var,
        spectral_centroid_mean, spectral_centroid_var, spectral_bandwidth_mean, spectral_bandwidth_var,
        rolloff_mean, rolloff_var, zero_crossing_rate_mean, zero_crossing_rate_var,
        harmony_mean, harmony_var, perceptr_mean, perceptr_var, tempo,
        mfcc1_mean, mfcc1_var,mfcc2_mean,mfcc2_var,mfcc3_mean,mfcc3_var,mfcc4_mean,mfcc4_var,mfcc5_mean,mfcc5_var,mfcc6_mean,
        mfcc6_var,mfcc7_mean,mfcc7_var,mfcc8_mean,mfcc8_var,mfcc9_mean,mfcc9_var,mfcc10_mean,mfcc10_var,mfcc11_mean,
        mfcc11_var,mfcc12_mean,mfcc12_var,mfcc13_mean,mfcc13_var,mfcc14_mean,mfcc14_var,mfcc15_mean,mfcc15_var,mfcc16_mean,
        mfcc16_var,mfcc17_mean,mfcc17_var,mfcc18_mean,mfcc18_var,mfcc19_mean,mfcc19_var,mfcc20_mean,mfcc20_var, genre]

    # print(len(features))
    # print(f'{nome_audio}: {features}\n\n')
    return features


# csv_array =[]
# csv_array.append(['filename','length','chroma_stft_mean','chroma_stft_var','rms_mean','rms_var','spectral_centroid_mean',
#                   'spectral_centroid_var','spectral_bandwidth_mean','spectral_bandwidth_var','rolloff_mean','rolloff_var',
#              'zero_crossing_rate_mean','zero_crossing_rate_var','harmony_mean','harmony_var','perceptr_mean','perceptr_var',
#              'tempo','mfcc1_mean','mfcc1_var','mfcc2_mean','mfcc2_var','mfcc3_mean','mfcc3_var','mfcc4_mean','mfcc4_var','mfcc5_mean',
#              'mfcc5_var','mfcc6_mean','mfcc6_var','mfcc7_mean','mfcc7_var','mfcc8_mean','mfcc8_var','mfcc9_mean','mfcc9_var','mfcc10_mean',
#              'mfcc10_var','mfcc11_mean','mfcc11_var','mfcc12_mean','mfcc12_var','mfcc13_mean','mfcc13_var','mfcc14_mean','mfcc14_var',
#              'mfcc15_mean','mfcc15_var','mfcc16_mean','mfcc16_var','mfcc17_mean','mfcc17_var','mfcc18_mean','mfcc18_var','mfcc19_mean',
#              'mfcc19_var','mfcc20_mean','mfcc20_var','label'])


# for genre in genres:
#     for audio in os.listdir(f'content_10sec/audio10sec/{genre}'):
#         print(audio)
#         path_audio = f'content_10sec/audio10sec/{genre}/{audio}'
#         feature = get_features(path_audio, audio, genre)
#         csv_array.append(feature)

# import csv

# with open('Data/features_10_sec.csv', 'w', newline='') as file_csv:
#     writer = csv.writer(file_csv)

#     # Scrivi i dati nel file CSV
#     writer.writerows(csv_array)


def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfcss_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcss_scaled_features = np.mean(mfcss_features.T, axis=0)
    return mfcss_scaled_features

audio_data_set_path = 'content_10sec/audio10sec/'
metadata = pd.read_csv('Data/features_10_sec.csv')
metadata.head()

metadata.drop(labels=552, axis=0, inplace=True)

from tqdm import tqdm

extracted_features = []
for index_num, row in tqdm(metadata.iterrows()):
    try:
        # final_class_label indica il genere
        final_class_label = row["label"]

        # path_to_audio_file indica il percorso al file audio
        file_name = os.path.join(os.path.abspath(audio_data_set_path), final_class_label+'/', str(row["filename"]))
        data = features_extractor (file_name)
        extracted_features.append([data, final_class_label])
    except Exception as e:
        print(f'Error: {e}')
        continue

extracted_features_df = pd.DataFrame(extracted_features, columns=['feature', 'class'])
extracted_features_df.head()

extracted_features_df['class'].value_counts()

X = np.array(extracted_features_df['feature'].tolist())
y = np.array(extracted_features_df['class'].tolist())

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

y = to_categorical(labelencoder.fit_transform(y))
print(y.shape)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics

### No of classes
num_labels = y.shape[1]
model = Sequential()
model.add(Dense(1024,input_shape=(40,),activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(512,activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(256,activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(32,activation="relu"))
model.add(Dropout(0.3))


model.add(Dense(num_labels, activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

def addestra_modello():
    import time
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)

    from tensorflow.keras.callbacks import ModelCheckpoint
    from datetime import datetime

    num_batch_size = 32

    #checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification_{current_time}.hdf5', verbose = 1, save_best_only= True)

    start = datetime.now()
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 500, batch_size = num_batch_size)#, callbacks=[checkpointer], verbose=1)

    duration = datetime.now()- start

    print(duration)

    model.evaluate(X_test, y_test, verbose=0)

    pd.DataFrame(history.history).plot(figsize=(12,6))
    plt.show()

    #model.predict_classes(X_test)
    model.predict(X_test)
    model.save('my_model.h500-10sec')
    return model

def carica_modello():
    model = tf.keras.models.load_model('my_model.h500-10sec')
    return model


#model = addestra_modello()
model = carica_modello()




def select_audio_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", ".wav")])
    audio_file_path.set(file_path)
    #return file_path
    

def process_audio():
    # Get the selected audio file path
    file_path = audio_file_path.get()
    if file_path:
        # TODO: Add your audio processing logic here
        print("Processing audio file:", file_path)
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfcss_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfcss_scaled_features = np.mean(mfcss_features.T, axis=0)

        print(mfcss_scaled_features)
        mfcss_scaled_features = mfcss_scaled_features.reshape(1, -1)
        print(mfcss_scaled_features)
        print(mfcss_scaled_features.shape)
        predicted_label = model.predict(mfcss_scaled_features)
        print(predicted_label)

        class_probabilities = predicted_label[0]
        class_indices = range(len(genres))
        plt.bar(class_indices, class_probabilities, align='center')
        plt.xticks(class_indices, genres, rotation=45)
        plt.xlabel('Genere')
        plt.ylabel('Probabilità')
        plt.title('Prediction')
        plt.show()

        predicted_genre = genres[np.argmax(predicted_label)]  # Ottiene il genere previsto utilizzando l'indice di valore massimo
        print("Predicted Genre:", predicted_genre)  # Stampa il genere previsto
        # prediction_class = labelencoder.inverse_transform(predicted_label)
        # print(prediction_class)


#filename = 'Data/genres_original/jazz/jazz.00090.wav'
#filename = select_audio_file()
#filename = 'brani_test/nirvana.wav'


# Create the main window
window = tk.Tk()
window.title("Audio File Processor")

# Create a label for the audio file path
audio_file_path = tk.StringVar()
file_path_label = tk.Label(window, textvariable=audio_file_path, width=50, wraplength=400)
file_path_label.pack(pady=10)

# Create a button to select the audio file
select_file_button = tk.Button(window, text="Select Audio File", command=select_audio_file)
select_file_button.pack(pady=10)

# Create a button to process the audio file
process_button = tk.Button(window, text="Process Audio", command=process_audio)
process_button.pack(pady=10)

# Set window background color
window.configure(bg='#F5F5F5')

# Set window size and center it on the screen
window_width = 500
window_height = 200
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)
window.geometry(f'{window_width}x{window_height}+{x}+{y}')

# Start the GUI event loop
window.mainloop()






