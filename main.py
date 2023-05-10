import tensorflow as tf
import numpy as np
import scipy
from scipy import misc
import glob
from PIL import Image
import os
import matplotlib.pyplot as plt
import librosa
from keras import layers
from keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, 
                          Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout)
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
import pydot
#from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pydub import AudioSegment
import shutil
from keras.preprocessing.image import ImageDataGenerator
import random


def get_melspectrogram(audio_recording, genre):
    y, sr = librosa.load(audio_recording, duration=3)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    plt.plot()
    img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel', fmax=8000)
    plt.title(f"Mel spectrogram, {audio_recording}")
    audio_recording_save = audio_recording[:-4]
    split_audio_recording_save = audio_recording_save.split('/')
    split_audio_recording_save_canzone = split_audio_recording_save[-1]
    print(split_audio_recording_save_canzone)
    plt.savefig(f'content/spectrograms3sec/train/{genre}/{split_audio_recording_save_canzone}.png', format='png', dpi=1200, bbox_inches='tight')
    plt.close()

# definiamo i generi dei brani musicali.
genres = ["blues", "classical", "country", "disco", "pop", "hiphop", "metal", "reggae", "rock"]

# os.listdir() mostra tutte le cartelle del path in cui ci troviamo.
#print('listdir:', os.listdir())

# Perciò se la cartella 'content' non è presente nel path creiamo le varie directory:
if 'content' not in os.listdir():
    os.makedirs('content/spectrograms3sec')
    os.makedirs('content/spectrograms3sec/train')
    os.makedirs('content/spectrograms3sec/test')

    # creiamo le cartelle spectogram3sec/train, /test, e audio3sec con all'interno le cartelle con i vari generi musicali 
    for g in genres:
        ".join() utilizzato per unire gli elementi di una sequenza di stringhe in una sola stringa utilizzando un delimitatore specificato"
        path_audio = os.path.join('content/audio3sec',f'{g}')
        os.makedirs(path_audio)
        path_train = os.path.join('content/spectrograms3sec/train',f'{g}')
        path_test = os.path.join('content/spectrograms3sec/test',f'{g}')
        os. makedirs(path_train)
        os. makedirs(path_test)

# qui splittiamo gli audio in 3 parti (ogni brano originale dura 30 secondi, perciò ogni brano splittato avrà una durata di 10 secondi) 
# e li inseriamo nella cartella audio3sec
i = 0
for g in genres:
    j=0
    print(f"{g}")
    for filename in os.listdir(os.path.join('Data/genres_original',f"{g}")):
        song  =  os.path.join(f'Data/genres_original/{g}',f'{filename}')
        if filename[0]!= '.':
            j = j+1
            for w in range(0,3):
                i = i+1
                #print(i)
                t1 = 10*(w)*1000
                t2 = 10*(w+1)*1000
                newAudio = AudioSegment.from_wav(song)
                new = newAudio[t1:t2]
                new.export(f'content/audio3sec/{g}/{g+str(j)+str(w)}.wav', format="wav")

# per ogni brano da 3 secondi di ogni genere generiamo lo spettogramma e lo salviamo in content/spectrograms3sec/train
for genre in genres:
    filename = f"content/audio3sec/{genre}/"
    for file in os.listdir(filename):
        if file[0]!= '.':
            file_tot = filename+file
            get_melspectrogram(file_tot, genre)


# Ora abbiamo i nostri dati completi, quindi dobbiamo dividere i dati in set di addestramento e set di convalida. 
# I nostri dati completi sono nella directory spectrograms3sec/train, quindi dobbiamo prendere parte dei dati completi e 
# spostarli nella nostra directory di test.
directory = "/content/spectrograms3sec/train/"
for g in genres:
    filenames = os.listdir(os.path.join(directory,f"{g}"))
    random.shuffle(filenames)
    test_files = filenames[0:100]

    # Per ogni genere, mescoliamo casualmente i nomi dei file, selezioniamo i primi 100 nomi di file e 
    # li spostiamo nella directory di test/convalida.
    for f in test_files:
        shutil.move(directory + f"{g}"+ "/" + f,"/content/spectrograms3sec/test/" + f"{g}")


# Creeremo generatori di dati sia per la formazione che per il set di test.
train_dir = "/content/spectrograms3sec/train/"
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(288,432),color_mode="rgba",class_mode='categorical',batch_size=128)

validation_dir = "/content/spectrograms3sec/test/"
vali_datagen = ImageDataGenerator(rescale=1./255)
vali_generator = vali_datagen.flow_from_directory(validation_dir,target_size=(288,432),color_mode='rgba',class_mode='categorical',batch_size=128)

"""
Il metodo flow_from_directory() deferisce automaticamente le etichette utilizzando la nostra struttura di directory e 
le codifica di conseguenza.

ImageDataGenerator semplifica l'addestramento su grandi set di dati utilizzando il fatto che durante l'addestramento il modello viene addestrato 
su un solo lotto per passaggio, quindi, durante l'addestramento, il generatore di dati carica solo un lotto nella memoria alla volta, 
quindi non c'è esaurimento delle risorse di memoria
"""

"""
In questa funzione vengono applicati i vari strati di neuroni della rete neurale.
"""
def GenreModel(input_shape = (288,432,4),classes=9):
  
  X_input = Input(input_shape)

  X = Conv2D(8,kernel_size=(3,3),strides=(1,1))(X_input)
  X = BatchNormalization(axis=3)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2))(X)
  
  X = Conv2D(16,kernel_size=(3,3),strides = (1,1))(X)
  X = BatchNormalization(axis=3)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2))(X)
  
  X = Conv2D(32,kernel_size=(3,3),strides = (1,1))(X)
  X = BatchNormalization(axis=3)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2))(X)

  X = Conv2D(64,kernel_size=(3,3),strides=(1,1))(X)
  X = BatchNormalization(axis=-1)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2))(X)
  
  X = Conv2D(128,kernel_size=(3,3),strides=(1,1))(X)
  X = BatchNormalization(axis=-1)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2))(X)

  
  X = Flatten()(X)
  
  X = Dropout(rate=0.3)

  X = Dense(classes, activation='softmax', name='fc' + str(classes))(X)

  model = Model(inputs=X_input,outputs=X,name='GenreModel')

  return model


import keras.backend as K
def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
  

model = GenreModel(input_shape=(288,432,4),classes=9)
opt = Adam(learning_rate=0.0005)
model.compile(optimizer = opt,loss='categorical_crossentropy',metrics=['accuracy',get_f1]) 

model.fit_generator(train_generator,epochs=70,validation_data=vali_generator)
