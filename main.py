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
#from keras.preprocessing.image import img_to_array
import random
import keras.backend as K
from keras.models import load_model
import numpy as np
from keras import layers
from keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, 
                          Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,
                          Dropout)
from keras.models import Model, load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from keras.initializers import glorot_uniform
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import librosa
import numpy as np
import librosa.display
from pydub import AudioSegment
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from PIL import Image
from numpy import asarray


" utilizzato per ottenere lo spettogramma dei vari brani e salvarlo nel path specificato"
def get_and_save_melspectrogram(audio_recording, genre):
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

"In questa funzione vengono applicati i vari strati di neuroni della rete neurale"
def GenreModel(input_shape = (288,432,4),classes=9):
  
  X_input = Input(input_shape)

  X = Conv2D(8,kernel_size=(3,3),strides=(1,1))(X_input)
  """
  La normalizzazione del batch (Batch Normalization) è una tecnica di regolarizzazione molto utilizzata nelle reti neurali profonde 
  per migliorare la velocità di convergenza del modello durante l'addestramento e la sua capacità di generalizzazione.
  """
  X = BatchNormalization(axis=3)(X)
  
  """
  La funzione di attivazione ReLU (Rectified Linear Unit) è una funzione matematica che viene comunemente utilizzata nelle reti neurali artificiali 
  come funzione di attivazione per introdurre la non linearità nei dati in ingresso. La funzione ReLU restituisce un valore di 0 per 
  tutti i valori di input negativi e restituisce il valore di input stesso per tutti i valori positivi. 
  """
  X = Activation('relu')(X)

  """
  L'operazione di MaxPooling è un'operazione comune nella fase di estrazione delle feature nelle reti neurali convoluzionali (CNN). 
  L'obiettivo del MaxPooling è quello di ridurre la dimensionalità dei dati di input, mantenendo le informazioni più rilevanti.
  """
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

  "il layer di Flatten svolge il ruolo di appiattire l'input in modo che possa essere utilizzato come input per una serie di neuroni completamente connessi."
  X = Flatten()(X)
  
  """
    Un layer di output Dense è un tipo di strato completamente connesso (o fully connected layer) presente in una rete neurale artificiale, 
    che ha come obiettivo quello di produrre un output in cui ogni neurone dell'ultimo strato di attivazione è collegato a tutti i neuroni dell'output.
  """
  X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=9))(X)

  model = Model(inputs=X_input,outputs=X,name='GenreModel')

  return model

def crea_cartelle_necessarie(genres):
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

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def addestra_modello():
    # Creeremo generatori di dati sia per la formazione che per il set di test.
    train_dir = "content/spectrograms3sec/train/"
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(train_dir,target_size=(288,432),color_mode="rgba",class_mode='categorical',batch_size=128)

    validation_dir = "content/spectrograms3sec/test/"
    vali_datagen = ImageDataGenerator(rescale=1./255)
    vali_generator = vali_datagen.flow_from_directory(validation_dir,target_size=(288,432),color_mode='rgba',class_mode='categorical',batch_size=128)

    """
    Il metodo flow_from_directory() deferisce automaticamente le etichette utilizzando la nostra struttura di directory e 
    le codifica di conseguenza.

    ImageDataGenerator semplifica l'addestramento su grandi set di dati utilizzando il fatto che durante l'addestramento il modello viene addestrato 
    su un solo lotto per passaggio, quindi, durante l'addestramento, il generatore di dati carica solo un lotto nella memoria alla volta, 
    quindi non c'è esaurimento delle risorse di memoria
    """
    model = GenreModel(input_shape=(288,432,4),classes=9)
    opt = Adam(learning_rate=0.0005)
    model.compile(optimizer = opt,loss='categorical_crossentropy',metrics=['accuracy',get_f1]) 

    #model.fit_generator(train_generator,epochs=70,validation_data=vali_generator)

    history = model.fit(train_generator, epochs=50, validation_data=vali_generator)

    accuracy = history.history['val_accuracy'][-1]
    print(accuracy)
    model.save('my_model.h50')
    return model


# definiamo i generi dei brani musicali.
genres = ["blues", "classical", "country", "disco", "pop", "hiphop", "metal", "reggae", "rock"]

crea_cartelle_necessarie(genres)

# qui splittiamo gli audio in 3 parti (ogni brano originale dura 30 secondi, perciò ogni brano splittato avrà una durata di 10 secondi) 
# e li inseriamo nella cartella audio3sec
if not os.listdir('content/audio3sec/blues'):
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
if not os.listdir('content/spectrograms3sec/train/blues'):
    for genre in genres:
        filename = f"content/audio3sec/{genre}/"
        for file in os.listdir(filename):
            if file[0]!= '.':
                file_tot = filename+file
                get_and_save_melspectrogram(file_tot, genre)


# Ora abbiamo i nostri dati completi, quindi dobbiamo dividere i dati in set di addestramento e set di convalida. 
# I nostri dati completi sono nella directory spectrograms3sec/train, quindi dobbiamo prendere parte dei dati completi e 
# spostarli nella nostra directory di test. 
directory = "content/spectrograms3sec/train/"
if not os.listdir('content/spectrograms3sec/test/blues'):
    for g in genres:
        filenames = os.listdir(os.path.join(directory,f"{g}"))
        random.shuffle(filenames)
        test_files = filenames[0:100]

        # Per ogni genere, mescoliamo casualmente i nomi dei file, selezioniamo i primi 100 nomi di file e 
        # li spostiamo nella directory di test/convalida.
        for f in test_files:
            shutil.move(directory + f"{g}"+ "/" + f,"content/spectrograms3sec/test/" + f"{g}")

# Dopo l'esecuzione di queste righe di codice nella cartella train rimarranno 200 brani per ogni genere,
# mentre nella cartella test ci saranno 100 brani per ogni genere.

#model = addestra_modello()

custom_objects = {'get_f1': get_f1}

# caricamento del modello già addestrato e salvato come 'my_model.h5'
model = tf.keras.models.load_model('my_model.h50', custom_objects=custom_objects)

audio_recording = 'maroon5.wav'
start_time = 10 # start time in seconds (1 minute and 38 seconds)
duration = 300 # duration in seconds

# Load the audio recording
y, sr = librosa.load(audio_recording, sr=None)

# Calculate the start and end frame indices for the desired segment
start_frame = int(start_time * sr)
end_frame = int((start_time + duration) * sr)

# Extract the desired segment of audio
audio_segment = y[start_frame:end_frame]

# Pad or trim the audio segment to the desired length
audio_segment = librosa.util.fix_length(audio_segment, size=221000)

# Compute the mel spectrogram
S = librosa.feature.melspectrogram(y=audio_segment, sr=sr, n_mels=288, fmax=8000)
#S_dB = librosa.power_to_db(S, ref=np.max)
#S_dB = np.expand_dims(S_dB, axis=-1)
#S_dB = np.expand_dims(S_dB, axis=0)

# """ValueError: Input 0 of layer "GenreModel" is incompatible with the layer: expected shape=(None, 288, 432, 4), found shape=(None, 288, 317, 1)
# come risolvo l'errore
# RISOLTO
# """
# #np.reshape(image,(1,288,432,4))
#S_dB = np.reshape((1,288,432,4))

tf.reshape(S,(None, 288,432, 4))

# image = Image.open(S)


#image = np.reshape(S_dB,(1,288,432,4))

prediction = model.predict(S)

prediction = prediction.reshape((9,)) 


class_label = np.argmax(prediction)


# Pad the input with zeros to match the expected shape of the model
#S_dB_padded = np.zeros((S_dB.shape[0], 288, 432, 4))
# print(S_dB_padded)

"""
ValueError: Input 0 of layer "GenreModel" is incompatible with the layer: expected shape=(None, 288, 432, 4), found shape=(32, 317)
"""

# Make the prediction using the padded input
# prediction = model.predict(S)
# predicted_class = np.argmax(prediction)
# """
# Il numero che viene passato alla funzione get_and_save_melspectrogram corrisponde alla classe di appartenenza del brano, 
# ovvero il genere musicale. Nel codice, viene utilizzato come indice per salvare lo spettrogramma nella directory corrispondente alla classe del brano. 
# Il numero 0 indica ad esempio il genere "blues", il numero 1 il genere "classica", e così via.
# """
print('Il genere predetto è:', class_label) #genres[predicted_class])

