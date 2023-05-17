import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import IPython
import os
import IPython.display as ipd # to play the Audio Files
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
import librosa # main package for working with Audio Data
import librosa.display
from pydub import AudioSegment
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
import sys
sys.path.append('/to/ffmpeg')


audio_data_set_path = 'content1/audio3sec/'
metadata = pd.read_csv('Data/features_3_sec.csv')
metadata.head()

genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]


def crea_cartelle_necessarie(genres):
    # os.listdir() mostra tutte le cartelle del path in cui ci troviamo.
    #print('listdir:', os.listdir())

    # Perciò se la cartella 'content' non è presente nel path creiamo le varie directory:
    if 'content1' not in os.listdir():

        # creiamo le cartelle spectogram3sec/train, /test, e audio3sec con all'interno le cartelle con i vari generi musicali 
        for g in genres:
            ".join() utilizzato per unire gli elementi di una sequenza di stringhe in una sola stringa utilizzando un delimitatore specificato"
            path_audio = os.path.join('content1/audio3sec/',f'{g}')
            os.makedirs(path_audio)

crea_cartelle_necessarie(genres)

def split_audio(genres):
    # qui splittiamo gli audio in 3 parti (ogni brano originale dura 30 secondi, perciò ogni brano splittato avrà una durata di 10 secondi) 
    # e li inseriamo nella cartella audio3sec
    i = 0
    for g in genres:
        j=0
        print(f"{g}")
        for filename in os.listdir(os.path.join('Data/genres_original',f"{g}")):
            #print(filename)
            #print(j)
            song  =  os.path.join(f'Data/genres_original/{g}',f'{filename}')
            if (filename[0]!= '.'):
                for w in range(0,10):
                    i = i+1
                    #print(i)
                    t1 = 3*(w)*1000
                    t2 = 3*(w+1)*1000
                    newAudio = AudioSegment.from_wav(song)
                    new = newAudio[t1:t2]
                    if len(str(j)) == 1:
                        new.export(f'content1/audio3sec/{g}/{g}.0000{j}.{w}.wav', format="wav")
                        
                    else:
                        new.export(f'content1/audio3sec/{g}/{g}.000{j}.{w}.wav', format="wav")
                j = j+1

split_audio(genres)


def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfcss_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcss_scaled_features = np.mean(mfcss_features.T, axis=0)
    return mfcss_scaled_features

# rimuove 
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


import tensorflow as tf

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

import time
t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)

from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime

num_batch_size = 32

#checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification_{current_time}.hdf5', verbose = 1, save_best_only= True)

start = datetime.now()
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 50, batch_size = num_batch_size)#, callbacks=[checkpointer], verbose=1)

" history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=300, batch_size=128)"
duration = datetime.now()- start

print(duration)

model.evaluate(X_test, y_test, verbose=0)

pd.DataFrame(history.history).plot(figsize=(12,6))
plt.show()

#model.predict_classes(X_test)
model.predict(X_test)


#filename = 'Data/genres_original/jazz/jazz.00090.wav'
filename = 'ludwig.wav'
audio, sample_rate = librosa.load(filename, res_type='kaiser_fast')
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
