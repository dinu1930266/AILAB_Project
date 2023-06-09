# Beat tracking example
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np



# 1. Get the file path to an included audio example
filename = librosa.example('nutcracker')
print(filename)
# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`
y, sr = librosa.load(filename)
print(sr)
# 3. Run the default beat tracker
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
print('Estimated tempo: {:.2f} beats per minute'.format(tempo))
# 4. Convert the frame indices of beat events into timestamps
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
print(beat_times)
