import librosa
import numpy as np
import os
from sklearn.svm import SVC
import speech_recognition as sr

# Define function to extract MFCC features from audio file
def extract_features(filename):
    y, sr = librosa.load(filename)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    return np.mean(mfccs, axis=1)

# Define function to train speaker identification model
def train_speaker_model(data_dir):
    X = []
    y = []
    for speaker_dir in os.listdir(data_dir):
        speaker_path = os.path.join(data_dir, speaker_dir)
        for filename in os.listdir(speaker_path):
            if filename.endswith('.wav'):
                filepath = os.path.join(speaker_path, filename)
                features = extract_features(filepath)
                X.append(features)
                y.append(speaker_dir)
    clf = SVC(kernel='linear')
    print(np.array(X))
    print(np.array(y))
    print(X.shape)
    clf.fit(X, y)
    return clf

# Train speaker identification model on sample training data
data_dir = 'main/data/train'
speaker_model = train_speaker_model(data_dir)

# Load sample audio file for speaker identification and transcription
audio_file = 'main/data/test/female1.wav'
audio_data, sr = librosa.load(audio_file)

# Perform voice identification
audio_features = extract_features(audio_file)
speaker_label = speaker_model.predict([audio_features])[0]
print('Speaker:', speaker_label)


