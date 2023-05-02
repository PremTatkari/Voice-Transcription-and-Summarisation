import pyaudio
import numpy as np
import librosa
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import sounddevice as sd
import noisereduce as nr


# Define the audio stream parameters
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
NUM_FEATURES = 40

# Create a pyaudio object for streaming audio
p = pyaudio.PyAudio()

# Open the audio stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Wait for the stream to settle
time.sleep(1)

# Define the function to extract features from the audio stream
def extract_features(audio_data):
    audio_data = librosa.effects.preemphasis(audio_data)
    audio_data = librosa.util.normalize(audio_data)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=RATE, n_fft=1024, hop_length=512, n_mfcc=NUM_FEATURES)
    delta1 = librosa.feature.delta(mfccs, width=3, order=1, mode='interp')
    delta2 = librosa.feature.delta(mfccs, width=3, order=2, mode='interp')
    return np.concatenate((np.mean(mfccs, axis=1), np.mean(delta1, axis=1), np.mean(delta2, axis=1)))



# Define function to train speaker identification model
def train_speaker_model(data_dir):
    X = []
    y = []
    for speaker_dir in os.listdir(data_dir):
        speaker_path = os.path.join(data_dir, speaker_dir)
        print(speaker_path)
        for filename in os.listdir(speaker_path):
            if filename.endswith('.wav'):
                filepath = os.path.join(speaker_path, filename)
                audio_data, _ = librosa.load(filepath, sr=RATE, dtype=np.float32)
                
                # Reduce noise using noisereduce library
                noisy_part = audio_data[:int(1.5 * RATE)]
                reduced_noise = nr.reduce_noise(y=audio_data, y_noise=noisy_part, sr=RATE)
                
                features = extract_features(reduced_noise)
                X.append(features)
                y.append(speaker_dir)
    clf = SVC(kernel='linear')
    clf.fit(X, y)
    return clf


# Define the function to perform speaker identification
def identify_speaker(audio_data, speaker_model):
    noisy_part = audio_data[:int(1.5 * RATE)]
    reduced_noise = nr.reduce_noise(y=audio_data, y_noise=noisy_part, sr=RATE)
    audio_features = extract_features(reduced_noise)
    speaker_label = speaker_model.predict([audio_features])[0]
    return speaker_label

# Define function to test speaker identification model
def test_speaker_model(test_dir, speaker_model):
    X = []
    y_true = []
    for speaker_dir in os.listdir(test_dir):
        speaker_path = os.path.join(test_dir, speaker_dir)
        print(speaker_path)
        for filename in os.listdir(speaker_path):
            if filename.endswith('.wav'):
                filepath = os.path.join(speaker_path, filename)
                audio_data, _ = librosa.load(filepath, sr=RATE, dtype=np.float32)
                noisy_part = audio_data[:int(1.5 * RATE)]
                reduced_noise = nr.reduce_noise(y=audio_data, y_noise=noisy_part, sr=RATE)
                features = extract_features(reduced_noise)
                X.append(features)
                y_true.append(speaker_dir)
    y_pred = speaker_model.predict(X)
    accuracy = accuracy_score(y_true, y_pred)
    print(test_dir)
    return accuracy


def main():
    # Load the trained speaker identification model
    data_dir = 'main/data/train/'
    speaker_model = train_speaker_model(data_dir)

    # Test the speaker identification model
    test_dir = 'main/data/test'
    accuracy = test_speaker_model(test_dir, speaker_model)
    print('Accuracy:', accuracy)

    # Initialize previous speaker label
    prev_speaker = None

    # Continuously stream audio and perform speaker identification
    while True:
        # Read a chunk of audio from the stream
        data = stream.read(CHUNK, exception_on_overflow=False)

        # Convert the raw audio data to a numpy array
        audio_data = np.frombuffer(data, dtype=np.float32)

        # Perform speaker identification
        speaker_label = identify_speaker(audio_data, speaker_model)

       # if(speaker_label != "bg_noise"):
        print('Speaker:', speaker_label)

        # Print the predicted speaker label if it has changed
        # if speaker_label != prev_speaker:
        #     print('Speaker:', speaker_label)
        #     prev_speaker = speaker_label
        
if __name__ == '__main__':
    main()