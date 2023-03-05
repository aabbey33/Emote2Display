#!/usr/bin/python
# Written by Austin Abbey
# With assistance from
# https://www.thepythoncode.com/article/building-a-speech-emotion-recognizer-using-sklearn
#
# Initially inspired by
# https://github.com/Zolmeister/AudioMan
# https://zolmeister.com/2012/10/back-light-music-leds.html


import numpy as np
import tkinter as tk

import librosa
import glob
import os
import pickle
import soundfile
import pyaudio
import sys
import wave

from array import array
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from threading import Thread

SILENCE = 10
THRESHOLD = 500

CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
WAVE_OUTPUT_FILENAME = "sample.wav"

P = pyaudio.PyAudio()

stream = P.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK)

print("* recording")

maxNormal = 1
prevVals = [0, 255]
prev = 0
allArray = []


def is_silent(snd_data):
    return max(snd_data) < THRESHOLD


def normalize(snd_data):
    maximum = 16384
    times = float(maximum) / max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i * times))
    return r


def trim(snd_data):
    def _trim(snd_data2):
        snd_started = False
        r = array('h')

        for i in snd_data2:
            if not snd_started and abs(i) > THRESHOLD:
                snd_started = True
                r.append(i)
            elif snd_started:
                r.append(i)
        return r

    snd_data = _trim(snd_data)
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data


def add_silence(snd_data, seconds):
    r = array('h', [0 for _ in range(int(seconds * RATE))])
    r.extend(snd_data)
    r.extend([0 for _ in range(int(seconds * RATE))])
    return r


class Main:

    def __init__(self):
        print("Starting...")

        print("Loading model...")
        self.model = pickle.load(open("result/mlp_classifier.model", 'rb'))

        print("launching TK...")
        self.root = tk.Tk()
        self.label = tk.Label(text="neutral", font=("Arial", 120, "bold"))
        self.label.pack()
        self.root.update()
        Thread(target=self.aud_loop()).start()

        self.root.mainloop()

        # do the loop

    def aud_loop(self):
        while True:
            r = array('h')

            num_silent = 0

            while 1:
                snd_data = array('h', stream.read(CHUNK))
                if sys.byteorder == 'big':
                    snd_data.byteswap()
                r.extend(snd_data)

                silent = is_silent(snd_data)

                if silent:
                    num_silent += 1

                if num_silent > SILENCE:
                    break

            data = r

            data = trim(data)
            data = add_silence(data, 0.5)

            print("writing data")
            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(data)
            wf.close()


            print("extracting features")
            try:
                features = extract_feature(WAVE_OUTPUT_FILENAME, mfcc=True, chroma=True, mel=True).reshape(1, -1)
                print("predicting")
                result = self.model.predict(features)[0]
                print(result)

                print("updating TK")
                self.label.config(text=f"{result}")
                self.label.update()
                print("TK Updated")

                # todo: add persistence after further consideration of implementation

            except Exception as e:
                print(e)



def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file 'file_name'
    Features supported:
        -MFCC (mfcc)
        - Chroma (chroma)
        - MEL Spectrogram Frequency (mel)
        - Contrast (contrast)
        - Tonnetz (tonnetz)
    e.g:
    'features = extract_feature(path, mel=True, mfcc=True)'
    """

    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")

    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])

        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))

        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))

        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))

        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz))
    print(result)
    return result


class Model:
    int2emotion = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised"
    }

    AVAILABLE_EMOTIONS = {
        "angry",
        "sad",
        "neutral",
        "happy"
    }

    def load_data(self, test_size=0.2):
        X, y = [], []
        for file in glob.glob("RAVDESS/data/Actor_*/*.wav"):
            basename = os.path.basename(file)

            emotion = self.int2emotion[basename.split("-")[2]]

            if emotion not in self.AVAILABLE_EMOTIONS:
                continue

            features = extract_feature(file, mfcc=True, chroma=True, mel=True)

            X.append(features)
            y.append(emotion)
        return train_test_split(np.array(X), y, test_size=test_size, random_state=7)

    def train_model(self):
        X_train, X_test, y_train, y_test = self.load_data(test_size=0.25)

        print("[+] Number of training samples:", X_train.shape[0])
        print("[+] Number of testing samples:", X_test.shape[0])
        print("[+] Number of features", X_train.shape[1])

        model_params = {
            'alpha': 0.1,
            'batch_size': 256,
            'epsilon': 1e-08,
            'hidden_layer_sizes': (300,),
            'learning_rate': 'adaptive',
            'max_iter': 500,
        }

        model = MLPClassifier(**model_params)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

        print(f"Accuracy: {accuracy * 100}%")

        if not os.path.isdir("result"):
            os.mkdir("result")

        pickle.dump(model, open("result/mlp_classifier.model", 'wb'))


# Training our model
# Model().train_model()

if not os.path.isdir("result"):
    print("Model not detected! Training now...")
    Model().train_model()

Main()

stream.close()
P.terminate()
