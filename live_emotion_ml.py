#!/usr/bin/python
# Written by Austin Abbey
# With assistance from
# https://www.thepythoncode.com/article/building-a-speech-emotion-recognizer-using-sklearn
#
# Initially inspired by
# https://github.com/Zolmeister/AudioMan
# https://zolmeister.com/2012/10/back-light-music-leds.html

"""
A hobby Python script made to perform sentiment analysis on live
microphone capture, mostly for cosplay purposes and future
interaction with arduino's as an alternative to
gesture, vocal, or physical controls
"""

import glob
import os
import pickle
import sys
import tkinter as tk
import wave
from array import array
from threading import Thread

import librosa
import numpy as np
import pyaudio
import soundfile
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

SILENCE = 10
THRESHOLD = 500

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
WAVE_OUTPUT_FILENAME = "sample.wav"

P = pyaudio.PyAudio()

stream = P.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK,
)

print("* recording")

# MAX_NORMAL = 1
# PREV_VALS = [0, 255]
# PREV = 0
# ALL_ARRAY = []


def is_silent(snd_data):
    """Checks if the input data is silent"""
    return max(snd_data) < THRESHOLD


def normalize(snd_data):
    """Normalizes the audio passed through"""
    maximum = 16384
    times = float(maximum) / max(abs(i) for i in snd_data)

    array_r = array("h")
    for i in snd_data:
        array_r.append(int(i * times))
    return array_r


def trim(snd_data):
    """trims the input audio signal data"""

    def _trim(snd_data2):
        snd_started = False
        array_r = array("h")

        for i in snd_data2:
            if not snd_started and abs(i) > THRESHOLD:
                snd_started = True
                array_r.append(i)
            elif snd_started:
                array_r.append(i)
        return array_r

    snd_data = _trim(snd_data)
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data


def add_silence(snd_data, seconds):
    """adds silence to input data signal"""
    array_r = array("h", [0 for _ in range(int(seconds * RATE))])
    array_r.extend(snd_data)
    array_r.extend([0 for _ in range(int(seconds * RATE))])
    return array_r


class Main:
    """The main class for the script. it will be called later on in the script."""

    def __init__(self):
        print("Starting...")

        print("Loading model...")
        self.model = pickle.load(open("result/mlp_classifier.model", "rb"))

        print("launching TK...")
        self.root = tk.Tk()
        self.label = tk.Label(text="neutral", font=("Arial", 120, "bold"))
        self.label.pack()
        self.root.update()
        Thread(target=self.aud_loop()).start()

        self.root.mainloop()

        # do the loop

    def aud_loop(self):
        """
        general audio loop, intended to run
        in an independent thread through the main loop
        """
        while True:
            array_r = array("h")

            num_silent = 0

            while 1:
                snd_data = array("h", stream.read(CHUNK))
                if sys.byteorder == "big":
                    snd_data.byteswap()
                array_r.extend(snd_data)

                silent = is_silent(snd_data)

                if silent:
                    num_silent += 1

                if num_silent > SILENCE:
                    break

            data = array_r

            data = trim(data)
            data = add_silence(data, 0.5)
            print("writing data")
            wave_file = wave.open(WAVE_OUTPUT_FILENAME, "wb")
            wave_file.setnchannels(CHANNELS)
            wave_file.setsampwidth(pyaudio.get_sample_size(FORMAT))
            wave_file.setframerate(RATE)
            wave_file.writeframes(data)
            wave_file.close()
            print("extracting features")
            try:
                features = extract_feature(
                    WAVE_OUTPUT_FILENAME, mfcc=True, chroma=True, mel=True
                ).reshape(1, -1)
                print("predicting")
                result = self.model.predict(features)[0]
                print(result)

                print("updating TK")
                self.label.config(text=f"{result}")
                self.label.update()
                print("TK Updated")
                # todo: add persistence after further consideration of implementation
            except Exception as file_exception:
                print(file_exception)


def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file 'file_name'
    Features supported:
        - MFCC (mfcc)
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
        sound_file_x = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(sound_file_x))
        result = np.array([])

        if mfcc:
            mfccs = np.mean(
                librosa.feature.mfcc(y=sound_file_x, sr=sample_rate, n_mfcc=40).T,
                axis=0,
            )
            result = np.hstack((result, mfccs))

        if chroma:
            chroma = np.mean(
                librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0
            )
            result = np.hstack((result, chroma))

        if mel:
            mel = np.mean(
                librosa.feature.melspectrogram(sound_file_x, sr=sample_rate).T, axis=0
            )
            result = np.hstack((result, mel))

        if contrast:
            contrast = np.mean(
                librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0
            )
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(
                librosa.feature.tonnetz(
                    y=librosa.effects.harmonic(sound_file_x), sr=sample_rate
                ).T,
                axis=0,
            )
            result = np.hstack((result, tonnetz))
    print(result)
    return result


class Model:
    """
    Handles the ML model. mostly composed of
    training and creation of the model
    """

    int2emotion = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }

    AVAILABLE_EMOTIONS = {"angry", "sad", "neutral", "happy"}

    def load_data(self, test_size=0.2):
        """Loading the sound files to train the ML model"""
        sound_file_x, sound_file_y = [], []
        for file in glob.glob("RAVDESS/data/Actor_*/*.wav"):
            basename = os.path.basename(file)

            emotion = self.int2emotion[basename.split("-")[2]]

            if emotion not in self.AVAILABLE_EMOTIONS:
                continue

            features = extract_feature(file, mfcc=True, chroma=True, mel=True)

            sound_file_x.append(features)
            sound_file_y.append(emotion)
        return train_test_split(
            np.array(sound_file_x), sound_file_y, test_size=test_size, random_state=7
        )

    def train_model(self):
        """trains the ML model"""
        train_x, test_x, y_train, y_test = self.load_data(test_size=0.25)

        # type: ignore
        print("[+] Number of training samples:", train_x.shape[0])
        # type: ignore
        print("[+] Number of testing samples:", test_x.shape[0])
        # type: ignore
        print("[+] Number of features", train_x.shape[1])

        model_params = {
            "alpha": 0.1,
            "batch_size": 256,
            "epsilon": 1e-08,
            "hidden_layer_sizes": (300,),
            "learning_rate": "adaptive",
            "max_iter": 500,
        }

        model = MLPClassifier(**model_params)

        model.fit(train_x, y_train)

        y_pred = model.predict(test_x)

        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

        print(f"Accuracy: {accuracy * 100}%")

        if not os.path.isdir("result"):
            os.mkdir("result")

        pickle.dump(model, open("result/mlp_classifier.model", "wb"))


# Training our model
# Model().train_model()

if not os.path.isdir("result"):
    print("Model not detected! Training now...")
    Model().train_model()

Main()

stream.close()
P.terminate()
