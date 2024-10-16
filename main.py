"""
A hobby Python script made to perform sentiment analysis on live
microphone capture, mostly for cosplay purposes and future
interaction with arduino's as an alternative to
gesture, vocal, or physical controls
"""

from array import array
import glob
import pickle
import os
import sys
from threading import Thread
import wave
import tkinter as tk

import numpy as np
import librosa
import pyaudio
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import soundfile

from live_emotion_ml import SILENCE, extract_feature

from sound_functions import add_silence, is_silent, trim

SAMPLES = 1024  # 1024 samples per chunk
FORMAT = pyaudio.paInt16
CHANNELS = 1  # Mono
RATE = 16000  # Sample Rate
WAVE_OUTPUT_FILENAME = "sample.wav"


P = pyaudio.PyAudio()

audio_stream = P.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=SAMPLES,
)

print("Active...")


class Main:
    """
    Run an audio loop continuously to process sound data.
    Trim the sound data, add silence, and write it to a WAV file.
    Extract features from the processed sound data and predict the result
    using a pre-trained model.
    Update the TK interface with the predicted result.
    Handle exceptions that may occur during the process.
    """

    def __init__(self):

        print("Loading model")
        self.model = pickle.load(open("result/mlp_classifier.model", "rb"))

        print("launching TK")
        self.root = tk.Tk()
        self.label = tk.Label(text="neutral", font=("Arial", 120, "bold"))
        self.label.pack()
        self.root.update()

        # Starting Main Audio Thread
        Thread(target=self.audio_loop().start())

        # Start the TK Loop
        self.root.mainloop()

    def audio_loop(self):
        """
        Run an audio loop continuously to process sound data.
        Trim the sound data, add silence, and write it to a WAV file.
        Extract features from the processed sound data and predict the result
        using a pre-trained model.
        Update the TK interface with the predicted result.
        Handle exceptions that may occur during the process.
        """

        while True:
            array_r = array("h")

            num_silent = 0

            while True:
                snd = array("h", audio_stream.read(SAMPLES))
                if sys.byteorder == "big":
                    snd.byteswap()
                array_r.extend(snd)

                silent = is_silent(snd)

                if silent:
                    num_silent += 1
                if num_silent > SILENCE:
                    break

            data = array_r

            data = trim(data)
            data = add_silence(data, 0.5)

            # Writing data to wave file

            self.wave_handler(data)

            # Extracting Features / Primary audio

            try:
                features = extract_feature(
                    WAVE_OUTPUT_FILENAME, mfcc=True, chroma=True, mel=True
                ).reshape(1, -1)
                # predicting
                result = self.model.predict(features)[0]
                print(result)

                # updating TK
                self.label.config(text=f"{result}")
                self.label.update()
                # Tk updated

            except Exception as file_exception:
                print(file_exception)

    def wave_handler(self, data):
        """
        Handles writing audio data to a WAV file using the specified
        parameters.

        Args:
            data: Audio data to be written to the WAV file.

        Returns:
            None
        """
        wave_file = wave.open(WAVE_OUTPUT_FILENAME, "wb")
        wave_file.setnchannels(CHANNELS)
        wave_file.setsampwidth(pyaudio.get_sample_size(FORMAT))
        wave_file.setframerate(RATE)
        wave_file.writeframes(data)
        wave_file.close()

    def extract_feature(self, file_name, **kwargs):
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
                    librosa.feature.melspectrogram(sound_file_x, sr=sample_rate).T,
                    axis=0,
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


if __name__ == "__main__":
    if not os.path.isdir("result"):
        print("Model not detected! Training now...")
        Model().train_model()
    Main()

audio_stream.close()
P.terminate()
