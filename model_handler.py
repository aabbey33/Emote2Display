"""
    Handles the ML model. mostly composed of
    training and creation of the model
    """

import glob
import os
import pickle

import numpy as np
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.neural_network import MLPClassifier  # type: ignore

import main

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


def load_data(test_size=0.2):
    """Loading the sound files to train the ML model"""
    sound_file_x, sound_file_y = [], []
    for file in glob.glob("RAVDESS/data/Actor_*/*.wav"):
        basename = os.path.basename(file)

        emotion = int2emotion[basename.split("-")[2]]

        if emotion not in AVAILABLE_EMOTIONS:
            continue

        features = main.Main.extract_feature(
            self=main.Main(), file_name=file, mfcc=True, chroma=True, mel=True
        )

        sound_file_x.append(features)
        sound_file_y.append(emotion)
    return train_test_split(
        np.array(sound_file_x), sound_file_y, test_size=test_size, random_state=7
    )


def train_model():
    """trains the ML model"""
    train_x, test_x, y_train, y_test = load_data(test_size=0.25)

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
