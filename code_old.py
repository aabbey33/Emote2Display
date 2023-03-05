from array import array
import glob
import librosa
import numpy as np
import math
import os
import pickle
import pyaudio
import pygame
import soundfile
import struct
import sys
import time
import wave

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

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
    "surprised",
    "happy"
}

SILENCE = 10
THRESHOLD = 500

black = (0, 0, 0)
yellow = (255, 255, 0)
white = (255, 255, 255)

chunk = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 25000
WAVE_OUTPUT_FILENAME = "sample.wav"

print("loading model...")
model = pickle.load(open("result/mlp_classifier.model", 'rb'))

p = pyaudio.PyAudio()

# Some pygame stuff for implementation later
screenW = 500
screenH = 500
screen = pygame.display.set_mode([screenW, screenH])

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=chunk)

print("* recording")

maxNormal = 1
prevVals = [0, 255]
prev = 0
all = []


def redoVal(r):  # Will I need this? Meant to take the large values and split them depending on amplitude
    if r <= 85:
        return 1
    elif 86 <= r <= 170:
        return 2
    elif r >= 171:
        return 3


# Repurpose for sending to displays
def sendVal(r):
    global maxNormal  # why did I establish these again?
    global prev
    global prevVals

    r = float(r)
    maxNormal = float(maxNormal)

    if r > maxNormal:
        return r
    # no seriously what is this things purpose, it takes r and establishes maxNormal as r?

    normalized = int(r / maxNormal * 255)
    prevVals.append(normalized)

    while len(prevVals) >= 100:
        prevVals = prevVals[1:]
        if sum(prevVals) * 1.0 / len(prevVals) <= 10:
            minNormal = 1
            maxNormal = 1
    norm = (normalized + prev) / 2

    try:
        if norm <= 0:
            norm = 1
    except:
        norm = 1
    print(norm)

    bgC = (int(norm) * int(norm) / 255, int(norm) / 6, int(norm) / 2)
    """ pygame handling
    if norm was 112.0 then background would be (49, 18, 56)
    screen.fill(bgC)
    pygame.event.get()
    pygame.display.update() 
    """


def drawFace(passon):
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    print(screen.get_size())
    w = screen.get_size()[0]
    h = screen.get_size()[1]

    if passon == "angry":  # angy
        pygame.draw.circle(background, white, (w / 2, h / 2), h / 2.25)
        pygame.draw.circle(background, black, (w / 2, h / 2), h / 2.35)
        pygame.draw.circle(background, white, (w / 2.75, 180), h / 10.5)  # left eye
        pygame.draw.circle(background, black, (w / 2.75, 180), h / 13)
        pygame.draw.circle(background, white, (w / 1.60, 180), h / 10.5)  # right eye
        pygame.draw.circle(background, black, (w / 1.60, 180), h / 13)
        smileRect = pygame.Rect((w / 3, h / 2), (w / 3, h / 4))
        pygame.draw.arc(background, white, smileRect, 0, math.pi, 14)
    elif passon == "happy":  # happy
        pygame.draw.circle(background, white, (w / 2, h / 2), h / 2.25)
        pygame.draw.circle(background, black, (w / 2, h / 2), h / 2.35)
        pygame.draw.circle(background, white, (w / 2.75, 180), h / 10.5)  # left eye
        pygame.draw.circle(background, black, (w / 2.75, 180), h / 13)
        pygame.draw.circle(background, white, (w / 1.60, 180), h / 10.5)  # right eye
        pygame.draw.circle(background, black, (w / 1.60, 180), h / 13)
        smileRect = pygame.Rect((w / 3, h / 2), (w / 3, h / 4))
        pygame.draw.arc(background, white, smileRect, math.pi, 0, 14)
    elif passon == "neutral":  # neutral
        pygame.draw.circle(background, white, (w / 2, h / 2), h / 2.25)
        pygame.draw.circle(background, black, (w / 2, h / 2), h / 2.35)
        pygame.draw.circle(background, white, (w / 2.75, 180), h / 10.5)  # left eye
        pygame.draw.circle(background, black, (w / 2.75, 180), h / 13)
        pygame.draw.circle(background, white, (w / 1.60, 180), h / 10.5)  # right eye
        pygame.draw.circle(background, black, (w / 1.60, 180), h / 13)
        # smileRect = pygame.Rect((w/3,h/3), (w/3, h/4))
        # pygame.draw.arc(background, white, smileRect, math.pi, 0, 8)
        pygame.draw.line(background, white, (w / 1.5, h / 1.60), (w / 3, h / 1.60), 14)
    elif passon == "surprised":  # suprrised
        pygame.draw.circle(background, white, (w / 2, h / 2), h / 2.25)
        pygame.draw.circle(background, black, (w / 2, h / 2), h / 2.35)
        pygame.draw.circle(background, white, (w / 2.75, 180), h / 7.5)  # left eye
        pygame.draw.circle(background, black, (w / 2.75, 180), h / 8)
        pygame.draw.circle(background, white, (w / 1.60, 180), h / 7.5)  # right eye
        pygame.draw.circle(background, black, (w / 1.60, 180), h / 8)
    elif passon == "sad":  # sad
        pygame.draw.circle(background, white, (w / 2, h / 2), h / 2.25)
        pygame.draw.circle(background, black, (w / 2, h / 2), h / 2.35)
        pygame.draw.circle(background, white, (w / 2.75, 180), h / 10.5)  # left eye
        pygame.draw.circle(background, black, (w / 2.75, 180), h / 13)
        pygame.draw.circle(background, white, (w / 1.60, 180), h / 10.5)  # right eye
        pygame.draw.circle(background, black, (w / 1.60, 180), h / 13)
        smileRect = pygame.Rect((w / 3, h / 2), (w / 3, h / 4))
        pygame.draw.arc(background, white, smileRect, 0, math.pi, 14)

    screen.blit(background, (0, 0))

    pygame.display.flip()
    pygame.event.get()
    pygame.display.update()


def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD


def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM) / max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i * times))
    return r


def trim(snd_data):
    "Trim the blank spots at the start and end"

    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i) > THRESHOLD:
                snd_started = True
                r.append(i)
            elif snd_started:
                r.append(i)
        return r

    # trim to the left

    snd_data = _trim(snd_data)
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data


def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds * RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds * RATE))])
    return r


"""
As an autistic, handling emotions is not my strong suit (from https://www.thepythoncode.com/article/building-a-speech-emotion-recognizer-using-sklearn)
"""


def extract_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")

    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype='float32')
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
    return result


def load_data(test_size=0.2):  # likely unecessary
    X, y = [], []
    for file in glob.glob("data/Actpr_*/*.wav"):
        basename = os.path.basename(file)
        emotion = int2emotion[basename.split("-")[2]]
        if emotion not in AVAILABLE_EMOTIONS:
            continue
        features = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(features)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=7)


for i in range(0, round(RATE / chunk * RECORD_SECONDS)):
    # try:
    #    data = stream.read(chunk)
    # except:
    #    continue
    # stream.write(data, chunk)
    # all.append(data)

    r = array('h')

    print("reading the stream")

    num_silent = 0

    while 1:
        snd_data = array('h', stream.read(chunk))
        if sys.byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent:
            num_silent += 1
            # print(num_silent)

        if num_silent > SILENCE:
            break

    print("writing data")

    data = r
    # open wave file, set stats
    # data = normalize(data)
    data = trim(data)
    data = add_silence(data, 0.5)

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

    # reopen for reading
    w = wave.open(WAVE_OUTPUT_FILENAME, 'rb')
    summ = 0
    value = 1
    delta = 1
    amps = []

    passOn = "neutral"

    try:
        features = extract_feature(WAVE_OUTPUT_FILENAME, mfcc=True, chroma=True, mel=True).reshape(1, -1)

        result = model.predict(features)[0]

        passOn = result

        lastface = ""

        print("result:", passOn)

        # if lastface != passOn:
        #    drawFace(passOn)
        #    lastface = passOn

        if passOn == "neutral" and lastface == "happy":
            for i in range(0, 20):
                drawFace(lastface)
                print("persistence")
            lastface = lastface
        elif passOn == "neutral" and lastface == "sad":
            for i in range(0, 20):
                drawFace(lastface)
                print("persistence")
            lastface = lastface
        elif passOn == "neutral" and lastface == "angry":
            for i in range(0, 20):
                drawFace(lastface)
                print("persistence")
            lastface = lastface
        elif passOn == "neutral" and lastface == "surprised":
            for i in range(0, 20):
                drawFace(lastface)
                print("persistence")
            lastface = lastface
        else:
            drawFace(passOn)
            lastface = passOn

    except:
        continue

    # for i in range(0, w.getnframes()):
    # data = struct.unpack('<h', w.readframes(1))
    # summ += (data[0] * data[0]) / 2
    # if(i != 0 and (i % 1470) == 0):
    # value = int(math.sqrt(summ / 1470.0) / 10)
    # amps.append(value - delta)
    # summ = 0
    # tarW = str(amps[0] * 1.0 / delta / 100)
    # sendVal(tarW)
    # features=extract_feature(WAVE_OUTPUT_FILENAME,mfcc=True,chroma=True,mel=True).reshape(1,-1)
    # result=model.predict(features)[0]

    # print("result:",result)

    # delta = value

    all = []

print("This should never print")

stream.close()
p.terminate()

# Write data to file