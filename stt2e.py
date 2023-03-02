#!/usr/bin/python
# With assistance from:
# https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
# https://towardsdatascience.com/text2emotion-python-package-to-detect-emotions-from-textual-data-b2e7b7ce1153
# https://www.simplilearn.com/tutorials/python-tutorial/speech-recognition-in-python

import operator
from threading import Thread

import speech_recognition as sr
import text2emotion as te
import tkinter as tk


class Listener:

    def __init__(self):
        self.recognizer = sr.Recognizer()

        # self.listener = Listener

        # Handle Visuals here and output
        print("placeholder")

        self.root = tk.Tk()
        self.label = tk.Label(text="Neutral", font=("Arial", 120, "bold"))
        self.label.pack()

        Thread(target=self.run_listener).start()

        self.root.mainloop()

        # do the loop

    def run_listener(self):
        while True:
            try:
                with sr.Microphone() as mic:
                    self.recognizer.adjust_for_ambient_noise(mic, duration=0.2)

                    audio = self.recognizer.listen(mic)

                    text = self.recognizer.recognize_google(audio)

                    text = text.lower()

                    if text is not None:
                        self.recognize(text)

            except Exception as e:
                print(e)
                continue

    def recognize(self, phrase):
        emote_rec = te.get_emotion(phrase)

        max_stat = max(emote_rec.items(), key=operator.itemgetter(1))[0]
        print(max_stat)
        self.label.config(text=f"{max_stat}")


Listener()
