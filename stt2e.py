#!/usr/bin/python
# With assistance from:
# https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
# https://towardsdatascience.com/text2emotion-python-package-to-detect-emotions-from-textual-data-b2e7b7ce1153
# https://www.simplilearn.com/tutorials/python-tutorial/speech-recognition-in-python

import operator
from threading import Lock, Thread

import speech_recognition as sr
import text2emotion as te


r = sr.Recognizer()

# Emotion Bit

lock = Lock()


# def recognize_speech_from_mic(recognizer, microphone):
#     if not isinstance(recognizer, sr.Recognizer):
#         raise TypeError("recognizer must be recognizer instance")
#
#     if not isinstance(microphone, sr.Microphone):
#         raise TypeError("microphone bust be Microphone instance")
#
#     with sr.Microphone() as source:
#         recognizer.adjust_for_ambient_noise(source)
#         audio = recognizer.listen(source)
#
#     response = {
#         "success": True,
#         "error": None,
#         "transcription": None
#     }
#
#     try:
#         response["transcription"] = recognizer.recognize_google(audio)
#     except sr.RequestError:
#         response["success"] = False
#         response["error"] = "API unavailable"
#     except sr.UnknownValueError:
#         response["error"] = "Unable to recognize speech"
#
#     return response


# if __name__ == "__main__":
#
#
#     recognizer = sr.Recognizer()
#     microphone = sr.Microphone()
#
#     while True:
#         text = recognize_speech_from_mic(recognizer, microphone)
#         if text["transcription"]:
#             print(text["transcription"])
#             Thread(target=recognize, args=(text["transcription"],lock))
#         if not text["success"]:
#             print(text["success"])
#             Thread(target=recognize, args=(text["transcription"],lock))
#         if text["error"]:
#             print(f"ERROR: {text['error']}")


# function majoris
# while True:
#    with sr.Microphone() as source2:
#
#        r.adjust_for_ambient_noise(source2, duration=0.2)
#
#        audio = r.listen(source2)
#
# myText = r.recognize_google(audio) # I think I might move to Vosk API to have lower latency responses. latency will
# be the name of the game, after all
#
#        myText = myText.lower()
#
#        print("input: ",myText)
#        #SpeakText(myText)
#        recognize(myText)

class Listener:

    def __init__(self):
        self.recognizer = sr.Recognizer()

        # self.listener = Listener

        # Handle Visuals here and output
        print("placeholder")

        Thread(target=self.run_listener).start()

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




Listener()

# def recognize(phrase):
#     emoteRec = te.get_emotion(phrase)  # getting the emotion from a phrase
#     # print(emoteRec)
#     maxStat = max(emoteRec.items(), key=operator.itemgetter(1))[0]
#     print(maxStat)
#     lock.release()
