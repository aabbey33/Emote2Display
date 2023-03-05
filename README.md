# Emote2Display

A hobby Python script made to perform sentiment analysis on live microphone capture, mostly for cosplay purposes and future interaction with arduino's as an alternative to gesture, vocal, or physical controls.

merely a proof-of-concept idea as I believe there are better ways to handle things.



I'd love to make an iteration working w/ phonemes, but that will require further research.



- LiveEmotionML 
leveraging ML models to take live audio capture and perform sentiment analysis.

- Stt2e
takes microphone capture, transcripes to text using speechrecognition and them performs sentiment analysis using text2emotion, a prototype for LiveEmotionML



With assistance from

https://www.thepythoncode.com/article/building-a-speech-emotion-recognizer-using-sklearn

https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary

https://towardsdatascience.com/text2emotion-python-package-to-detect-emotions-from-textual-data-b2e7b7ce1153

https://www.simplilearn.com/tutorials/python-tutorial/speech-recognition-in-python

and inspired by
https://github.com/Zolmeister/AudioMan
https://zolmeister.com/2012/10/back-light-music-leds.html

# TODO
add arduino support for cosplay purposes.


Requires Python 3.8
