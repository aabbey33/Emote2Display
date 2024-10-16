from array import array

import pyaudio


SILENCE = 10
THRESHOLD = 500

SAMPLES = 1024  # 1024 samples per chunk
FORMAT = pyaudio.paInt16
CHANNELS = 1  # Mono
RATE = 16000  # Sample Rate


def is_silent(snd):
    """
    Check if the maximum value in the sound array is below the
    defined threshold.

    Parameters:
    snd (array): The sound array to be checked.

    Returns:
    bool: True if the maximum value in the sound array is less
    than the threshold, False otherwise.
    """
    return max(snd) < THRESHOLD


def normalize(snd):
    """
    Normalize the input sound array to a maximum value of 16384.

    Parameters:
        snd (array): Input sound array to be normalized.

    Returns:
        array: Normalized sound array.
    """
    maximum = 16384
    times = float(maximum) / max(abs(i) for i in snd)

    array_r = array("h")
    for i in snd:
        array_r.append(int(i * times))
    return array_r


def trim(snd):
    """
    Trim the input sound array based on the defined threshold value.

    Parameters:
    snd (array): The input sound array to be trimmed.

    Returns:
    array: The trimmed sound array.
    """

    def _trim(snd2):
        snd_started = False
        array_r = array("h")

        for i in snd2:
            if not snd_started and abs(*i) > THRESHOLD:
                snd_started = True
                array_r.append(i)
            elif snd_started:
                array_r.append(i)
        return array_r

    snd = _trim(snd)
    snd.reverse()
    snd = _trim(snd)
    snd.reverse()
    return snd


def add_silence(snd, seconds):
    """
    Add silence to the beginning and end of the sound array.

    Args:
        snd (array): The sound array to which silence will be added.
        seconds (float): The duration of silence to add in seconds.

    Returns:
        array: The sound array with silence added.
    """
    array_r = array("h", [0 for _ in range(int(seconds * RATE))])
    array_r.extend(snd)
    array_r.extend([0 for _ in range(int(seconds * RATE))])
    return array_r
