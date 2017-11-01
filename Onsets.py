import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.misc
import subprocess
import os

def getMultiFeatureOnsets(XAudio, Fs, hopSize):
    """
    Call Essentia's implemtation of multi feature
    beat tracking
    :param XAudio: Numpy array of raw audio samples
    :param Fs: Sample rate
    :param hopSize: Hop size of each onset function value
    :returns (tempo, beats): Average tempo, numpy array
        of beat intervals in seconds
    """
    from essentia import Pool, array
    import essentia.standard as ess
    X = array(XAudio)
    b = ess.BeatTrackerMultiFeature()
    beats = b(X)
    print("Beat confidence: ", beats[1])
    beats = beats[0]
    tempo = 60/np.mean(beats[1::] - beats[0:-1])
    beats = np.array(np.round(beats*Fs/hopSize), dtype=np.int64)
    return (tempo, beats)

def getDegaraOnsets(XAudio, Fs, hopSize):
    """
    Call Essentia's implementation of Degara's technique
    :param XAudio: Numpy array of raw audio samples
    :param Fs: Sample rate
    :param hopSize: Hop size of each onset function value
    :returns (tempo, beats): Average tempo, numpy array
        of beat intervals in seconds
    """
    from essentia import Pool, array
    import essentia.standard as ess
    X = array(XAudio)
    b = ess.BeatTrackerDegara()
    beats = b(X)
    tempo = 60/np.mean(beats[1::] - beats[0:-1])
    beats = np.array(np.round(beats*Fs/hopSize), dtype=np.int64)
    return (tempo, beats)

def getRNNDBNOnsets(filename, Fs, hopSize):
    """
    Call Madmom's implementation of RNN + DBN beat tracking
    :param filename: Path to audio file
    :param Fs: Sample rate
    :param hopSize: Hop size of each onset function value
    :returns (tempo, beats): Average tempo, numpy array
        of beat intervals in seconds
    """
    print("Computing madmom beats...")
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    proc = DBNBeatTrackingProcessor(fps=100)
    act = RNNBeatProcessor()(filename)
    b = proc(act)
    tempo = 60/np.mean(b[1::] - b[0:-1])
    beats = np.array(np.round(b*Fs/hopSize), dtype=np.int64)
    return (tempo, beats)

def getBeats(XAudio, Fs, TempoBias, hopSize, filename = ""):
    """
    Get beat intervals using dynamic programming beat
    tracking with a tempo bias, or if a tempo bias
    isn't specified, use the madmom RNN+DBN implementation
    :param XAudio: Flat numpy array of audio samples
    :param Fs: Sample rate
    :param hopSize: Hop size of each onset function value
    :param filename: Path to audio file
    :returns (tempo, beats): Average tempo, numpy array
        of beat intervals in seconds
    """
    if TempoBias == 0:
        if len(filename) == 0:
            return getDegaraOnsets(XAudio, Fs, hopSize)
        return getRNNDBNOnsets(filename, Fs, hopSize)
    try:
        import librosa
        return librosa.beat.beat_track(XAudio, Fs, start_bpm = TempoBias, hop_length = hopSize)
    except:
        print("Falling back to Degara for beat tracking...")
        if len(filename) == 0:
            return getDegaraOnsets(XAudio, Fs, hopSize)
        return getRNNDBNOnsets(filename, Fs, hopSize)
