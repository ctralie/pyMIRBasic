"""
Chroma / HPCPs
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


HPCP_PRECISION = 1e-5

#Norm-preserving square root (as in "chrompwr.m" by Ellis)
def sqrtCompress(X):
    """
    Square root compress chroma bin values
    :param X: An (NBins x NWindows) array of chroma
    :returns Y: An (NBins x NWindows) sqrt normalized
        chroma matrix
    """
    Norms = np.sqrt(np.sum(X**2, 0))
    Norms[Norms == 0] = 1
    Y = (X/Norms[None, :])**0.5
    NewNorms = np.sqrt(np.sum(Y**2, 0))
    NewNorms[NewNorms == 0] = 1
    Y = Y*(Norms[None, :]/NewNorms[None, :])
    return Y

def getHPCPEssentia(XAudio, Fs, winSize, hopSize, squareRoot = False, NChromaBins = 36):
    """
    Wrap around the essentia library to compute HPCP features
    :param XAudio: A flat array of raw audio samples
    :param Fs: Sample rate
    :param winSize: Window size of each STFT window
    :param hopSize: Hop size between STFT windows
    :param squareRoot: Do square root compression?
    :param NChromaBins: How many chroma bins (default 36)
    :returns H: An (NChromaBins x NWindows) matrix of all
        chroma windows
    """
    import essentia
    from essentia import Pool, array
    import essentia.standard as ess
    print("Getting HPCP Essentia...")
    spectrum = ess.Spectrum()
    window = ess.Windowing(size=winSize, type='hann')
    spectralPeaks = ess.SpectralPeaks()
    hpcp = ess.HPCP(size = NChromaBins)
    H = []
    for frame in ess.FrameGenerator(XAudio, frameSize=winSize, hopSize=hopSize, startFromZero = True):
        S = spectrum(window(frame))
        freqs, mags = spectralPeaks(S)
        H.append(hpcp(freqs, mags))
    H = np.array(H)
    H = H.T
    if squareRoot:
        H = sqrtCompress(H)
    return H

def getParabolicPeaks(X, doParabolic = True):
    """
    Find peaks in intermediate locations using parabolic interpolation
    :param X: A 1D array in which to find interpolated peaks
    :return (bins, freqs): p is signed interval to the left/right of the max
        at which the true peak resides, and b is the peak value
    """
    #Find spectral peaks
    idx = np.arange(1, S.size-1)
    idx = idx[(S[idx-1] < S[idx])*(S[idx+1] < S[idx])]

    p = 0.5*(alpha - gamma)/(alpha-2*beta+gamma)
    b = beta - 0.25*(alpha - gamma)*p
    return (p, b)

def getHarmonicContribTable(NHarmonics):
    harmonicPeaks = []
    for i in range(NHarmonics+1):
        semitone = 12.0*np.log2(i+1.0)
        octweight = max(1.0, (semitone/12.0)*0.5)
        while semitone >= 12.0-HPCP_PRECISION:
            semitone -= 12.0
        

def getHPCP(XAudio, Fs, winSize, hopSize, NChromaBins = 36, minFreq = 40, maxFreq = 5000, 
            bandSplitFreq = 500, NHarmonics = 0, windowSize = 1):
    """
    My implementation of HPCP
    :param XAudio: The raw audio
    :param Fs: The sample rate
    :param winSize: The window size of each HPCP window in samples
    :param hopSize: The hop size between windows
    :param NChromaBins: The number of semitones for each HPCP window (default 36)
    :param minFreq: Minimum frequency to consider (default 40hz)
    :param maxFreq: Maximum frequency to consider (default 5000hz)
    :param bandSplitFreq: The frequency separating low and high bands (default 500hz)
    :param NHarmonics: The number of harmonics to contribute to each semitone (default 0)
    :param windowSize: Size in semitones of window used for weighting
    """
    #Squared cosine weight type
    #windowSize 1 for semitone weighting

    NWin = int(np.floor((len(XAudio)-winSize)/float(hopSize))) + 1
    f, t, S = spectrogram(XAudio[0:winSize], nperseg=winSize, window='blackman')
    #Do STFT window by window,
    for i in range(NWin):
        hpcpLo = np.zeros(NChromaBins)
        hpcpHi = np.zeros(NChromaBins)
        f, t, S = spectrogram(XAudio[i*hopSize:i*hopSize+winSize], nperseg=winSize, window='blackman')
        S = S.flatten()

        #Do parabolic interpolation on each peak
        #https://ccrma.stanford.edu/~jos/parshl/Peak_Detection_Steps_3.html

        #Add contribution of each peak

        #unitMax normalization of low and hi individually

    return None


def getCensFeatures(XAudio, Fs, hopSize, squareRoot = False):
    """
    Wrap around librosa to compute CENs features
    :param XAudio: A flat array of raw audio samples
    :param Fs: Sample rate
    :param hopSize: Hop size between STFT windows
    :param squareRoot: Do square root compression?
    :returns X: A (12 x NWindows) matrix of all
        chroma windows
    """
    import librosa
    X = librosa.feature.chroma_cens(y=XAudio, sr=Fs, hop_length = hopSize)
    if squareRoot:
        X = sqrtCompress(X)
    X = np.array(X, dtype = np.float32)
    return X


if __name__ == '__main__':
    """
    Compare my HPCP features to Essentia's HPCP Features
    """
    from AudioIO import getAudio
    XAudio, Fs = getAudio("piano-chrom.wav")
    w = int(np.floor(Fs/4)*2)

    hopSize = 512#8192
    winSize = hopSize*4#8192#16384
    NChromaBins = 12

    H = getHPCPEssentia(XAudio, Fs, winSize, hopSize, NChromaBins = NChromaBins)
    #H2 = getHPCP(XAudio, Fs, winSize, hopSize, NChromaBins = NChromaBins)
    H2 = H

    Cens = getCensFeatures(XAudio, Fs, hopSize, squareRoot = True)

    plt.subplot(311)
    plt.imshow(H, cmap = 'afmhot', interpolation = 'none', aspect = 'auto')
    plt.title("HPCP Essentia")
    plt.subplot(312)
    plt.imshow(H2, cmap = 'afmhot', interpolation = 'none', aspect = 'auto')
    plt.title("My HPCP")
    plt.subplot(313)
    plt.imshow(Cens, cmap = 'afmhot', interpolation = 'none', aspect = 'auto')
    plt.title("CENS")
    plt.show()
