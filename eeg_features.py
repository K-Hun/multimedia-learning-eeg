import scipy.io as sio
import os
import numpy as np
from pywt import wavedec
from entropy import *
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_rel, f_oneway
from pyeeg import *
from pyentrp import entropy as ent

def getSpectralEntropy(coeffs):
    Power_Ratio = coeffs/sum(coeffs)
    Spectral_Entropy = 0
    for i in range(0, len(Power_Ratio) - 1):
        Spectral_Entropy += Power_Ratio[i] * log(Power_Ratio[i])
    Spectral_Entropy /= log(len(Power_Ratio))	# to save time, minus one is omitted
    return -1 * Spectral_Entropy

def getApproximateEntropy(signal1d, r, m):
    return app_entropy(signal1d, r)

def getSampleEntropy(signal1d, r, m):
    std_ts = np.std(signal1d)
    return np.average(ent.sample_entropy(signal1d, 2, 0.2*std_ts))
    #return sample_entropy(signal1d, order=2, metric='chebyshev')
    #return samp_entropy(signal1d, r, m)

def getAveragePowerSpectrum(signal1d):
    return np.average(signal1d)

