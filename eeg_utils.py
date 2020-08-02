import scipy.io as sio
import os
import numpy as np
from pywt import wavedec
from entropy import *
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_rel, f_oneway
from pyeeg import *
import eeg_consts


def printFrequencyRangesForDWT(fs, level):
    print("*******************************************")
    max_fs = fs/2
    compos = []
    for i in range(level):
        dec = []
        dec.append(max_fs/2)
        dec.append(max_fs)
        max_fs = max_fs/2
        print(dec)
        compos.append(dec)
    print([0, max_fs])
    print("*******************************************")

def loadEEG(matFilePath, isBaseline = False):
    signals = []
    eegMat = sio.loadmat(matFilePath)


    if isBaseline:
        varName = 'BASELINE'
        rows = eegMat[varName]
    else:
        varName = 'MULTIMEDIA'
        rows = eegMat[varName][0][0][1]

    for row in rows:
        signals.append(row)
    return np.transpose(np.array(signals)) # after transposing the signal, each row indicates a channel

def getChannelSignalByName(eegSignals, channelName, channelsMap,  damagedChannelNumbers = []):
    chMap = channelsMap.copy()
    if len(damagedChannelNumbers) > 0:
        for chNum in damagedChannelNumbers:
            chMap.remove(channelsMap[chNum-1])
            if channelsMap[chNum-1] == channelName:
                return []
    return eegSignals[chMap.index(channelName)]


def extractBandsWithDWT(signal1D):
    cA7, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = wavedec(signal1D, 'db4', level = 7)
    eegBands = {'delta': cA7, 'theta': cD7, 'alpha': cD6, 'beta': cD5}
    return eegBands

def getCohensd(m1,m2,std1,std2):
    return (m2-m1)/sqrt((std1**2 + std2**2)/2)

def compareTwoGroups(sig1, sig2):
    sig1ave = np.average(sig1)
    sig2ave = np.average(sig2)
    sig1std = np.std(sig1)
    sig2std = np.std(sig2)
    stat, p = ttest_ind(sig1, sig2)
    return  sig1ave, sig1std, sig2ave, sig2std, stat, p , getCohensd(sig1ave, sig2ave, sig1std, sig2std)

def createChannelsMapZeros(channelsMap, signalLen):
    sigMap = {}
    for ch in channelsMap:
        sigMap[ch] = np.zeros(signalLen)
    return sigMap

def averageDataSet(desiredChannelsNameList, lenOfEachSignal, parentDir, listOfSubjects, prefix, extension , isBaseline = False):
    dataSetAve = createChannelsMapZeros(channelsMap=desiredChannelsNameList, signalLen=lenOfEachSignal)
    dataSetLenForEachChannel = {}
    for ch in desiredChannelsNameList:
        dataSetLenForEachChannel[ch] = len(listOfSubjects)

    for subjectInfo in listOfSubjects:
        subject = subjectInfo[0]
        dmgChs = subjectInfo[1]
        signals = loadEEG(parentDir+'/'+prefix+subject+extension, isBaseline=isBaseline)
        print(subject)
        for selectedChannel in desiredChannelsNameList:
            extracted = getChannelSignalByName(signals, selectedChannel, eeg_consts.CHANNELS_NAME,  dmgChs)
            if len(extracted) == 0:
                dataSetLenForEachChannel[selectedChannel] -= 1
                continue
            dataSetAve[selectedChannel] = dataSetAve[selectedChannel] + extracted
    for selectedChannel in desiredChannelsNameList:
        dataSetAve[selectedChannel] = dataSetAve[selectedChannel] / dataSetLenForEachChannel[selectedChannel]
    return dataSetAve
