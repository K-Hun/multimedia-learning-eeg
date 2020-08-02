import scipy.io as sio
import os
import numpy as np
from pywt import wavedec
from entropy import *
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_rel, f_oneway, zscore
from pyeeg import *
import eeg_consts
import eeg_utils
import eeg_features
from scipy import stats


CHANNELS_NAME = ['FP1', 'F7', 'F3', 'FC5', 'C3', 'CP5',
                 'T7', 'P7', 'FP2', 'F8', 'F4', 'FC6', 'C4', 'CP6', 'T8',
                 'P8', 'FPZ', 'FZ', 'FC2', 'FC1', 'CZ', 'CP2', 'CP1', 'PZ',
                 'P4', 'P3', 'O2', 'OZ', 'O1']
BANDS_NAME = ['delta', 'theta', 'alpha', 'beta']



subjects_p = [
'aghabarari',
'bavafa',
'ehsan',
'farrokhloo',
'hrkazemi',
'omidvari',
'razmara',
'zegaibinia',
'pashayi',
'mkalantari',
'hmostafavi',
'habdoli',
'saeedsoleimani',
'asabri',
]


subjects_np = [
'alihosseini',
'amirasadi',
'elyas',
'mohebbi',
'pouraghil',
'rasooli',
'rezaie',
'sahand',
'avakh',
'emousavi',
'alisamei',
'alifalahat',
'ahmadsedaghatzadeh',
'frezaei',
        ]


subjects_p = [
'aghabarari',
]

subjects_np = [
'alihosseini',
        ]


mulDir = 'D:/eeg_thesis_data/EEG-CLEAN-MUL'
baseDir = 'D:/eeg_thesis_data/EEG-CLEAN-BASE'
dur = 60*5
step = dur*1000
parentDir = 'C:/Users/K Hun/Desktop/CogMulProj/fuck_data/'+str(dur)+'s/eeg/'
m = 0.2
all_subjects = [subjects_p, subjects_np]
pre = ['11p','11np']
subject_idx = 1
for sub_set_index in range(2):
    for subject in all_subjects[sub_set_index]:
        print(subject)

        mulSig = eeg_utils.loadEEG(mulDir+'/'+pre[sub_set_index]+'_mul_'+subject+".mat")
        baseSig = eeg_utils.loadEEG(baseDir+"/"+pre[sub_set_index]+"_base_"+subject+".mat", isBaseline=True)
        the_file = open(parentDir+pre[sub_set_index]+"_" + subject + '.csv', 'a')

        block_index = 1
        for i in range(0,340000,step):
            start = i
            end = i + step
            if end > 340000:
                break
            features = ""
            print("B#"+str(block_index))

            for channel in range(len(CHANNELS_NAME)):
                mulSigBands = eeg_utils.extractBandsWithDWT(mulSig[channel][start:end])
                baseSigBands = eeg_utils.extractBandsWithDWT(baseSig[channel])

                for band in BANDS_NAME:
                    mulSigChBand = np.absolute(mulSigBands[band])
                    baseSigChBand = np.absolute(baseSigBands[band])
                    

                    #trialBandPower = eeg_features.getAveragePowerSpectrum(mulSigChBand)
                    #baseBandPower = eeg_features.getAveragePowerSpectrum(baseSigChBand)
                    #trialBandPower = np.mean(np.argsort(mulSigChBand)[-10:])
                    trialBandPower = stats.hmean(mulSigChBand)
                    #baseBandPower = np.median(baseSigChBand)
                    baseBandPower = stats.hmean(baseSigChBand)

                    ERD_ERS = (baseBandPower - trialBandPower) / baseBandPower

                    appEn = eeg_features.getApproximateEntropy(mulSigChBand, 2, m)
                    sampEn = eeg_features.getSampleEntropy(mulSigChBand, 2, m)
                    specEn = eeg_features.getSpectralEntropy(mulSigChBand)
                    print(band, len(mulSigChBand), sampEn)


                    features += str(ERD_ERS)+", "+str(appEn)+", "+str(sampEn)+", "+str(specEn)+", " 

#                exit()
            the_file.write(features[:-2]+", "+ str(subject_idx) +", " + str(block_index)+ "\n")
            block_index += 1
        subject_idx = subject_idx + 1


