import numpy as np
import pandas as pd
import math
from scipy import stats
from matplotlib import pyplot as plt

EPSILON = 1e-15

def time_domain_features(a) :
    
    max =  np.maximum.accumulate(a)
    min = np.minimum.accumulate(a)
    

    absmean = np.zeros((a.shape[0], 1))
    sum = 0
    for i in range(a.shape[0]):
        print(i, 'absmean')
        sum += abs(a[i])
        absmean[i] = sum / (i + 1)


    mean = np.zeros((a.shape[0], 1))
    sum = 0
    for i in range(a.shape[0]):
        print(i, 'mean')
        sum += a[i]
        mean[i] = sum / (i + 1)
    

    rms = np.zeros((a.shape[0], 1))
    sum = 0
    for i in range(a.shape[0]):
        print(i, 'rms')
        sum += a[i] ** 2
        rms[i] = np.sqrt(sum / (i+1))
       
    
    smr = np.zeros((a.shape[0], 1))
    sum = 0
    for i in range(a.shape[0]):
        print(i, 'smr')
        sum += math.sqrt(abs(a[i]))
        smr[i] = ((sum / (i+1))) ** 2


    a_temp = np.absolute(a)
    peaktopeak =  np.maximum.accumulate(a_temp) * 2
        

    stddev = np.zeros((a.shape[0], 1))
    sum1 = 0
    sum2 = 0
    for i in range(a.shape[0]):
        print(i, 'stddev')
        # subarr = a[0:i+1]
        # stddev[i] = np.sqrt(np.sum((subarr - mean[i])**2) / (i+1))
        # stddev[i] = stats.tstd(subarr)

        sum1 += a[i]
        sum2 += a[i] ** 2
        stddev[i] = np.sqrt((sum2 / (i+1)) - ((sum1 / (i+1))**2))
    

    kurtosis = np.zeros((a.shape[0], 1))
    for i in range(a.shape[0]):
        print(i, 'kurtosis')
        subarr = a[0:i+1]
        # kurtosis[i] = (np.sum((subarr - mean[i]) ** 4)) / (i+1)
        kurtosis[i] = stats.kurtosis(subarr)
    

    kurtosisfactor = kurtosis / (EPSILON + (stddev ** 4))
    waveformfactor =  rms / (EPSILON + absmean)
    crestfactor =  peaktopeak / (EPSILON + rms)
    impactfactor = peaktopeak / (EPSILON + absmean)
    clearancefactor =  peaktopeak / (EPSILON + rms)

    res = np.concatenate((max, min, absmean, mean, rms, smr, peaktopeak, 
                      stddev, kurtosis, kurtosisfactor, waveformfactor, crestfactor,
                      impactfactor, clearancefactor), axis=1)
    
    return res

# a = np.array([1,2,3,1,4,5,6])
# tdf = time_domain_features()

# max = tdf.running_max(a)
# min = tdf.running_min(a)

# absmean = tdf.running_absmean(a)
# mean = tdf.running_mean(a)

# rms = tdf.running_rms(a)
# smr = tdf.running_smr(a)

# peaktopeak = tdf.running_peaktopeak(a)
# stddev = tdf.running_stddev(a)

# kurtosis = tdf.running_kurtosis(a)
# kurtosisfactor = tdf.running_kurtosisfactor(a)

# waveformfactor = tdf.running_waveformfactor(a)
# crestfactor = tdf.running_crestfactor(a)
# impactfactor = tdf.running_impactfactor(a)
# clearancefactor = tdf.running_clearancefactor(a)

# res = np.concatenate((max, min, absmean, mean, rms, smr, peaktopeak, 
#                       stddev, kurtosis, kurtosisfactor, waveformfactor, crestfactor,
#                       impactfactor, clearancefactor), axis=1)
# print(res)







