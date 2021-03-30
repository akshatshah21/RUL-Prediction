import numpy as np
import pandas as pd
import math
from scipy import stats
from matplotlib import pyplot as plt


import time
EPSILON = 1e-15

def time_domain_features(a) :
    
    squared_a = np.square(a)
    abs_a = np.absolute(a)
    sqrt_a = np.sqrt(abs_a)
    arange_a = np.arange(1, a.shape[0]+1)

    print("Calculating Max min")
    t = time.time()
    max =  np.maximum.accumulate(a).reshape(-1, 1)
    min = np.minimum.accumulate(a).reshape(-1, 1)
    print("Done: ", time.time() - t)
    

    print("Calculating Abs Mean")
    t = time.time()
    absmean = np.cumsum(abs_a) / arange_a
    absmean = absmean.reshape(-1, 1)
    print("Done: ", time.time() - t)


    print("Calculating Mean")
    t = time.time()    
    mean = np.cumsum(a) / arange_a
    mean = mean.reshape(-1, 1)
    print("Done: ", time.time() - t)


    print("Calculating RMS")
    t = time.time()
    rms = np.sqrt(np.cumsum(squared_a) / arange_a )
    rms = rms.reshape(-1, 1)
    print("Done: ", time.time() - t)

    
    print("Calculating SMR")
    t = time.time()
    smr = np.square(np.cumsum(sqrt_a) / arange_a )
    smr = smr.reshape(-1, 1)
    print("Done: ", time.time() - t)


    print("Calculating p2p")
    t = time.time()
    peaktopeak =  np.maximum.accumulate(abs_a) * 2
    peaktopeak = peaktopeak.reshape(-1, 1)
    print("Done: ", time.time() - t)
        

    print("Calculating Std Dev")
    t = time.time()
    stddev = np.sqrt( (np.cumsum(squared_a) / arange_a) -  (np.square(np.cumsum(a) / arange_a)) )
    stddev = stddev.reshape(-1, 1)
    print("Done: ", time.time() - t)
    

    # print("Calculating Kurtosis")
    # t = time.time()
    # kurtosis = np.zeros((a.shape[0], 1))
    # for i in range(a.shape[0]):
    #     # print(i, 'kurtosis')
    #     subarr = a[0:i+1]
    #     # kurtosis[i] = (np.sum((subarr - mean[i]) ** 4)) / (i+1)
    #     kurtosis[i] = stats.kurtosis(subarr)
    #     # print(kurtosis[i])
    #     # if i>5: 
    #     #     break

    # sub = np.cumsum(a).reshape(-1, 1) - (arange_a.reshape(-1, 1) * mean.reshape(-1, 1)).reshape(-1, 1)

    # a_temp = np.cumsum(a).reshape(-1, 1)
    # b_temp = arange_a.reshape(-1, 1) * mean.reshape(-1, 1)
    # sub = (a_temp - b_temp).reshape(-1, 1)
    # # print(a_temp.shape, b_temp.shape, sub.shape)

    # num = np.power(sub, 4)  # .reshape(-1, 1)
    # den = arange_a  # .reshape(-1, 1)
    # kurtosis = num / den
    # print(kurtosis)
    # print("Done: ", time.time() - t)
    

    # kurtosisfactor = kurtosis / (EPSILON + (stddev ** 4))

    waveformfactor =  rms / (EPSILON + absmean)
    crestfactor =  peaktopeak / (EPSILON + rms)
    impactfactor = peaktopeak / (EPSILON + absmean)
    clearancefactor =  peaktopeak / (EPSILON + rms)

    # res = np.concatenate((max, min, absmean, mean, rms, smr, peaktopeak, 
    #                   stddev, kurtosis, kurtosisfactor, waveformfactor, crestfactor,
    #                   impactfactor, clearancefactor), axis=1)
    
    res = np.concatenate((max, min, absmean, mean, rms, smr, peaktopeak, 
                      stddev, waveformfactor, crestfactor,
                      impactfactor, clearancefactor), axis=1)

    return res

