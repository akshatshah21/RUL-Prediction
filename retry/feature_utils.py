# Helper functions for 14 time domain features calculation

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

    max =  np.maximum.accumulate(a).reshape(-1, 1)
    min = np.minimum.accumulate(a).reshape(-1, 1)

    absmean = np.cumsum(abs_a) / arange_a
    absmean = absmean.reshape(-1, 1)
    
    mean = np.cumsum(a) / arange_a
    mean = mean.reshape(-1, 1)

    rms = np.sqrt(np.cumsum(squared_a) / arange_a )
    rms = rms.reshape(-1, 1)
 
    smr = np.square(np.cumsum(sqrt_a) / arange_a )
    smr = smr.reshape(-1, 1)

    peaktopeak =  np.maximum.accumulate(abs_a) * 2
    peaktopeak = peaktopeak.reshape(-1, 1)
    
    stddev = np.sqrt( (np.cumsum(squared_a) / arange_a) -  (np.square(np.cumsum(a) / arange_a)) )
    stddev = stddev.reshape(-1, 1)

    kurtosis = np.zeros((a.shape[0], 1))
    for i in range(a.shape[0]):
        subarr = a[0:i+1]
        # kurtosis[i] = (np.sum((subarr - mean[i]) ** 4)) / (i+1)
        kurtosis[i] = stats.kurtosis(subarr)
   

    # kurtosisfactor = kurtosis / (EPSILON + (stddev ** 4))

    waveformfactor =  rms / (EPSILON + absmean)
    
    crestfactor =  peaktopeak / (EPSILON + rms)
    
    impactfactor = peaktopeak / (EPSILON + absmean)
    
    clearancefactor =  peaktopeak / (EPSILON + rms)
    
    res = np.concatenate((max, min, absmean, mean, rms, smr, peaktopeak, 
                          stddev, kurtosis, waveformfactor, crestfactor,
                          impactfactor, clearancefactor), axis=1)

    
    return res
