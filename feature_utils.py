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

    print("Calculating Max min")
    t = time.time()
    max =  np.maximum.accumulate(a).reshape(-1, 1)
    min = np.minimum.accumulate(a).reshape(-1, 1)
    print("Done: ", time.time() - t)
    

    print("Calculating Abs Mean")
    t = time.time()
    absmean = np.cumsum(abs_a) / arange_a
    absmean = absmean.reshape(-1, 1)
    # print(absmean, absmean.shape)
    print("Done: ", time.time() - t)


    print("Calculating Mean")
    t = time.time()    
    mean = np.cumsum(a) / arange_a
    mean = mean.reshape(-1, 1)
    # print('mean',mean.shape)
    print("Done: ", time.time() - t)


    print("Calculating RMS")
    t = time.time()
    rms = np.sqrt(np.cumsum(squared_a) / arange_a )
    rms = rms.reshape(-1, 1)
    # print(rms)
    print("Done: ", time.time() - t)

    
    print("Calculating SMR")
    t = time.time()
    smr = np.square(np.cumsum(sqrt_a) / arange_a )
    smr = smr.reshape(-1, 1)
    # print(smr)
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
    # print(stddev)
    print("Done: ", time.time() - t)
    

    print("Calculating Waveform factor")
    waveformfactor =  rms / (EPSILON + absmean)
    print("Done")


    print("Calculating Crest factor")
    crestfactor =  peaktopeak / (EPSILON + rms)
    print("Done")


    print("Calculating Impact factor")
    impactfactor = peaktopeak / (EPSILON + absmean)
    print("Done")


    print("Calculating Clearance factor")
    clearancefactor =  peaktopeak / (EPSILON + rms)
    print("Done")


    res = np.concatenate((max, min, absmean, mean, rms, smr, peaktopeak, 
                          stddev, waveformfactor, crestfactor,
                          mpactfactor, clearancefactor), axis=1)

    
    return res
