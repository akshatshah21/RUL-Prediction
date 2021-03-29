import numpy as np
import pandas as pd

EPSILON = 1e-15

class time_domain_features :
    
    def __init__(self):
        pass
    
    def running_max(self, a) :
        max = np.zeros((a.shape[0], 1))
        
        for i in range(a.shape[0]):
            print(i)
            subarr = a[0:i+1]
            max[i] = np.max(subarr)
        return max

    def running_min(self, a) :
        min = np.zeros((a.shape[0], 1))

        for i in range(a.shape[0]):
            subarr = a[0:i+1]
            min[i] = np.min(subarr)
        return min
    
    def running_abs_mean(self, a):
        abs_mean = np.zeros((a.shape[0], 1))

        for i in range(a.shape[0]):
            subarr = a[0:i+1]
            abs_mean[i] = np.mean(np.absolute(subarr))
        return abs_mean

    def running_mean(self, a):
        mean = np.zeros((a.shape[0], 1))

        for i in range(a.shape[0]):
            subarr = a[0:i+1]
            mean[i] = np.mean(subarr)
        return mean
    
    def running_rms(self, a):
        rms = np.zeros((a.shape[0], 1))

        for i in range(a.shape[0]):
            subarr = a[0:i+1]
            rms[i] = np.sqrt(np.mean((subarr)**2))
        return rms
    
    def running_smr(self, a):
        smr = np.zeros((a.shape[0], 1))

        for i in range(a.shape[0]):
            subarr = a[0:i+1]
            smr[i] = (np.mean(np.sqrt(np.abs(subarr))))**2
        return smr 
    
    def running_peaktopeak(self, a):
        peaktopeak = np.zeros((a.shape[0], 1))

        for i in range(a.shape[0]):
            subarr = a[0:i+1]
            peaktopeak[i] = abs(np.max(np.absolute(subarr))) * 2
        return peaktopeak
    
    def running_stddev(self, a) :
        stddev = np.zeros((a.shape[0], 1))

        for i in range(a.shape[0]):
            subarr = a[0:i+1]
            mean = np.mean(subarr)
            stddev[i] = np.sqrt(np.mean((subarr - mean)**2))
        return stddev 
    
    def running_kurtosis(self, a) :
        kurtosis = np.zeros((a.shape[0], 1))

        for i in range(a.shape[0]):
            subarr = a[0:i+1]
            mean = np.mean(subarr)
            kurtosis[i] = np.mean((subarr - mean)**4)
        return kurtosis
    
    def running_kurtosisfactor(self, a) :
        kurtosis = self.running_kurtosis(a)
        stddev_4 = self.running_stddev(a) ** 4
        return kurtosis / (EPSILON + stddev_4)
    
    def running_waveformfactor(self, a) :
        rms = self.running_rms(a)
        abs_mean = self.running_abs_mean(a)
        return rms / (EPSILON + abs_mean)
    
    def running_crestfactor(self, a) :
        peaktopeak = self.running_peaktopeak(a)
        rms = self.running_rms(a)
        return peaktopeak / (EPSILON + rms)
        
    def running_impactfactor(self, a) :
        peaktopeak = self.running_peaktopeak(a)
        abs_mean = self.running_abs_mean(a)
        return peaktopeak / (EPSILON + abs_mean)
    
    def running_clearancefactor(self, a) :
        peaktopeak = self.running_peaktopeak(a)
        rms = self.running_rms(a)
        return peaktopeak / (EPSILON + rms)


# a = np.array([1,2,3,1,4,5,6])
# tdf = time_domain_features()

# max = tdf.running_max(a)
# min = tdf.running_min(a)

# abs_mean = tdf.running_abs_mean(a)
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

# res = np.concatenate((max, min, abs_mean, mean, rms, smr, peaktopeak, 
#                       stddev, kurtosis, kurtosisfactor, waveformfactor, crestfactor,
#                       impactfactor, clearancefactor), axis=1)
# print(res)







