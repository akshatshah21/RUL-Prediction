import numpy as np
import pandas as pd
import os
from time_domain_features import time_domain_features

class data_extraction :

    def __init__(self) :
        pass

    def get_files(self, dir) :
        files = [file for file in os.listdir(dir) if 'acc' in file]
        return files

    def get_data(self, dir) :
        data_list = []
        files = self.get_files(dir)
        # i=0
        for file in files :
            # print(i)
            # i+=1
            df = pd.read_csv(f'{dir}/{file}', header=None)
            df = df.drop(4, axis=1)
            data_list += df.values.tolist()
        data = np.array(data_list)
        # print(data.shape)
        return data



d = data_extraction()
data = d.get_data(r'./dataset/Learning_set/Bearing1_1/')

a = data[:, 4].reshape(-1, 1)
# a = np.array([1,2,3,4,5,6])
tdf = time_domain_features()

max = tdf.running_max(a)
min = tdf.running_min(a)

abs_mean = tdf.running_abs_mean(a)
mean = tdf.running_mean(a)

rms = tdf.running_rms(a)
smr = tdf.running_smr(a)

peaktopeak = tdf.running_peaktopeak(a)
stddev = tdf.running_stddev(a)

kurtosis = tdf.running_kurtosis(a)
kurtosisfactor = tdf.running_kurtosisfactor(a)

waveformfactor = tdf.running_waveformfactor(a)
crestfactor = tdf.running_crestfactor(a)
impactfactor = tdf.running_impactfactor(a)
clearancefactor = tdf.running_clearancefactor(a)

res = np.concatenate((max, min, abs_mean, mean, rms, smr, peaktopeak, 
                      stddev, kurtosis, kurtosisfactor, waveformfactor, crestfactor,
                      impactfactor, clearancefactor), axis=1)
print(res.shape)







