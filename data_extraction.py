import numpy as np
import pandas as pd
import os
# from time_domain_features import time_domain_features
from optimized_tdf import time_domain_features

class data_extraction :

    def __init__(self) :
        pass

    def get_files(self, dir) :
        files = [file for file in sorted(os.listdir(dir)) if 'acc' in file]
        return files

    # def get_data(self, dir) :
    #     data_list = []
    #     files = self.get_files(dir)
    #     # i=0
    #     for file in files :
    #         # print(i)
    #         # i+=1
    #         df = pd.read_csv(f'{dir}/{file}', header=None)
    #         df = df.drop(4, axis=1)

    #         for i in range(0, len(df), 100) :
    #             obs = df.iloc[i:i+100, 4].to_numpy()
    #             max = np.max(obs)
    #             data_list.append(max)
            
    #     data = np.array(data_list)
    #     print(data.shape)
    #     return data

    def get_data(self, dir) :
        data_list = []
        files = self.get_files(dir)
        # i=0
        for file in files :
            # print(i)
            # i+=1
            df = pd.read_csv(f'{dir}/{file}', header=None)
            df = df.drop(4, axis=1)
            df = df.iloc[:, :]
            data_list += df.values.tolist()
            
        data = np.array(data_list)
        print(data.shape)
        return data


d = data_extraction()
data = d.get_data(r'../dataset/Learning_set/Bearing1_1/')

a = data[:, 4].reshape(-1, 1)

res = time_domain_features(a)

np.savez_compressed('./my_data/time_domain_features_72l_compressed.npz', res)
print(res.shape)






