import numpy as np
import pandas as pd
import os

class data_extraction :

    def __init__(self) :
        pass

    def get_files(self, dir) :
        files = [file for file in sorted(os.listdir(dir)) if 'acc' in file]
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
            df = df.iloc[::100, :]
            data_list += df.values.tolist()
            
        data = np.array(data_list)
        return data







