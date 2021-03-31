import numpy as np
import pandas as pd
import os

class data_extraction :

    def get_files(self, dir) :
        files = [file for file in sorted(os.listdir(dir)) if 'acc' in file]
        
        return files


    def get_data(self, dir) :
        data_list = []
        files = self.get_files(dir)
        
        for file in files :
            df = pd.read_csv(f'{dir}/{file}', header=None)
            df = df.drop(4, axis=1)
            df = df.iloc[::100, :]
            data_list += df.values.tolist()
            
        data = np.array(data_list)
        
        return data







