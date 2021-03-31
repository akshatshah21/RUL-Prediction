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
            df = df.drop(5, axis=1)
            df = df.iloc[::100, :]
            data_list += df.values.tolist()
            
        data = np.array(data_list)
        
        return data

        


    def get_max_data_per_file(self, dir) :
        data_list = []
        files = self.get_files(dir)
        for file in files :
            df = pd.read_csv(f'{dir}/{file}', header=None)
            df = df.drop(5, axis=1)
            index = df.iloc[:, -1].argmax()
            data_list += df.iloc[index, :].tolist()

        data = np.array(data_list)
        return data






