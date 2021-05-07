import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from feature_utils import time_domain_features
from mahalanobis_distance import mahalanobis_distance



class DataExtraction:
    def __init__(self):
        self.MD_THRESHOLD = None
        self.w = None
        self.md = None


    def get_dataset(self, folder_name, mode) :
        '''
        parameters:
        train_folder_name = folder path containing training csv's, eg: './dataset/Learning_set/Bearing1_1/', './dataset/Learning_set/Bearing1_2/'
        mode = 'all'/'step'/'max'
            all: consider all data points across all csv files
            step: consider every 100th data point across all csv files
            max: consider the data point with maximum value of horizontal acceleration in each file

        returns:
        data = an np array with shape (n, 5)
        '''
        files = [file for file in sorted(os.listdir(folder_name)) if 'acc' in file]
        data_list = []
        for file in files :
            df = pd.read_csv(f'{folder_name}/{file}', header=None)
            df = df.drop(5, axis=1)
            if mode == 'step' :
                df = df.iloc[::100, :]
            elif mode == 'max' :
                index = df.iloc[:, -1].argmax()
                df = df.iloc[index, :].to_frame().T

            data_list += df.values.tolist()
            
        data = np.array(data_list)
        return data


    
    def set_md_threshold(self, train_folder_name, mode='step') :
        '''
        parameters:
        train_folder_name = folder path containing training csv's, eg: './dataset/Learning_set/Bearing1_1/', './dataset/Learning_set/Bearing1_2/'
        mode = 'all'/'step'/'max'
            all: consider all data points across all csv files
            step: consider every 100th data point across all csv files
            max: consider the data point with maximum value of horizontal acceleration in each file

        sets:
        MD_THRESHOLD = threshold to detect degradation
        w = log of MD value when acceleration > 20g for the first time ...(threshold used for RUL prediction)
        '''
    
        data = self.get_dataset(train_folder_name, mode)
        
        index = np.argmax(data[:, -1] > 20)

        features = time_domain_features(data[:, -1])

        self.md = mahalanobis_distance(features)

        distances = np.zeros((data.shape[0], 1))

        for i in range(data.shape[0]) :
            distances[i] = self.md.distance(features[i, :])

        mean =  np.mean(distances[:1500])
        stddev = np.std(distances[:1500])

        self.MD_THRESHOLD = mean + 3 * stddev
        self.w = np.log(distances[index][0])

        # print(self.w, self.MD_THRESHOLD)



    def get_test_data(self, test_folder_name, mode='step', save_to_file=False, file_path=None):
        '''
        parameters:
        testfolder_name = folder path containing testing csv's, eg: './dataset/Learning_set/Bearing1_1/', './dataset/Learning_set/Bearing1_2/'
        mode = 'all'/'step'/'max'
            all: consider all data points across all csv files
            step: consider every 100th data point across all csv files
            max: consider the data point with maximum value of horizontal acceleration in each file
        save_to_file = True or False. Used to save the md_time np array to file_path such as './mydata/file_name.npz'
        file_name = if save_to_file is true, the md_time np array is stored in filepath such as'./mydata/file_name.npz'

        returns:
        md_time : np array of shape(num_samples, 2) where 
            md_time[:, 0] = time in microseconds
            md_time[:, 1] = MD value
        '''

        data = self.get_dataset(test_folder_name, mode)
        md_time = np.zeros((data.shape[0], 2))

        time = self.get_time(data[:, :-1])
        features = time_domain_features(data[:, -1])

        for i in range(data.shape[0]) :
            md_time[i, 0] = time[i]
            md_time[i, 1] = self.md.distance(features[i, :])
            # print(md_time[i, 1])
        
        if save_to_file and file_path is not None :
            np.savez_compressed(f'{file_path}', md_time)

        return md_time



    def get_time(self, data) :
        '''
        parameters:
        data = np array of shape (num_samples, 4) where
            data[:, 0] = hours
            data[:, 1] = minutes
            data[:, 2] = seconds
            data[:, 3] = microseconds
        
        returns:
        time = np array of shape (num_samples,) which indicates time in microseconds
        '''

        time = data[:, -1]
        time+= data[:, 0] * 3.6 * (10**9)
        time+= data[:, 1] * 60 * (10**6)
        time+= data[:, 2] * (10**6)

        return time


    
    def plot_test_data(self, test_data=None, load_from_file=False, file_path=None) :
        '''
        parameters: 
        test_data : np array of shape (num_samples, 2) returned by the get_test_data function
        load_from_file : True or False. 
        file_path : file path of the npz file where test_data is stored. eg. './my_data/file_name.npz

        The test_data can either be taken from the function parameters or load the test_data from a file
        '''

        if load_from_file and file_path is not None:
            test_data = np.load(file_path)["arr_0"]
        elif test_data is None :
            print('test_data is None')
            return

        distances = np.log(np.absolute(test_data[:, 1] - self.mean))
        # distances = np.log(test_data[:, 1])
        # distances = test_data[:, 1]
        plt.plot(distances)
        plt.xlabel("Samples")
        plt.ylabel("MD values")
        plt.grid()
        plt.show()



    def get_test_data_from_file(self, file_path=None):
        '''
        parameters:  
        file_path : file path of the npz file where test_data is stored. eg. './my_data/file_name.npz

        returns:
        test_data : np array of shape(num_samples, 2) where 
            test_data[:, 0] = time in microseconds
            test_data[:, 1] = MD value
        '''

        test_data = np.load(file_path)["arr_0"]
        return test_data
