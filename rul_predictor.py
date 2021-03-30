import numpy as np
import pandas as pd
from feature_utils import time_domain_features
from mahalanobis_distance import mahalanobis_distance



# μ σ η Σ 
class RUL_PREDICTOR:
    '''

    '''

    def __init__(self):

        self.md_threshold = None
        self.w = None

        # self.samples = [{'y_i': , 't_i': , 'η_i': , 'η0_bar': , 'V_η0': , 'σ^2': , 'Q': , 'η_i_cap': , 'V_i|i': }]
        self.y_list = None          # index of those who cross md_threshold

        self.theta = {'η0_bar': 0, 'V_η0': 1, 'Q': 0, 'σ^2': 0}
        
        self.theta_difference_threshold = None  # for convergence test in EM

        # current_sample = {md': , 't_i': , 'η_i': , 'η0_bar': , 'V_η0': , 'σ^2': , 'Q': , 'η_i_cap': , 'V_i|i': }
        # prev_sample = {md': , 't_i': , 'η_i-1': , 'η0_bar': , 'V_η0': , 'σ^2': , 'Q': , 'η_i-1_cap': , 'V_i-1|i-1': }
        pass



    def get_dataset(self, folder_name) :

        files = [file for file in sorted(os.listdir(dir)) if 'acc' in file]
        data_list = []
        for file in files :
            df = pd.read_csv(f'{dir}/{file}', header=None)
            df = df.drop(4, axis=1)
            df = df.iloc[::100, :]
            data_list += df.values.tolist()
            
        data = np.array(data_list)
        return data


    def set_md_threshold(self, train_folder_name) :
        '''
        parameters:
        train_folder_list = list of folder path containing training csv's, eg: ['Learning_set/Bearing1_1', 'Learning_set/Bearing1_2']

        returns / sets:
        μ, σ, md_threshold = threshold to detect degradation
        w = MD value when acceleration > 20g ...(threshold used for RUL prediction)
        '''
        # use static variables for storing features / data / md values

        data = get_dataset('./' + train_folder_name)
        index = 0
        for d in data[:, -1] :
            if d > 20 :
                index = i
                break
        features = time_domain_features(data[:, -1])
        md = mahalanobis_distance(features)

        distances = np.zeros((data.shape[0], 1))

        for i in range(data.shape[0]) :
            distances[i] = md.distance(a[i])

        mean =  np.mean(distances))
        stddev = np.std(distances))

        self.md_threshold = mean + 3 * stddev
        self.w = distances[index]

    def get_test_data(self, test_folder_name):

        data = get_dataset('./' + test_folder_name)
        md_time = np.zeros((data.shape[0], 2))

        time = get_time(data[:, 0:-1])
        features = time_domain_features(data[:, -1])

        md = mahalanobis_distance(features)

        distances = np.zeros((data.shape[0], 1))
        for i in range(data.shape[0]) :
            md_time[i, 0] = time[i]
            md_time[i, 1] = md.distance(a[i])

            distances[i] = md_time[i, 1]

        return md_time


    def get_time(self, data) :

        time = data[:, -1]
        time+= data[:, 0] * 3.6 * (10**9)
        time+= data[:, 1] * 60 * (10**6)
        time+= data[:, 2] * (10**6)

        return time;


    def test_data(self, test_folder_list):
        '''
        parameters: 
        test_folder_list = list of folder path containing test csv's, eg: ['Test_set/Bearing1_3', 'Test_set/Bearing1_4', ...]

        returns:
        '''

        # Loop on test data
        # maintain y_all_list --> all MD values
        # maintain y_list --> those who cross md_threshold
        # if md > md_threshold:
        #   EM()    # for theta
        #   calculate RUL using formula
        #   update η_i
        #   update y_i
        #   KF()    # for η_i_cap, V_i|i

        pass

    
    def EM(self):
        '''
        utilize self.samples, self.y_list, self.theta, self.theta_difference_threshold and set new theta
        '''

        pass

    
    def RTS(self):
        pass

    
    def KF(self):
        pass


    def predict_RUL(self):
        '''
        ( sqrt(2)x(w - y_i) / sqrt(V_i|i) ) x D( η_i_cap / sqrt(2xV_i|i) )
        '''

        pass