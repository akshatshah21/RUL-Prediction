import numpy as np

# from feature_utils import *       # functions for calculating 14 features
# 

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


    def train_data(self, train_folder_list):
        '''
        parameters:
        train_folder_list = list of folder path containing training csv's, eg: ['Learning_set/Bearing1_1', 'Learning_set/Bearing1_2']

        returns / sets:
        μ, σ, md_threshold = threshold to detect degradation
        w = MD value when acceleration > 20g ...(threshold used for RUL prediction)
        '''

        # use static variables for storing features / data / md values
        pass


    def get_features(self, data):
        '''
        parameters: 
        data = vibration data?

        returns:
        feature vector 'dictionary' (length 14)
        '''
       
        pass

    
    def get_mahalonobis_distance(self, feature_vector, mean_vector):
        '''
        parameters:
        feature_vector dictionary = 14 time domain features
        mean_vector = mean values for 14 features

        returns:
        mahalonobis_dist
        '''

        pass


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