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

    def EM(self, y, t, i):
    # initial values of theta
        eta_bar = 0
        var_of_eta = 0
        Q = 0
        σ_square = 0
        for k in range(self.iter):
            expected_η, expected_η_square, expected_η_η_1 = RTS()
            #E part
            t1 = np.log(var_of_eta)
            t2 = (expected_η_square[0] - 2*expected_η[0]*eta_bar + eta_bar**2 )/var_of_eta
            t3 = 0
            t4 = 0
            for j in range(1, i+1):
                t3+= np.log(Q) + (expected_η_square[j] - 2*expected_η_η_1[j] + expected_η_square[j-1])/Q
                t4+= np.log(sigma_square) + ((y[j]-y[j-1])**2 - 2*expected_η[j-1]*(y[j]-y[j-1])*(t[j]-t[j-1]) + ((t[j]-t[j-1])**2)*expected_η_square[j-1])/(sigma_square*(t[j]-t[j-1]))
            likelihood = -t1 - t2 - t3 - t4
            print("Likelihood: ", likelihood)
            #M part 
            f3 = 0
            f4 = 0
            for j in range(1, i+1):
                f3+= (expected_η_square[j] - 2*expected_η_η_1[j] + expected_η_square[j-1])
                f4+= ((y[j]-y[j-1])**2 - 2*expected_η[j-1]*(y[j]-y[j-1])*(t[j]-t[j-1]) + ((t[j]-t[j-1])**2)*expected_η_square[j-1])/(t[j]-t[j-1])
            
            eta_bar = expected_η[0]
            var_of_eta = expected_η_square[0] - expected_η[0]**2
            Q = f3
            sigma_square = f4    
        
    def EM(self):
        '''
        utilize self.samples, self.y_list, self.theta, self.theta_difference_threshold and set new theta
        '''

        pass

    
    def RTS(self):
    
        # class member η_cap: list of floats
        # η_cap_prior: list of floats
        # η_cap_smoothed: list of floats
        # class member P: list of floats
        # P_prior: list of floats
        # class member / input Q: float
        # class member / input σ: float
        # class member i: int
        # class member current_sample: dict
        # class member prev_sample: dict
        # S: list of floats
        # η_cap_0 = [normal distribution(eta_0_bar, var_eta_0)]  
        # Kalman Filter, forward pass
        del_y = self.current_sample["md"] - self.prev_sample["md"]
        del_t = self.current_sample[""]

        self.P_prior.append(self.P[i-1] + self.Q) # P_prior[i] = P[i-1] + Q

        K = del_t**2 * self.P_prior[i] + self.σ**2 * del_t

        self.η_cap.append(
            self.n_cap[i-1] + 
            self.P_prior[i] * del_t * np.inv(K) * (del_y - self.η_cap[i-1] * del_t)
        )

        self.P.append(
            self.P_prior[i] - 
            self.P_prior[i] * del_t**2 * np.inv(K) * self.P_prior[i]
        )

        # Backward iteration
        S = [0 for _ in range(i)]
        η_cap_smoothed = [0 for _ in range(i+1)]
        P_smoothed = [0 for _ in range(i+1)]
        η_cap_smoothed[-1] = self.η_cap[-1]
        P_smoothed[-1] = self.P[-1]

        for j in reversed(range(i)):

            S[j] = P[j] * (1 / P_prior[j+1])

            # both of these require smoothed[j+1] (which will be smoothed[i] the first time) So I am assuming η_cap_smoothed[i] = η_cap[i], same for P
            η_cap_smoothed[j] = self.η_cap[j] + S[j] * (η_cap_smoothed[j+1] - η_cap_prior[j+1])
            P_smoothed[j] = self.P[j] + S[j] * (P_smoothed[j+1] - P_prior[j+1]) * S[j]
        

        M = [0 for _ in range(i)]
        M[-1] = (1-K*del_t) * self.P[i-1]

        expected_η = [0 for _ in range(i)]
        expected_η_square = [0 for _ in range(i)]
        expected_η_η_1 = [0 for _ in range(i)]

        for j in range(i-1, 0, -1):
            M[j] = self.P[j] * S[j-1] + S[j] * (M[j+1] - self.P[j]) * S[j-1]
            
        for j in range(i, -1, -1):
            expected_η[j] = η_cap_smoothed[j]
            expected_η_square[j] = η_cap_smoothed[j]**2 + P_smoothed[j]
            if j is not 0:
                expected_η_η_1[j] = η_cap_smoothed[j] * η_cap_smoothed[j-1] + M[j]            
            
        return expected_η, expected_η_square, expected_η_η_1

    
    def KF(self, del_y, del_t):
        
        pass


    def predict_RUL(self):
        '''
        ( sqrt(2)x(w - y_i) / sqrt(V_i|i) ) x D( η_i_cap / sqrt(2xV_i|i) )
        '''

        pass