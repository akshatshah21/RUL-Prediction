import numpy as np
from numpy.core.fromnumeric import var

# from feature_utils import *       # functions for calculating 14 features
#

# μ σ η Σ


class RULPredictor:
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
        # initial values of theta
        η_0_bar = 0
        P_0 = 0
        Q = 0
        σ_square = 0
        for k in range(self.iter):
            expected_η, expected_η_square, expected_η_η_1 = self.RTS(
                η_0_bar=η_0_bar, P_0=P_0, Q=Q, σ_square=σ_square)
            # E part
            t1 = np.log(P_0)
            t2 = (expected_η_square[0] - 2 *
                  expected_η[0]*η_0_bar + η_0_bar**2)/P_0
            t3 = 0
            t4 = 0
            for j in range(1, self.i+1):
                del_y = self.samples[j].md - self.samples[j-1].md
                del_t = self.samples[j].t - self.samples[j-1].t
                t3 += np.log(Q) + (expected_η_square[j] - 2 *
                                   expected_η_η_1[j] + expected_η_square[j-1])/Q
                t4 += np.log(σ_square) + (del_y**2 - 2*expected_η[j-1]*del_y*del_t + (del_t**2)*expected_η_square[j-1])/(σ_square*del_t)
            likelihood = -t1 - t2 - t3 - t4
            print("Likelihood: ", likelihood)
            # M part
            f3 = 0
            f4 = 0
            for j in range(1, self.i+1):
                del_y = self.samples[j].md - self.samples[j-1].md
                del_t = self.samples[j].t - self.samples[j-1].t
                f3 += (expected_η_square[j] - 2 *
                       expected_η_η_1[j] + expected_η_square[j-1])
                f4 += (del_y**2 - 2*expected_η[j-1]*del_y*del_t + (del_t**2)*expected_η_square[j-1])/del_t

            η_0_bar = expected_η[0]
            P_0 = expected_η_square[0] - expected_η[0]**2
            Q = f3
            σ_square = f4
        return η_0_bar, P_0, Q, σ_square

    def RTS(self, η_0_bar, P_0, Q, σ_square):
        '''
        Rauch-Tung-Striebel Smoother
        Involves Kalman Filter as forward iteration, and the backward iteration calculates smoothened mean and variance.
        Calculates E(η_j), E(η_j^2) and E(η_j * η_j-1) for use in the Expection Maximization algorithm (the RULPredictor.EM method)
        ----------
        Parameters:
        η_0_bar: Initial mean of drift
        P_0: Initial variance of drift
        Q: Process covariance (variance of the error term added to drift)
        σ_square: Square of the diffusion coefficient
        '''
        # Assuming following class members:
        # i: int, denoting latest sample number
        # samples: list of samples (dicts) observed yet

        η_cap = [np.random.rand() * P_0**0.5 + η_0_bar]
        P = [P_0]

        # Kalman Filter, forward pass
        P_prior = []
        for j in range(1, self.i+1):
            del_y = self.samples[j].md - self.samples[j-1].md
            del_t = self.samples[j].t - self.samples[j-1].t

            P_prior.append(P[j-1] + Q)

            K = del_t**2 * P_prior[j] + σ_square * del_t

            η_cap.append(
                η_cap[j-1] +
                P_prior[j] * del_t * (1/K) * (del_y - η_cap[j-1] * del_t)
            )

            P.append(
                P_prior[j] -
                P_prior[j] * del_t**2 * (1/K) * P_prior[j]
            )

        # Backward iteration
        S = [0 for _ in range(self.i)]
        η_cap_smoothed = [0 for _ in range(self.i+1)]
        P_smoothed = [0 for _ in range(self.i+1)]

        # My assumption: η_cap_smoothed[i] = η_cap[i], same for P_smoothed.
        # Cannot find anything else for this
        η_cap_smoothed[-1] = η_cap[-1]
        P_smoothed[-1] = P[-1]

        for j in range(self.i-1, -1, -1):
            S[j] = P[j] * (1 / P_prior[j+1])
            η_cap_smoothed[j] = η_cap[j] + S[j] * \
                (η_cap_smoothed[j+1] - η_cap[j])
            P_smoothed[j] = P[j] + S[j] * \
                (P_smoothed[j+1] - P_prior[j+1]) * S[j]

        del_t = self.samples[self.i].t - \
            self.samples[self.i-1].t  # What about when i == 0?
        M = [0 for _ in range(self.i)]
        M[-1] = (1-K*del_t) * self.P[self.i-1]  # What about when i == 0?

        expected_η = [0 for _ in range(self.i)]
        expected_η_square = [0 for _ in range(self.i)]
        expected_η_η_1 = [0 for _ in range(self.i)]

        for j in range(self.i-1, 0, -1):
            M[j] = P[j] * S[j-1] + S[j] * (M[j+1] - P[j]) * S[j-1]

        for j in range(self.i, -1, -1):
            expected_η[j] = η_cap_smoothed[j]
            expected_η_square[j] = η_cap_smoothed[j]**2 + P_smoothed[j]
            if j != 0:
                expected_η_η_1[j] = η_cap_smoothed[j] * \
                    η_cap_smoothed[j-1] + M[j]

        return expected_η, expected_η_square, expected_η_η_1

    def predict_RUL(self):
        '''
        ( sqrt(2)x(w - y_i) / sqrt(V_i|i) ) x D( η_i_cap / sqrt(2xV_i|i) )
        '''

        pass
