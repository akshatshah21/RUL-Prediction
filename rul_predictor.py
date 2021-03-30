import pandas as pd
import numpy as np
from scipy.special import dawsn
import pickle
import os
import sys
# from feature_utils import *       # functions for calculating 14 features
# μ σ η Σ

np.random.seed(42)
class RULPredictor:
    '''
    # TODO
    * Initial values of theta
    * w threshold
    * EM_ITER
    * Flow of test data. Do we need eq 8-9
    * w-y or D are coming negative, is there a relation

    '''

    def __init__(self, debug=False):

        self.MD_THRESHOLD =9
        self.w = 16

        # self.samples = [{
        # 'md': ,
        # 't': ,
        # 'η0_bar': ,
        # 'V_η0': ,
        # 'σ_square': ,
        # 'Q': ,
        # 'η': ,
        # 'V':
        # }]
        self.samples = []
        self.i = -1
        self.EM_ITER = 5
        self.V_prior = 0
        self.debug = debug

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

    def test_data(self):
        '''
        parameters: 
        test_folder_list = list of folder path containing test csv's, eg: ['Test_set/Bearing1_3', 'Test_set/Bearing1_4', ...]

        returns:
        '''
        def parse_time(x):
            pass

        def get_files(dir):
            files = [file for file in sorted(os.listdir(dir)) if 'acc' in file]
            return files

        def get_data(dir):
            # data = np.array([[32.65616884, 30781378320.0],
            #                  [31.87692635, 30781378360.0],
            #                  [32.49983207, 30781378400.0],
            #                  [47.73417412, 30781378440.0],
            #                  [54.6600644, 30781378480.0],
            #                  [53.04537443, 30781378520.0]])

            with open("dataset/data_pickle_1", 'rb') as f:
                data = pickle.load(f)
                data = data[100:200]
                if self.debug:
                    print(data.shape)
                    print(data[:10])
            return data

            data_list = []
            files = self.get_files(dir)
            for file in files:
                df = pd.read_csv(f'{dir}/{file}', header=None)
                df = df.drop(4, axis=1)
                data_list += df.values.tolist()

            data = np.array(data_list)
            # print(data.shape)
            return data

        test_data = get_data("dataset/test_set/Bearing1_3")
        # parse time 0 -> time, 1 -> acc
        test_data = test_data[test_data[:, 1] > self.MD_THRESHOLD]


        for sample in test_data:
            self.i += 1
            self.samples.append({
                "t": sample[0],
                "md": sample[1],
                "η0_bar": 0,
                "V_η0": 1.01,
                "σ_square": 1.01,
                "Q": 1.01,
                "η_cap": 0,
                "V": 1.01
            })
            
            if self.i > 0:
                self.samples[self.i]["η0_bar"], self.samples[self.i]["V_η0"], self.samples[self.i]["σ_square"], self.samples[self.i]["Q"] = self.EM()

                v = np.random.rand() * self.samples[self.i]["Q"]**0.5

                del_t = self.samples[self.i]["t"] - self.samples[self.i-1]["t"]
                eps = np.random.rand() * del_t**0.5
                

                self.samples[self.i]["η_cap"] = self.samples[self.i-1]["η_cap"] + v

                # self.samples[self.i]["md"] = self.samples[self.i-1]["md"] + \
                #     self.samples[self.i-1]["η_cap"] + \
                #     self.samples[self.i]["σ_square"] * eps

                self.KF(self.samples[self.i]["Q"], self.samples[self.i]["σ_square"])

                if self.debug:
                    print("η_cap", self.samples[self.i]["η_cap"])
                    print("V", self.samples[self.i]["V"])
                    print("sig sq ", self.samples[self.i]["σ_square"])
                    print("eps ", eps)
                    print("v=", v)
                    print("η0_bar", self.samples[self.i]["η0_bar"])
                    print("Q:", self.samples[self.i]["Q"])
                    print("md", self.samples[self.i]["md"])
                    print()

                rul = self.predict_RUL() / (10**6)
                print(f"RUL at i={self.i}, t={self.samples[self.i]['t']}: {rul}")
                if self.debug:
                    # print(self.samples)
                    input(f"i {self.i}")
                

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
        P_0 = 1.01
        Q = 1.01
        σ_square = 1.01
        for k in range(self.EM_ITER):
            expected_η, expected_η_square, expected_η_η_1 = self.RTS(
                η_0_bar=η_0_bar, P_0=P_0, Q=Q, σ_square=σ_square)
            # E part
            t1 = np.log(P_0)
            t2 = (expected_η_square[0] - 2 *
                  expected_η[0]*η_0_bar + η_0_bar**2)/P_0
            t3 = 0
            t4 = 0
            for j in range(1, self.i+1):
                del_y = self.samples[j]["md"] - self.samples[j-1]["md"]
                del_t = self.samples[j]["t"] - self.samples[j-1]["t"]

                # print(f"del_t: {del_t}")
                # print(f"del_y: {del_y}")

                t3 += np.log(Q) + (expected_η_square[j] - 2 *
                                   expected_η_η_1[j] + expected_η_square[j-1])/Q
                t4 += np.log(σ_square) + (del_y**2 - 2*expected_η[j-1]*del_y*del_t + (
                    del_t**2)*expected_η_square[j-1])/(σ_square*del_t)
            likelihood = -t1 - t2 - t3 - t4
            # if self.debug:
            #     print(f"i={self.i} Likelihood: {likelihood}")

            # M part
            f3 = 0
            f4 = 0
            for j in range(1, self.i+1):
                del_y = self.samples[j]["md"] - self.samples[j-1]["md"]
                del_t = self.samples[j]["t"] - self.samples[j-1]["t"]
                f3 += (expected_η_square[j] - 2 *
                       expected_η_η_1[j] + expected_η_square[j-1])
                f4 += (del_y**2 - 2*expected_η[j-1]*del_y *
                       del_t + (del_t**2)*expected_η_square[j-1])/del_t

            η_0_bar = expected_η[0]
            P_0 = expected_η_square[0] - expected_η[0]**2 + 0.0001
            Q = f3
            σ_square = f4
        return η_0_bar, P_0, Q, σ_square

    def RTS(self, η_0_bar, P_0, Q, σ_square):
        '''
        Rauch-Tung-Striebel Smoother
        Involves Kalman Filter as forward iteration, and the backward iteration calculates smoothened mean and variance.
        Calculates E(η_j), E(η_j^2) and E(η_j * η_j-1) for use in the Expection Maximization algorithm (the RULPredictor.EM method).
        This method should only be called for self.i >= 1
        ----------
        Parameters:
        η_0_bar: Initial mean of drift
        P_0: Initial variance of drift
        Q: Process covariance (variance of the error term added to drift)
        σ_square: Square of the diffusion coefficient
        '''
        η_cap = [np.random.rand() * P_0**0.5 + η_0_bar]
        P = [P_0]

        # Kalman Filter, forward pass
        P_prior = []
        for j in range(1, self.i+1):
            del_y = self.samples[j]["md"] - self.samples[j-1]["md"]
            del_t = self.samples[j]["t"] - self.samples[j-1]["t"]

            P_prior.append(P[j-1] + Q)

            K = del_t**2 * P_prior[j-1] + σ_square * del_t

            η_cap.append(
                η_cap[j-1] +
                P_prior[j-1] * del_t * (1/K) * (del_y - η_cap[j-1] * del_t)
            )

            P.append(
                P_prior[j-1] -
                P_prior[j-1] * del_t**2 * (1/K) * P_prior[j-1]
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
            S[j] = P[j] * (1 / P_prior[j])
            η_cap_smoothed[j] = η_cap[j] + S[j] * \
                (η_cap_smoothed[j+1] - η_cap[j])
            P_smoothed[j] = P[j] + S[j] * \
                (P_smoothed[j+1] - P_prior[j]) * S[j]

        del_t = self.samples[self.i]["t"] - \
            self.samples[self.i-1]["t"]
        M = [0 for _ in range(self.i+1)]
        M[-1] = (1-K*del_t) * P[self.i-1]

        expected_η = [0 for _ in range(self.i+1)]
        expected_η_square = [0 for _ in range(self.i+1)]
        expected_η_η_1 = [0 for _ in range(self.i+1)]

        for j in range(self.i-1, 0, -1):
            M[j] = P[j] * S[j-1] + S[j] * (M[j+1] - P[j]) * S[j-1]

        for j in range(self.i, -1, -1):
            expected_η[j] = η_cap_smoothed[j]
            expected_η_square[j] = η_cap_smoothed[j]**2 + P_smoothed[j]
            if j != 0:
                expected_η_η_1[j] = η_cap_smoothed[j] * \
                    η_cap_smoothed[j-1] + M[j]

        return expected_η, expected_η_square, expected_η_η_1

    def KF(self, Q, σ_square):
        '''
        Kalman Filter
        '''
        del_y = self.samples[self.i]["md"] - self.samples[self.i-1]["md"]
        del_t = self.samples[self.i]["t"] - self.samples[self.i-1]["t"]

        self.V_prior = self.samples[self.i-1]["V"] + Q

        K = del_t**2 * self.V_prior + σ_square * del_t

        self.samples[self.i]["η_cap"] = self.samples[self.i-1]["η_cap"] + self.V_prior * \
            del_t * (1/K) * (del_y - self.samples[self.i-1]["η_cap"] * del_t)

        self.samples[self.i]["V"] = self.V_prior - \
            self.V_prior * del_t**2 * (1/K) * self.V_prior

    def predict_RUL(self):
        '''
        ( sqrt(2)x(w - y_i) / sqrt(V_i|i) ) x D( η_i_cap / sqrt(2xV_i|i) )
        '''

        D = dawsn(self.samples[self.i]["η_cap"] / ((2*self.samples[self.i]["V"]) ** 0.5))

        if D > 1 or D < -1:
            print(D)
            sys.exit()

        rul = 2**0.5 * \
            (self.w - self.samples[self.i]["md"]) / self.samples[self.i]["V"] * D
        
        return rul

if __name__ == '__main__':
    rp = RULPredictor(debug=True)
    rp.test_data()