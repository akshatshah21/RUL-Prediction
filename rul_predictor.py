import pandas as pd
import os
from feature_utils import time_domain_features
from mahalanobis_distance import mahalanobis_distance
import numpy as np
from scipy.special import dawsn
import pickle
import warnings
import matplotlib.pyplot as plt
import sys


# μ σ η Σ

DEL_T = 0.1     # considering frequency and scaling by 100 as done in original paper's graph of RUL

np.random.seed(42)

class RULPredictor:
    '''
    # TODO
    * Initial values of theta
    * w threshold
    * EM_ITER
    * Flow of test data. Do we need eq 9 from paper
    * w-y or D are coming negative, is there a relation
    * units of time (hence DEL_T) 

    '''

    def __init__(self, debug=False):

        self.MD_THRESHOLD = None
        self.degrading = False
        self.w = None
        self.md = None

        self.samples = []

        self.η0_bar = 0
        self.V_η0 = 1.01
        self.σ_square = 1.01
        self.Q = 1.01

        self.η_cap = None
        self.P = None

        self.i = -1
        self.EM_ITER = 20
        self.V_prior = 0
        self.debug = debug
        self.start_time = 0

        self.RULs = []

    
    def get_dataset(self, folder_name) :
        files = [file for file in sorted(os.listdir(folder_name)) if 'acc' in file]
        data_list = []
        for file in files :
            df = pd.read_csv(f'{folder_name}/{file}', header=None)
            df = df.drop(5, axis=1)
            df = df.iloc[::100, :]
            data_list += df.values.tolist()
            
        data = np.array(data_list)
        return data


    def set_md_threshold(self, train_folder_name) :
        '''
        parameters:
        train_folder_list = list of folder path containing training csv's, eg: ['Learning_set/Bearing1_1/', 'Learning_set/Bearing1_2/']

        returns / sets:
        μ, σ, md_threshold = threshold to detect degradation
        w = MD value when acceleration > 20g ...(threshold used for RUL prediction)
        '''
        # use static variables for storing features / data / md values
    
        data = self.get_dataset(train_folder_name)
        index = np.argmax(data[:, -1] > 20)

        features = time_domain_features(data[:, -1])

        self.md = mahalanobis_distance(features)

        distances = np.zeros((data.shape[0], 1))

        for i in range(data.shape[0]) :
            distances[i] = self.md.distance(features[i, :])

        mean =  np.mean(distances)
        stddev = np.std(distances)

        self.MD_THRESHOLD = mean + 3 * stddev
        self.w = distances[index][0]


    
    def get_test_data(self, test_folder_name):

        data = self.get_dataset(test_folder_name)
        md_time = np.zeros((data.shape[0], 2))

        time = self.get_time(data[:, 0:-1])
        features = time_domain_features(data[:, -1])

        distances = np.zeros((data.shape[0], 1))
        for i in range(data.shape[0]) :
            md_time[i, 0] = time[i]
            md_time[i, 1] = self.md.distance(features[i, :])

            distances[i] = md_time[i, 1]
            
        return md_time


    def get_time(self, data) :

        time = data[:, -1]
        time+= data[:, 0] * 3.6 * (10**9)
        time+= data[:, 1] * 60 * (10**6)
        time+= data[:, 2] * (10**6)

        return time
        

    def test_data(self, test_data):
        '''
        parameters: 
        set path in get_data() to file containing MD values of samples and corresponding time in microseconds

        '''

        # def get_data():
        #     data = np.load("./dataset/1_1_corr.npz")["arr_0"]
        #     self.start_time = data[0][0]
        #     return data[:]

        # test_data = get_data()


        for sample in test_data:
            if not self.degrading and sample[1] > self.MD_THRESHOLD:    # 1st value to cross MD_THRESHOLD
                self.degrading = True
                print("testing")
                self.i = 0
                self.samples.append({
                    "t": (sample[0] - self.start_time)/1e6,
                    "md": sample[1],
                })
                continue
                
            if self.degrading and self.i >= 0:
                self.i += 1
                
                self.samples.append({
                    "t": (sample[0] - self.start_time)/1e6,
                    "md": sample[1],
                })
                self.η0_bar, self.V_η0, self.σ_square, self.Q = self.EM()   # RTS sets P and eta_cap
                
                ''' # Not sure if following update equations are needed
                v = np.random.rand() * self.samples[self.i]["Q"]**0.5

                del_t = self.samples[self.i]["t"] - self.samples[self.i-1]["t"]
                eps = np.random.rand() * del_t**0.5
                

                self.samples[self.i]["η_cap"] = self.samples[self.i-1]["η_cap"] + v

                self.samples[self.i]["md"] = self.samples[self.i-1]["md"] + \
                    self.samples[self.i-1]["η_cap"] + \
                    self.samples[self.i]["σ_square"] * eps

                self.KF(self.samples[self.i]["Q"], self.samples[self.i]["σ_square"])
                '''
                
                if self.debug:
                    print("md", self.samples[self.i]["md"])
                    
                    print("η0_bar", self.η0_bar)
                    print("V_η0", self.V_η0)
                    print("sig sq ", self.σ_square)
                    print("Q:", self.Q)
                    
                    print()

                rul = self.predict_RUL()
                self.RULs.append(rul)
                print(f"RUL at i={self.i}, t={self.samples[self.i]['t']}: {rul}")
                if self.debug:
                    input(f"i {self.i}")
        

    def EM(self):
        '''
        utilize self.samples, self.y_list, self.theta, self.theta_difference_threshold and set new theta
        '''
        
        # initial values of theta
        η_0_bar = 0
        P_0 = np.random.rand()
        Q = np.random.rand()
        σ_square = np.random.rand()
        
        for k in range(self.EM_ITER):
            expected_η, expected_η_square, expected_η_η_1 = self.RTS(
                η_0_bar=η_0_bar, P_0=P_0, Q=Q, σ_square=σ_square)
            
            # E part ####################################################################################
            t1 = np.log(P_0)
            t2 = (expected_η_square[0] - 2 *
                  expected_η[0]*η_0_bar + η_0_bar**2)/P_0
            t3 = 0
            t4 = 0
            for j in range(1, self.i+1):
                del_y = self.samples[j]["md"] - self.samples[j-1]["md"]
                
                # del_t = self.samples[j]["t"] - self.samples[j-1]["t"]
                del_t = DEL_T
                
                t3 += np.log(Q) + \
                    (expected_η_square[j] - 2 * expected_η_η_1[j] + expected_η_square[j-1])/Q
                t4 += np.log(σ_square) + \
                    (del_y**2 - 2*expected_η[j-1]*del_y*del_t + \
                        del_t**2 * expected_η_square[j-1]) / (σ_square*del_t)
            likelihood = -t1 - t2 - t3 - t4

            
            # M part ####################################################################################
            f3 = 0
            f4 = 0
            for j in range(1, self.i+1):
                del_y = self.samples[j]["md"] - self.samples[j-1]["md"]
                
                # del_t = self.samples[j]["t"] - self.samples[j-1]["t"]
                del_t = DEL_T
                
                f3 += (expected_η_square[j] - 2 *
                       expected_η_η_1[j] + expected_η_square[j-1])  
                f4 += ((del_y**2) - 2*expected_η[j-1]*del_y *
                       del_t + (del_t**2)*expected_η_square[j-1])/del_t
            
            

            η_0_bar = expected_η[0]
            P_0 = expected_η_square[0] - (expected_η[0]**2) 
            Q = f3 / self.i
            σ_square = f4 / self.i
            
            if self.debug:
                print(f"i={self.i} Likelihood: {likelihood}")
                print("eta 0 bar: ",η_0_bar)
                print("P_0: ",P_0)
                print("Q: ",Q)
                print("sigma square: ",σ_square)
        
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
            
            # del_t = self.samples[j]["t"] - self.samples[j-1]["t"]
            del_t = DEL_T
            
            try:
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
            except:
                print(σ_square)
                print(Q)


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
            η_cap_smoothed[j] = η_cap[j] + S[j] * (η_cap_smoothed[j+1] - η_cap[j])
            P_smoothed[j] = P[j] + S[j] * (P_smoothed[j+1] - P_prior[j]) * S[j]

        # del_t = self.samples[self.i]["t"] - self.samples[self.i-1]["t"]
        del_t = DEL_T

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

        self.η_cap = η_cap_smoothed[-1]
        self.P = P_smoothed[-1]
        
        return expected_η, expected_η_square, expected_η_η_1


    def KF(self, Q, σ_square):
        '''
        Kalman Filter
        '''
        
        del_y = self.samples[self.i]["md"] - self.samples[self.i-1]["md"]
        
        # del_t = self.samples[self.i]["t"] - self.samples[self.i-1]["t"]
        del_t = DEL_T

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
        
        try:
            D = dawsn(self.η_cap / ((2*self.P) ** 0.5))
        except:
            print(f"self.η_cap={self.η_cap}")
            print(f"self.P={self.P}")

        rul = abs(((2**0.5) * (self.w - self.samples[self.i]["md"]) * D)/ (self.P**0.5)) * 1e2
        
        return rul

    
    def plot_RUL(self):
        
        plt.plot(self.RULs)
        plt.xlabel("Samples")
        plt.ylabel("RUL in unknown units")
        plt.ylim((0, 1000))
        plt.grid()
        plt.show()

if __name__ == '__main__':
    # rp = RULPredictor(debug=False)
    # rp.test_data()
    # rp.plot_RUL()

    rp = RULPredictor(debug=False)
    rp.set_md_threshold('Learning_set/Bearing1_1/')
    data = rp.get_test_data('Learning_set/Bearing1_1/')
    rp.test_data(data)