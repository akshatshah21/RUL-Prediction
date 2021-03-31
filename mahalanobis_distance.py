import numpy as np
import random
from matplotlib import pyplot as plt

class mahalanobis_distance :
    
    def __init__(self, a) :

        for i in range(a.shape[0]) :
        j = random.randint(0, a.shape[1]-1)
        a[i][j] += 1e-4

        self.a = a
        self.covariance_matrix = np.cov(a, rowvar=False)
        self.covariance_matrix_inv = np.linalg.inv(self.covariance_matrix)
        self.mean = np.mean(self.a[:], axis=0).reshape(1, -1)

    def distance(self, z) :
        z_minus_mean = z - self.mean
        
        left_product = np.dot(z_minus_mean, self.covariance_matrix_inv)
        md = np.dot(left_product, z_minus_mean.T)

        return np.sqrt(md.diagonal())