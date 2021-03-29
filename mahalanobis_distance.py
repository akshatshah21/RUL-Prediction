import numpy as np

class mahalanobis_distance :
    
    def __init__(self, a) :
        self.a = a
        self.covariance_matrix = np.cov(a, rowvar=False)
        self.covariance_matrix_inv = np.linalg.inv(self.covariance_matrix)

    def distance(self, z) :
        z_minus_mean = z - np.mean(self.a, axis=0).reshape(1, -1)
        
        left_product = np.dot(z_minus_mean, self.covariance_matrix_inv)
        md = np.dot(left_product, z_minus_mean.T)

        return np.sqrt(md.diagonal())
    

# a = np.array([[64.0, 580.0, 29.0],
#               [66.0, 570.0, 33.0],
#               [68.0, 590.0, 37.0],
#               [69.0, 660.0, 46.0],
#               [73.0, 600.0, 55.0]])
# md = mahalanobis_distance(a)
# print(md.distance(np.array([66, 640, 44])))