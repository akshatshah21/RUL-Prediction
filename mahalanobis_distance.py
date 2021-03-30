import numpy as np
import random
from matplotlib import pyplot as plt

class mahalanobis_distance :
    
    def __init__(self, a) :
        self.a = a
        self.covariance_matrix = np.cov(a, rowvar=False)
        self.covariance_matrix_inv = np.linalg.inv(self.covariance_matrix)
        self.mean = np.mean(self.a[:], axis=0).reshape(1, -1)

    def distance(self, z) :
        z_minus_mean = z - self.mean
        
        left_product = np.dot(z_minus_mean, self.covariance_matrix_inv)
        md = np.dot(left_product, z_minus_mean.T)

        return np.sqrt(md.diagonal())
    

a = np.load('./my_data/time_domain_features_72l_compressed.npz')['arr_0']
# print(a)

for i in range(a.shape[0]) :
    j = random.randint(0, a.shape[1]-1)
    a[i][j] += 1e-4

md = mahalanobis_distance(a)

distances = np.zeros((a.shape[0], 1))
for i in range(a.shape[0]) :
    print(i)
    distances[i] = md.distance(a[i])

np.savez_compressed('./my_data/md_values_72l_compressed', distances)

print('mean', np.mean(distances))
print('stddev', np.std(distances))

plt.ylim(0, 50)
plt.plot(range(1, distances.shape[0]+1), distances, label='md')

plt.xlabel('samples')
plt.ylabel('md')
plt.grid()
plt.savefig('./my_data/plot_72l.png')
plt.show()