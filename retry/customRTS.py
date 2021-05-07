import numpy as np
from customKF import CustomKF

class CustomRTS():
    def __init__(self):
        pass

    def run(self, initial_mean, initial_variance, Q, sigma, y, t):
        z = np.diff(y)
        del_t = np.diff(t)
        print(z)
        print(del_t)

        # Forward batch filter
        kf = CustomKF(Q, sigma)
        prior_means, prior_variances, post_means, post_variances = kf.batch_filter(initial_mean, initial_variance, z, del_t)

        num_samples = len(z)

        # Smoother
        S = [0 for _ in range(num_samples)]
        smoothed_means = [0 for _ in range(num_samples)]
        smoothed_variances = [0 for _ in range(num_samples)]
        
        smoothed_means[-1] = post_means[-1]
        smoothed_variances[-1] = post_variances[-1]

        for j in range(num_samples-2, -1, -1):
            S[j] = post_variances[j] * (1 / prior_variances[j])
            smoothed_means[j] = post_means[j] + S[j] * (smoothed_means[j+1] - prior_means[j+1])
            smoothed_variances[j] = post_variances[j] + S[j] * S[j] * (smoothed_variances[j+1] - prior_variances[j+1])

        K = (del_t[-1] ** 2) * prior_variances[-1] + (sigma ** 2) * del_t[-1]

        M = [0 for _ in range(num_samples)]
        M[-1] = (1 - K * del_t[-1]) * post_variances[-2]

        for j in range(num_samples-2, 0, -1):
            M[j] = post_variances[j] * S[j-1] + S[j] * (M[j+1] - post_variances[j]) * S[j-1]

        expected_η = [0 for _ in range(num_samples)]
        expected_η_square = [0 for _ in range(num_samples)]
        expected_η_η_1 = [0 for _ in range(num_samples)]

        for j in range(num_samples-1, -1, -1):
            expected_η[j] = smoothed_means[j]
            expected_η_square[j] = smoothed_means[j]**2 + smoothed_variances[j]
            if j != 0:
                expected_η_η_1[j] = smoothed_means[j] * smoothed_means[j-1] + M[j]

        return expected_η, expected_η_square, expected_η_η_1
        