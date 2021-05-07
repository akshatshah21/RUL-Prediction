import random
from pprint import pprint

class CustomKF:
    '''
    x, state
    z, observation
    u = 0, control input
    A = 1, describes how state evolves from t-1 to t without control or noise
    B, mapping from control space to state space (describes how control u changes from t-1 to t)
    C, mapping of state x to observation z
    Q, measurement noise variance
    R, process noise variance
    delta, measurement noise/error
    '''
    def __init__(self, process_noise, sigma_square):
        self.R = process_noise
        self.sigma_square = sigma_square

    def predict_step(self, prev_mean, prev_variance):
        prior_mean = prev_mean
        prior_variance = prev_variance + self.R
        return prior_mean, prior_variance

    def correct_step(self, prior_mean, prior_variance, C, Q, observation): 
        # print("KF correct step:", observation, Q, C)       
        K = (prior_variance * C) / (C * prior_variance * C + Q)
        post_mean = prior_mean + K * (observation - C * prior_mean)
        post_variance = (1 - K * C) * prior_variance
        return post_mean, post_variance

    def batch_filter(self, initial_mean, initial_variance, z, del_t):
        assert len(z) == len(del_t), "Length of observations != Length of timestamps"

        prev_mean, prev_variance = initial_mean, initial_variance
        prior_means = []
        prior_variances = []
        post_means = []
        post_variances = []

        for i in range(len(z)):
            Q = self.sigma_square * del_t[i]

            prior_mean, prior_variance = self.predict_step(prev_mean, prev_variance)
            prior_means.append(prior_mean)
            prior_variances.append(prior_variance)

            # print("KF: prior mean", prior_mean, "prior variance", prior_variance)

            post_mean, post_variance = self.correct_step(prior_mean, prior_variance, C=del_t[i], Q=Q, observation=z[i])
            post_means.append(post_mean)
            post_variances.append(post_variance)

            # print("KF: post mean", post_mean, "post variance", post_variance)
            prev_mean = post_mean
            prev_variance = post_variance

        return prior_means, prior_variances, post_means, post_variances

if __name__ == '__main__':
    kf = CustomKF(process_noise=10, sigma_square=1)
    initial_mean = random.random()
    initial_variance = random.random()
    z = []
    del_t = []
    for i in range(10):
        z.append(i * random.random())
        del_t.append(random.random())

    prior_means, prior_variances, post_means, post_variances = kf.batch_filter(initial_mean, initial_variance, z, del_t)
    pprint(post_means)
    pprint(post_variances)
