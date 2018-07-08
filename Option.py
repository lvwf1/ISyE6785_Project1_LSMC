import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats
import time

class Option(object):
    """Compute European option value, greeks, and implied volatility.

    Parameters
    ==========
    S0 : int or float
        initial asset value
    K : int or float
        strike
    T : int or float
        time to expiration as a fraction of one year
    r : int or float
        continuously compounded risk free rate, annualized
    sigma : int or float
        continuously compounded standard deviation of returns
    kind : str, {'call', 'put'}, default 'call'
        type of option

    Resources
    =========
    http://www.thomasho.com/mainpages/?download=&act=model&file=256
    """

    def __init__(self, S0, K, T, r, sigma, kind='call'):
        if kind.istitle():
            kind = kind.lower()
        if kind not in ['call', 'put']:
            raise ValueError('Option type must be \'call\' or \'put\'')

        self.kind = kind
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

        self.d1 = ((np.log(self.S0 / self.K)
                + (self.r + 0.5 * self.sigma ** 2) * self.T)
                / (self.sigma * np.sqrt(self.T)))
        self.d2 = ((np.log(self.S0 / self.K)
                + (self.r - 0.5 * self.sigma ** 2) * self.T)
                / (self.sigma * np.sqrt(self.T)))

        # Several greeks use negated terms dependent on option type
        # For example, delta of call is N(d1) and delta put is N(d1) - 1
        self.sub = {'call' : [0, 1, -1], 'put' : [-1, -1, 1]}

    def value(self):
        """Compute option value."""
        return (self.sub[self.kind][1] * self.S0
               * scipy.stats.norm.cdf(self.sub[self.kind][1] * self.d1)
               + self.sub[self.kind][2] * self.K * np.exp(-self.r * self.T)
               * scipy.stats.norm.cdf(self.sub[self.kind][1] * self.d2))

# This is a function simulating the price path for a Geometric Brownian Motion price model
# dS = mu*S*dt + sigma*S*dW

def gen_paths(S0_1, S0_2, r, delta_1, delta_2, sigma_1, sigma_2, rho, T, M, I):
    dt = float(T) / M
    path_1 = np.zeros((M + 1, I), np.float64)
    path_2 = np.zeros((M + 1, I), np.float64)
    path_1[0] = S0_1
    path_2[0] = S0_2
    for t in range(1, M + 1):
        rand_1 = np.random.standard_normal(I)
        rand_2 = np.random.standard_normal(I)
        path_1[t] = path_1[t - 1] * np.exp((r - delta_1) * dt +
                                           sigma_1 * np.sqrt(dt) * rand_1)
        path_2[t] = path_2[t - 1] * np.exp((r - delta_2) * dt +
                                           rho * sigma_2 * np.sqrt(dt) * rand_1 +
                                           np.sqrt(1-rho**2) * sigma_2 * np.sqrt(dt) * rand_2)
    return [path_1,path_2]

def gen_paths_antithetic(S0_1, S0_2, r, delta_1, delta_2, sigma_1, sigma_2, rho, T, M, I):
    dt = float(T) / M
    path_1 = np.zeros((2*M + 1, I), np.float64)
    path_2 = np.zeros((2*M + 1, I), np.float64)
    path_1[0] = S0_1
    path_2[0] = S0_2
    for t in range(1, M + 1):
        rand_1 = np.random.standard_normal(I)
        rand_2 = np.random.standard_normal(I)
        rand_anti_1 = -1.0 * rand_1  # antithetic variates
        rand_anti_2 = -1.0 * rand_2  # antithetic variates
        path_1[2 * t] = path_1[2 * (t - 1)] * np.exp((r - delta_1) * dt +
                                                   sigma_1 * np.sqrt(dt) * rand_1)
        path_1[2 * t - 1] = path_1[max(2 * t - 3, 0)] * np.exp((r - delta_1) * dt +
                                                             sigma_1 * np.sqrt(dt) * rand_anti_1)
        path_2[2*t] = path_2[2*(t - 1)] * np.exp((r - delta_2) * dt +
                                               rho * sigma_2 * np.sqrt(dt) * rand_1 +
                                               np.sqrt(1 - rho ** 2) * sigma_2 * np.sqrt(dt) * rand_2)
        path_2[2*t-1] = path_2[max(2*t - 3, 0)] * np.exp((r - delta_2) * dt +
                                                       rho * sigma_2 * np.sqrt(dt) * rand_anti_1 +
                                                       np.sqrt(1 - rho ** 2) * sigma_2 * np.sqrt(dt) * rand_anti_2)
    return [path_1,path_2]

def hist_comp(dist1, dist2, lgnd, bin_num):
    hist_start = min(min(dist1), min(dist2))
    hist_end = max(max(dist1), max(dist2))
    bin_vec = np.linspace(hist_start, hist_end, bin_num)
    plt.hist([dist1, dist2], color=['r','g'], label=[lgnd[0],lgnd[1]], alpha=0.8, bins=bin_vec)
    plt.legend(loc='upper right')
    plt.show()

S0_1 = 100.0
S0_2 = 95.0
K = 90.0
r = 0.045
delta_1 = 0.02
sigma_1 = 0.2
delta_2 = 0.005
sigma_2 = 0.25
rho = 0.3
T = 0.5
M = 200 #252
i = 100
discount_factor = np.exp(-r * T)

## Closed-form option price
start_time = time.time()
option_1 = Option(S0_1, K, T, r, sigma_1, 'call')
option_2 = Option(S0_2, K, T, r, sigma_2, 'call')
print('B-S price for Asset 1: %f, time used: %f.' % (option_1.value(), time.time()-start_time))
print('B-S price for Asset 2: %f, time used: %f.' % (option_2.value(), time.time()-start_time))

## Set seed for a random number generator
start_time = time.time()
np.random.seed(17)
[path1,path2] = gen_paths(S0_1, S0_2, r, delta_1, delta_2, sigma_1, sigma_2, rho, T, M, i)
duration = time.time()-start_time

## Plot all sample paths
pd.DataFrame(path1).plot()
plt.xlabel('time')
plt.ylabel('price')
pd.DataFrame(path2).plot()
plt.xlabel('time')
plt.ylabel('price')
plt.show()


# Compute the value of a Call option
CallPayoffAverage_1 = np.average(np.maximum(0, path1[-1] - K))
CallPayoff_1 = discount_factor * CallPayoffAverage_1
print('MC estimator for Asset 1: %f, time used: %f.' % (CallPayoff_1, duration))
CallPayoffAverage_2 = np.average(np.maximum(0, path2[-1] - K))
CallPayoff_2 = discount_factor * CallPayoffAverage_2
print('MC estimator for Asset 2: %f, time used: %f.' % (CallPayoff_2, duration))

## Antithetic variate estimator
start_time = time.time()
[path_1_anti,path_2_anti] = gen_paths_antithetic(S0_1, S0_2, r, delta_1, delta_2, sigma_1, sigma_2, rho, T, M, i)
CallPayoffAverage_1 = np.average(np.maximum(0, path_1_anti[-1] - K))
CallPayoffAverage_tilda_1 = np.average(np.maximum(0, path_1_anti[-2] - K))
CallPayoff_anti_1 = discount_factor * (CallPayoffAverage_1+CallPayoffAverage_tilda_1)/2.0
CallPayoffAverage_2 = np.average(np.maximum(0, path_2_anti[-1] - K))
CallPayoffAverage_tilda_2 = np.average(np.maximum(0, path_2_anti[-2] - K))
CallPayoff_anti_2 = discount_factor * (CallPayoffAverage_2+CallPayoffAverage_tilda_2)/2.0
mcav_time = time.time() - start_time
print('Antithetic variate estimator for path1: %f, Antithetic variate time used: %f.' % (CallPayoff_anti_1,mcav_time))
print('Antithetic variate estimator for path2: %f, Antithetic variate time used: %f.' % (CallPayoff_anti_2,mcav_time))

M = 100  # number of Monte Carlo estimators
MC_vec_1 = []
MCAV_vec_1 = []
MC_vec_2 = []
MCAV_vec_2 = []
for j in range(M):
    np.random.seed(j+1)
    [path1,path2] = gen_paths(S0_1, S0_2, r, delta_1, delta_2, sigma_1, sigma_2, rho, T, M, i)
    CallPayoffAverage_1 = np.average(np.maximum(0, path1[-1] - K))
    CallPayoff_1 = discount_factor * CallPayoffAverage_1
    CallPayoffAverage_2 = np.average(np.maximum(0, path2[-1] - K))
    CallPayoff_2 = discount_factor * CallPayoffAverage_2
    MC_vec_1.append(CallPayoff_1)
    MC_vec_2.append(CallPayoff_2)
    [path_1_anti, path_2_anti] = gen_paths_antithetic(S0_1, S0_2, r, delta_1, delta_2, sigma_1, sigma_2, rho, T, M, i)
    CallPayoffAverage_1 = np.average(np.maximum(0, path_1_anti[-1] - K))
    CallPayoffAverage_tilda_1 = np.average(np.maximum(0, path_1_anti[-2] - K))
    CallPayoff_anti_1 = discount_factor * (CallPayoffAverage_1 + CallPayoffAverage_tilda_1) / 2.0
    CallPayoffAverage_2 = np.average(np.maximum(0, path_1_anti[-1] - K))
    CallPayoffAverage_tilda_2 = np.average(np.maximum(0, path_2_anti[-2] - K))
    CallPayoff_anti_2 = discount_factor * (CallPayoffAverage_2 + CallPayoffAverage_tilda_2) / 2.0
    MCAV_vec_1.append(CallPayoff_anti_1)
    MCAV_vec_2.append(CallPayoff_anti_2)

MC_mean_1 = np.average(MC_vec_1)
MC_std_1 = np.sqrt(np.var(MC_vec_1))
MC_mean_2 = np.average(MC_vec_2)
MC_std_2 = np.sqrt(np.var(MC_vec_2))

print('Naive MC estimator for Asset 1 mean: %f, standard dev: %f.' % (MC_mean_1, MC_std_1))
print('Antithetic Variates MC estimator for Asset 1 mean: %f, standard dev: %f.' % (np.average(MCAV_vec_1), np.sqrt(np.var(MCAV_vec_1))))

print('Naive MC estimator for Asset 2 mean: %f, standard dev: %f.' % (MC_mean_2, MC_std_2))
print('Antithetic Variates MC estimator for Asset 2 mean: %f, standard dev: %f.' % (np.average(MCAV_vec_2), np.sqrt(np.var(MCAV_vec_2))))


### Plot the histogram of the Monte Carlo estimators and the Antithetic Variate estimators
hist_start = min(min(MC_vec_1), min(MCAV_vec_1))
hist_end = max(max(MC_vec_1), max(MCAV_vec_1))
bin_num = 40
bin_vec = np.linspace(hist_start, hist_end, bin_num)
plt.hist([MC_vec_1, MCAV_vec_1], color=['r','g'], label=['MC','MCAV'], alpha=0.8, bins=bin_vec)
plt.legend(loc='upper right')
plt.show()
hist_start = min(min(MC_vec_2), min(MCAV_vec_2))
hist_end = max(max(MC_vec_2), max(MCAV_vec_2))
bin_num = 40
bin_vec = np.linspace(hist_start, hist_end, bin_num)
plt.hist([MC_vec_2, MCAV_vec_2], color=['r','g'], label=['MC','MCAV'], alpha=0.8, bins=bin_vec)
plt.legend(loc='upper right')
plt.show()