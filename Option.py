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
def gen_paths(S0, r, sigma, T, M, I):
    dt = float(T) / M
    paths = np.zeros((M + 1, I), np.float64)
    paths[0] = S0
    for t in range(1, M + 1):
        rand = np.random.standard_normal(I)
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt +
                                         sigma * np.sqrt(dt) * rand)
    return paths

def gen_paths_antithetic(S0, r, sigma, T, M, I):
    dt = float(T) / M
    paths = np.zeros((2*M + 1, I), np.float64)
    paths[0] = S0
    for t in range(1, M + 1):
        rand = np.random.standard_normal(I)
        rand_anti = -1.0*rand # antithetic variates
        paths[2*t] = paths[2*(t - 1)] * np.exp((r - 0.5 * sigma ** 2) * dt +
                                         sigma * np.sqrt(dt) * rand)
        paths[2*t-1] = paths[max(2*t - 3, 0)] * np.exp((r - 0.5 * sigma ** 2) * dt +
                                         sigma * np.sqrt(dt) * rand_anti)
    return paths

def hist_comp(dist1, dist2, lgnd, bin_num):
    hist_start = min(min(dist1), min(dist2))
    hist_end = max(max(dist1), max(dist2))
    bin_vec = np.linspace(hist_start, hist_end, bin_num)
    plt.hist([dist1, dist2], color=['r','g'], label=[lgnd[0],lgnd[1]], alpha=0.8, bins=bin_vec)
    plt.legend(loc='upper right')
    plt.show()

S0 = 34.
K = 100.0
r = 0.05
sigma = 0.35
T = 1
N = 1 #252
deltat = T / N
i = 1000
discount_factor = np.exp(-r * T)

## Closed-form option price
start_time = time.time()
option = Option(S0, K, T, r, sigma, 'call')
print('B-S price: %f, time used: %f.' % (option.value(), time.time()-start_time))

## Set seed for a random number generator
np.random.seed(17) #
start_time = time.time()
paths = gen_paths(S0, r, sigma, T, N, i)


## Plot all sample paths
#pd.DataFrame(paths).plot()
#plt.show()

# Compute the value of a Call option
CallPayoffAverage = np.average(np.maximum(0, paths[-1] - K))
CallPayoff = discount_factor * CallPayoffAverage
mc_time = time.time() - start_time
print('MC estimator: %f, MC time used: %f.' % (CallPayoff,mc_time))

## Antithetic variate estimator
np.random.seed(17) #
start_time = time.time()
paths_anti = gen_paths_antithetic(S0, r, sigma, T, N, i)
CallPayoffAverage = np.average(np.maximum(0, paths_anti[-1] - K))
CallPayoffAverage_tilda = np.average(np.maximum(0, paths_anti[-2] - K))
CallPayoff_anti = discount_factor * (CallPayoffAverage+CallPayoffAverage_tilda)/2.0
mcav_time = time.time() - start_time
print('Antithetic variate estimator: %f, Antithetic variate time used: %f.' % (CallPayoff_anti,mcav_time))

M = 10000  # number of Monte Carlo estimators
MC_vec = []
MCAV_vec = []
for j in range(M):
    np.random.seed(j+1)
    paths = gen_paths(S0, r, sigma, T, N, i)
    CallPayoffAverage = np.average(np.maximum(0, paths[-1] - K))
    CallPayoff = discount_factor * CallPayoffAverage
    MC_vec.append(CallPayoff)
    paths_anti = gen_paths_antithetic(S0, r, sigma, T, N, i)
    CallPayoffAverage = np.average(np.maximum(0, paths_anti[-1] - K))
    CallPayoffAverage_tilda = np.average(np.maximum(0, paths_anti[-2] - K))
    CallPayoff_anti = discount_factor * (CallPayoffAverage+CallPayoffAverage_tilda)/2.0
    MCAV_vec.append(CallPayoff_anti)


MC_mean = np.average(MC_vec)
MC_std = np.sqrt(np.var(MC_vec))

print('Naive MC estimator mean: %f, standard dev: %f.' % (MC_mean, MC_std))
print('Antithetic Variates MC estimator mean: %f, standard dev: %f.' % (np.average(MCAV_vec), np.sqrt(np.var(MCAV_vec))))

### Plot the histogram of the Monte Carlo estimators and the Antithetic Variate estimators
hist_start = min(min(MC_vec), min(MCAV_vec))
hist_end = max(max(MC_vec), max(MCAV_vec))
bin_num = 40
bin_vec = np.linspace(hist_start, hist_end, bin_num)
plt.hist([MC_vec, MCAV_vec], color=['r','g'], label=['MC','MCAV'], alpha=0.8, bins=bin_vec)
plt.legend(loc='upper right')
plt.show()

## LR() is the likelihood ratio function L in the paper. The following LR() corresponds to the log-normal price model
## Different price models have different LR functions for the Importance Sampling algorithm
def LR(s, s0, mu, rf, sigma, T):
    return np.power(s/s0, (rf-mu)/sigma/sigma)*np.exp((mu*mu - rf*rf)*T/sigma/sigma/2.0)

S0 = 34.
K = 70.0
r = 0.05
sigma = 0.35
T = 1
N = 1 #252
deltat = T / N
M = 10000
discount_factor = np.exp(-r * T)

start_time = time.time()
option = Option(S0, K, T, r, sigma, 'call')
print('B-S price: %f, time used: %f.' % (option.value(), time.time()-start_time))

## mu is the mean parameter for the new samplng distribution
mu = np.log(K) - np.log(S0) - 0.6  # K = 100, -0.96
print(mu)

np.random.seed(1711) #
start_time = time.time()
paths_org = gen_paths(S0, r, sigma, T, N, M)
paths_chg = gen_paths(S0, mu, sigma, T, N, M)

CallPayoffAverage = np.average(np.maximum(0, paths_org[-1] - K))
CallPrice = discount_factor * CallPayoffAverage

#print(np.maximum(0, paths_chg[-1] - K))

CallPayoffAverage_IS = np.average(np.maximum(0, paths_chg[-1] - K)*np.array(LR(paths_chg[-1], S0, mu, r, sigma, T)))
CallPrice_IS = discount_factor * CallPayoffAverage_IS

print('MC estimator: %f, MC Importance Sampling estimator: %f.' % (CallPrice, CallPrice_IS))

Run_num = 1000  # number of Monte Carlo estimators
M = 250
MC_vec = []
MCIS_vec = []
for j in range(Run_num):
    np.random.seed(j*j)
    paths = gen_paths(S0, r, sigma, T, N, M)
    CallPayoffAverage = np.average(np.maximum(0, paths[-1] - K))
    CallPrice = discount_factor * CallPayoffAverage
    MC_vec.append(CallPrice)

    paths_chg = gen_paths(S0, mu, sigma, T, N, M)
    CallPayoffAverage_IS = np.average(np.maximum(0, paths_chg[-1] - K)*np.array(LR(paths_chg[-1], S0, mu, r, sigma, T)))
    CallPrice_IS = discount_factor * CallPayoffAverage_IS
    MCIS_vec.append(CallPrice_IS)

MC_mean = np.average(MC_vec)
MC_std = np.sqrt(np.var(MC_vec))

print('Naive MC estimator mean: %f, standard dev: %f.' % (MC_mean, MC_std))
print('Importance Sampling MC estimator mean: %f, standard dev: %f.' % (np.average(MCIS_vec), np.sqrt(np.var(MCIS_vec))))
legend = ['MC', 'MCIS']
bin_num = 40
hist_comp(MC_vec, MCIS_vec, legend, bin_num)