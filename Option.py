import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats
import time
import math

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
    M : int
        grid or granularity for time (in number of total points)
    r : int or float
        continuously compounded risk free rate, annualized
    i : int
        number of simulations
    sigma : int or float
        continuously compounded standard deviation of returns
    delta : int or float
    rho: int or float
    kind : str, {'call', 'put'}, default 'call'
        type of option

    Resources
    =========
    http://www.thomasho.com/mainpages/?download=&act=model&file=256
    """

    def __init__(self, S0, K, T, M, r, delta, sigma, i, kind='call'):
        if kind.istitle():
            kind = kind.lower()
        if kind not in ['call', 'put']:
            raise ValueError('Option type must be \'call\' or \'put\'')

        self.S0 = S0
        self.K = K
        self.T = T
        self.M = int(M)
        self.r = r
        self.delta = delta
        self.sigma = sigma
        self.i = int(i)
        self.kind = kind
        self.time_unit = self.T / float(self.M)
        self.discount = np.exp(-self.r * self.time_unit)

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

    def AmericanPutPrice(self, seed):
        """ Returns Monte Carlo price matrix rows: time columns: price-path simulation """
        np.random.seed(seed)
        path = np.zeros((self.M + 1, self.i), dtype=np.float64)
        path[0] = self.S0
        for t in range(1, self.M + 1):
            rand = np.random.standard_normal(int(self.i / 2))
            rand = np.concatenate((rand, -rand))
            path[t] = (path[t - 1] * np.exp((self.r - self.delta) * self.time_unit + self.sigma * np.sqrt(self.time_unit) * rand))

        """ Returns the inner-value of American Option """
        if self.kind == 'call':
            payoff = np.maximum(path - self.K, np.zeros((self.M + 1, self.i), dtype=np.float64))
        else:
            payoff = np.maximum(self.K - path, np.zeros((self.M + 1, self.i), dtype=np.float64))

        value = np.zeros_like(payoff)
        value[-1] = payoff[-1]
        for t in range(self.M - 1, 0, -1):
            regression = np.polyfit(path[t], value[t + 1] * self.discount, 5)
            continuation_value = np.polyval(regression, path[t])
            value[t] = np.where(payoff[t] > continuation_value, payoff[t], value[t + 1] * self.discount)

        return np.sum(value[1] * self.discount) / float(self.i)

# This is a function simulating the price path for a Geometric Brownian Motion price model
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

#1
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

seed = 84
## Closed-form option price
start_time = time.time()
call_option_1 = Option(S0_1, K, T, M, r, delta_1, sigma_1, i, 'call')
call_option_2 = Option(S0_2, K, T, M, r, delta_2, sigma_2, i, 'call')
print('B-S price for Asset 1: %f, time used: %f.' % (call_option_1.value(), time.time()-start_time))
print('B-S price for Asset 2: %f, time used: %f.' % (call_option_2.value(), time.time()-start_time))

## Set seed for a random number generator
start_time = time.time()
np.random.seed(seed)
[path1,path2] = gen_paths(S0_1, S0_2, r, delta_1, delta_2, sigma_1, sigma_2, rho, T, M, i)
duration = time.time()-start_time

## Plot all sample paths
pd.DataFrame(path1).plot()
plt.xlabel('time')
plt.ylabel('price')
plt.title("Simulate 100 price paths of S1")
pd.DataFrame(path2).plot()
plt.xlabel('time')
plt.ylabel('price')
plt.title("Simulate 100 price paths of S2")
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
best_j1 = 0
best_j2 = 0
best_diff = math.inf

#test different seed for Monte Carlo estimators
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
    diff1 = abs(CallPayoff_anti_1 - call_option_1.value())
    diff2 = abs(CallPayoff_anti_2 - call_option_2.value())
    if diff1 + diff2 < best_diff:
        best_diff = diff1 + diff2
        best_j = j

print('Best Seed for Asset S1 and S2:', best_j)

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
plt.xlabel('price')
plt.ylabel('number of sample')
plt.legend(loc='upper right')
plt.title("Price Sample Distribution of Naive MC estimator Vs. Antithetic variate MC estimator of S1")
plt.show()
hist_start = min(min(MC_vec_2), min(MCAV_vec_2))
hist_end = max(max(MC_vec_2), max(MCAV_vec_2))
bin_num = 40
bin_vec = np.linspace(hist_start, hist_end, bin_num)
plt.hist([MC_vec_2, MCAV_vec_2], color=['r','g'], label=['MC','MCAV'], alpha=0.8, bins=bin_vec)
plt.xlabel('price')
plt.ylabel('number of sample')
plt.legend(loc='upper right')
plt.title("Price Sample Distribution of Naive MC estimator Vs. Antithetic variate MC estimator of S2")
plt.show()

#2 Price American Put Option
put_option_1 = Option(S0_1, K, T, M, r, delta_1, sigma_1, i, 'put')
mc_price = []
bs_price = []
best_diff = math.inf
for j in range(M):
    mc_put_option_1 = Option(S0_1, K, T, M, r, delta_1, sigma_1, i, 'put').AmericanPutPrice(j)
    mc_price.append(mc_put_option_1)
    bs_price.append(put_option_1.value())
    diff = abs(mc_put_option_1 - put_option_1.value())
    if diff < best_diff:
        best_diff = diff
        best_seed = j

plt.plot(mc_price,label='MC')
plt.plot(bs_price,label='BS')
plt.xlabel('seed')
plt.ylabel('price')
plt.legend(loc='upper right')
plt.title("Black Scholes Price Vs. Monte Carlo Price Simulation in Different Seed")
plt.show()

print('Best Seed of Put Option for Asset S1:', best_seed)
seed = best_seed
print('American Put Option Price (MC) for S1: %f' % Option(S0_1, K, T, M, r, delta_1, sigma_1, i, 'put').AmericanPutPrice(seed))
print('American Put Option Price (BS) for S1: %f' % put_option_1.value())

#3 Price American Call Option
K = 15
call_option_1 = Option(S0_1, K, T, M, r, delta_1, sigma_1, i, 'call')
mc_price = []
bs_price = []
best_diff = math.inf
for j in range(M):
    mc_call_option_1 = Option(S0_1, K, T, M, r, delta_1, sigma_1, i, 'call').AmericanPutPrice(j)
    mc_price.append(mc_call_option_1)
    bs_price.append(call_option_1.value())
    diff = abs(mc_call_option_1 - call_option_1.value())
    if diff < best_diff:
        best_diff = diff
        best_seed = j

plt.plot(mc_price,label='MC')
plt.plot(bs_price,label='BS')
plt.xlabel('seed')
plt.ylabel('price')
plt.legend(loc='upper right')
plt.title("Black Scholes Price Vs. Monte Carlo Price Simulation in Different Seed")
plt.show()

print('Best Seed of Call Option for Asset S1:', best_seed)
seed = best_seed
print ('American Call Option Price (MC): ', Option(S0_1, K, T, M, r, delta_1, sigma_1, i, 'call').AmericanPutPrice(seed))
print ('American Call Option Price (BS): ', Option(S0_1, K, T, M, r, delta_1, sigma_1, i, 'call').value())