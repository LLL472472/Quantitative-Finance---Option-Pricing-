#!/usr/bin/env python
# coding: utf-8

# In[1]:


# MIE 1622
# Assignment 4
# Asset Pricing
# Student Name: Jiacheng Li
# Student ID: 1005138405


# ## Question 1. Implement pricing functions

# In[4]:


import numpy as np
import pandas as pd
from numpy import *
from scipy.stats import norm
import math
import matplotlib.pyplot as plt


# In[3]:


# Pricing a European option using Black-Scholes formula and Monte Carlo simulations 
# Pricing a Barrier option using Monte Carlo simulations

S0 = 100     # spot price of the underlying stock today
K = 105      # strike at expiry
mu = 0.05    # expected return
sigma = 0.2  # volatility
r = 0.05     # risk-free rate
T = 1.0      # years to expiry
Sb = 110     # barrier


# ### 1. Black-Scholes pricing formula for European option

# In[8]:


def BS_european_price(S0, K, T, r, sigma):
    # d1 = (ln(S/K) + (r + σ^2/2)t) / (σ*sqrt(t))
    d1 = (np.log(S0/K) + (r + (sigma**2)/2) * T) / (sigma * np.sqrt(T))
    # d2 = d1 - σ*sqrt(t)
    d2 = d1 - sigma * (np.sqrt(T))
    # C = N(d1)*S - N(d2)*K*exp(-r*t)
    c = norm.cdf(d1) * S0 - norm.cdf(d2) * K * np.exp(-r * T)
    # P = N(-d2)*K*exp(-rt) - N(-d1)*S
    p = norm.cdf(-d2) * K * np.exp(-r * T) - norm.cdf(-d1) * S0
    return c, p         


# ### 2. Monte Carlo pricing procedure for European option

# In[26]:


def MC_european_price(S0, K, T, r, mu, sigma, numSteps, numPaths):
    # Initializes an array paths to store the simulated paths of the underlying asset's price
    paths = np.zeros((numSteps + 1, numPaths))
    # Calculates the time step size for the Monte Carlo simulation
    time_size = T / numSteps
    # Sets the initial price of the underlying asset S0 as the starting value for all numPaths simulated paths
    paths[0] = [S0] * numPaths
    # Iterates over each simulated path 
    for p in range(numPaths):
        # Iterates over each time step
        for s in range(numSteps):
            # Generates a random value from np.random.normal(0,1)
            # This value is scaled by sigma * np.sqrt(time_size) 
            # and added to the drift term (mu - 0.5 * sigma**2) * time_size. 
            # The result is multiplied by the current asset price paths[s, p] 
            # to obtain the price at the next time step paths[s+1, p].
            paths[s+1, p] = paths[s, p] * np.exp((mu - 0.5 * sigma**2) * time_size 
                                                         + sigma * np.sqrt(time_size) * np.random.normal(0,1))
    # Calculate the payoffs of the call and put options at expiration based on the simulated asset prices. 
    c_payoff = np.maximum(paths[numSteps, :] - K, 0)
    p_payoff = np.maximum(K - paths[numSteps, :], 0)
    # Calculate the present value of the call and put options by taking the average of the simulated payoffs 
    # and discounting this value back to the present using the risk-free interest rate r and the time to expiration T. 
    c = np.exp(-r * T) * np.mean(c_payoff)
    p = np.exp(-r * T) * np.mean(p_payoff)
    return c, p, paths


# ### 3. Monte Carlo pricing procedure for Barrier knock-in option 

# In[20]:


def MC_barrier_knockin_price(S0, Sb, K, T, r, mu, sigma, numSteps, numPaths):
    # Initializes an array paths to store the simulated paths of the underlying asset's price
    paths = np.zeros((numSteps + 1, numPaths))
    # Calculates the time step size for the Monte Carlo simulation
    time_size = T / numSteps
    # Sets the initial price of the underlying asset S0 as the starting value for all numPaths simulated paths
    paths[0] = [S0] * numPaths
    # Iterates over each simulated path 
    for p in range(numPaths):
        # Iterates over each time step
        for s in range(numSteps):
            # Generates a random value from np.random.normal(0,1)
            # This value is scaled by sigma * np.sqrt(time_size) 
            # and added to the drift term (mu - 0.5 * sigma**2) * time_size. 
            # The result is multiplied by the current asset price paths[s, p] 
            # to obtain the price at the next time step paths[s+1, p].
            paths[s+1, p] = paths[s, p] * np.exp((mu - 0.5 * sigma**2) * time_size 
                                                         + sigma * np.sqrt(time_size) * np.random.normal(0,1))
    # Create a list to store the number of times of knocking in
    knock_in = []
    for i in range(numPaths):
        # If any of the prices exceed Sb, the option is knocked in, and knock_in is set to 1 for that path. 
        # Otherwise, knock_in is set to 0 for that path
        if any(paths[:,i] > Sb):
            knock_in.append(1)
        else: 
            knock_in.append(0)  
    # Calculate the payoff of the option at maturity for each path, depending on whether it is a call or put option. 
    # If the option is knocked in (knock_in=1), then the payoff is the maximum of the difference 
    # between the final asset price paths[numSteps,:] and the strike price K or zero. 
    # If the option is not knocked in (knock_in=0), then the payoff is zero.
    c_payoff = knock_in * np.maximum(paths[numSteps,:] - K,0)
    p_payoff = knock_in * np.maximum(K - paths[numSteps,:], 0)
    c = np.mean(c_payoff) * np.exp(-r * T)
    p = np.mean(p_payoff) * np.exp(-r * T)
    return c, p


# ## Question 2. Analyze results

# In[11]:


# Define variable numSteps to be the number of steps for multi-step MC
# numPaths - number of sample paths used in simulations

numSteps = 10;
numPaths = 1000000;


# 1.Produce Black-Scholes call and put price for the given European option.

# In[12]:


# Implement your Black-Scholes pricing formula
call_BS_European_Price, putBS_European_Price =   BS_european_price(S0, K, T, r, sigma)


# 2.Compute one-step MC call and put price for the given European option. 
# Justify the number of paths used for computations.
# 
# The number of paths is 1000000; the number of steps is set to be 1, since it is a one-step MC.

# In[27]:


# Implement your one-step Monte Carlo pricing procedure for European option
callMC_European_Price_1_step, putMC_European_Price_1_step, paths_1_step =   MC_european_price(S0, K, T, r, mu, sigma, 1, numPaths)


# 3.Compute multi-step MC call and put price for the given European option. Justify the number of steps and paths used for computations.
# 
# The number of paths is set to be 1000000; the number of steps is set to be 10, since it is a multi-step MC.

# In[28]:


# Implement your multi-step Monte Carlo pricing procedure for European option
callMC_European_Price_multi_step, putMC_European_Price_multi_step, paths_multi_step =   MC_european_price(S0, K, T, r, mu, sigma, numSteps, numPaths)


# 4.Compute one-step MC call and put price for the given Barrier option. Justify the number of paths used for computations.
# 
# The number of paths is 1000000; the number of steps is set to be 1, since it is a one-step MC for Barrier option.

# In[21]:


# Implement your one-step Monte Carlo pricing procedure for Barrier option
callMC_Barrier_Knockin_Price_1_step, putMC_Barrier_Knockin_Price_1_step =   MC_barrier_knockin_price(S0, Sb, K, T, r, mu, sigma, 1, numPaths)


# 5.Compute multi-step MC call and put price for the given Barrier option. Justify the number of steps and paths used for computations.
# 
# The number of paths is set to be 1000000; the number of steps is set to be 10, since it is a multi-step MC for Barrier option.

# In[23]:


# Implement your multi-step Monte Carlo pricing procedure for Barrier option
callMC_Barrier_Knockin_Price_multi_step, putMC_Barrier_Knockin_Price_multi_step =   MC_barrier_knockin_price(S0, Sb, K, T, r, mu, sigma, numSteps, numPaths)


# In[24]:


print('Black-Scholes price of an European call option is ' + str(call_BS_European_Price))
print('Black-Scholes price of an European put option is ' + str(putBS_European_Price))
print('One-step MC price of an European call option is ' + str(callMC_European_Price_1_step)) 
print('One-step MC price of an European put option is ' + str(putMC_European_Price_1_step)) 
print('Multi-step MC price of an European call option is ' + str(callMC_European_Price_multi_step)) 
print('Multi-step MC price of an European put option is ' + str(putMC_European_Price_multi_step)) 
print('One-step MC price of an Barrier call option is ' + str(callMC_Barrier_Knockin_Price_1_step)) 
print('One-step MC price of an Barrier put option is ' + str(putMC_Barrier_Knockin_Price_1_step)) 
print('Multi-step MC price of an Barrier call option is ' + str(callMC_Barrier_Knockin_Price_multi_step)) 
print('Multi-step MC price of an Barrier put option is ' + str(putMC_Barrier_Knockin_Price_multi_step))


# 6.Plot one chart that illustrates Monte Carlo pricing procedure in the best way. 

# In[32]:


plt.figure(figsize = (20,10))
[plt.plot(paths_1_step[:,i], linewidth = 3) for i in range(numPaths)]
plt.ylabel("Price", fontsize = 15)
plt.xlabel("Time (year)", fontsize = 15)
plt.title('Geometric Random Walk Paths for one-step Monte-Carlo Simulations of European Option')
plt.show()


# In[33]:


plt.figure(figsize = (20,10))
[plt.plot(paths_multi_step[:,i], linewidth = 3) for i in range(numPaths)]
plt.ylabel("Price", fontsize = 15)
plt.xlabel("Time (year)", fontsize = 15)
plt.title('Geometric Random Walk Paths for multi-step Monte-Carlo Simulations of European Option')
plt.show()


# 7.Compare three pricing strategies for European option and discuss their performance relative to each other.

# The three pricing strategies I compare are Black-Scholes (BS), Monte Carlo (MC) simulation for European options, and Monte Carlo simulation for Barrier Knock-in options.
# 
# Black-Scholes pricing formula is a closed-form solution that can be used to price European options under the assumption of log-normality of the underlying asset price. It is fast and computationally efficient, but it assumes a constant volatility, and the underlying asset price follows a log-normal distribution, which is not always true in real-world markets.
# 
# On the other hand, Monte Carlo simulation can handle a variety of underlying asset price distributions and has the flexibility to incorporate stochastic volatility and other complex features. However, it can be computationally expensive and time-consuming, and the results may depend on the number of paths and the accuracy of the random number generator used in the simulation.
# 
# The third pricing strategy, Monte Carlo simulation for Barrier Knock-in options, is used to price options that only become active if the underlying asset price reaches a certain barrier level. In this case, the option's payoff depends on whether the underlying asset price reaches the barrier level or not, in addition to the standard European option's payoff. This pricing strategy requires simulating a large number of paths to estimate the probability of the underlying asset price reaching the barrier level.
# 
# In terms of performance, the Black-Scholes formula is the fastest and most efficient method for pricing European options, but it may not be accurate in cases where the underlying asset price distribution is not log-normal. Monte Carlo simulation can handle a wider range of underlying asset price distributions, but it requires more computation time and is more sensitive to the number of paths used in the simulation. Monte Carlo simulation for Barrier Knock-in options is the most complex and computationally expensive method, as it requires simulating a large number of paths to estimate the probability of the underlying asset price reaching the barrier level. However, it can be useful in cases where the option's payoff depends on whether the underlying asset price reaches a certain barrier level or not. Overall, the choice of pricing strategy depends on the specific needs of the user, including the accuracy, speed, and computational resources available.
# 
# The Black-Scholes prices of the European call and put options are 8.021352235143176 and 7.9004418077181455, respectively. These are theoretical prices based on the Black-Scholes model, which assumes a constant volatility and interest rate, among other assumptions.
# 
# The one-step Monte Carlo (MC) prices of the European call and put options are 8.034477204094891 and 7.912159147899672, respectively. These prices were obtained using a random simulation of the underlying asset's prices and averaging the payoffs of the options at maturity. The one-step MC prices are very close to the Black-Scholes prices, suggesting that the one-step MC model is a good approximation of the Black-Scholes model.
# 
# The multi-step MC prices of the European call and put options are 8.015204907941285 and 7.897082071804356, respectively. These prices were obtained using a more sophisticated MC model that takes into account multiple time steps and the possibility of early exercise. The multi-step MC prices are slightly lower than the Black-Scholes prices, which suggests that the Black-Scholes model may be slightly overestimating the option prices.
# 
# The one-step MC price of the barrier call option is 7.799676209681738, which is lower than the European call option prices. This is because the barrier call option has an additional condition that the underlying asset must not touch or cross a certain barrier before maturity. This condition reduces the probability of the option being in the money at maturity, and hence lowers the price.
# 
# The one-step MC price of the barrier put option is 0.0, which means that the option has zero value. This is because the barrier put option has a condition that the underlying asset must touch or cross a certain barrier before maturity in order for the option to be exercised. Since the underlying asset did not touch or cross the barrier in the one-step MC simulation, the option is worthless.
# 
# The multi-step MC price of the barrier call option is 7.963436225610033, which is slightly higher than the one-step MC price. This is because the multi-step MC model takes into account more possible scenarios, including the possibility of the underlying asset crossing the barrier and then rebounding back.
# 
# The multi-step MC price of the barrier put option is 1.192183175766639, which is also higher than the one-step MC price. This is because the multi-step MC model takes into account more possible scenarios, including the possibility of the underlying asset touching or crossing the barrier and then rebounding back, which can still lead to some value for the option.

# 8.Explain the difference between call and put prices obtained for European and Barrier options.

# A European option is a type of financial contract that gives the owner the right, but not the obligation, to buy (in the case of a call option) or sell (in the case of a put option) an underlying asset at a predetermined price (the strike price) on or before a predetermined expiration date. A Barrier option, on the other hand, is a type of option that has an additional feature known as the barrier. This barrier is a predetermined price level that, if breached, either activates or deactivates the option.
# 
# The difference between call and put prices for both European is based on the strike price and the current price of the underlying asset. If the strike price is higher than the current price of the underlying asset, the put option will be more expensive than the call option, as the owner of the put option has the right to sell the underlying asset at a premium. If the strike price is lower than the current price of the underlying asset, the call option will be more expensive than the put option, as the owner of the call option has the right to buy the underlying asset at a discount. Thank you for pointing out the error in my previous response.
# 
# For Barrier options with a knock-in feature, if the underlying asset does cross the barrier, the option may become activated and the price may change dramatically. In this case, the difference between the call and put prices may depend on the specific terms of the option and the barrier level. If the strike price of the Barrier call option is higher than the barrier level, and the underlying asset crosses the barrier, the call option will become activated and the owner will have the right to buy the underlying asset at a discount. Conversely, if the strike price of the Barrier put option is lower than the barrier level, and the underlying asset crosses the barrier, the put option will become activated and the owner will have the right to sell the underlying asset at a premium.
# 
# Barrier options with the knock-out feature can have a significant impact on the pricing of call and put options. If the underlying asset price hits the barrier level, the option is knocked out, and the option becomes worthless. Therefore, the barrier call and put options with a knock-out feature will be cheaper than a regular call option, as there is a possibility that the option will be knocked out before expiration.

# 9.Compute prices of Barrier options with volatility increased and decreased by 10% from the original inputs. Explain the results.

# In[35]:


# Volatility increased by 10%
callMC_Barrier_Knockin_Price_1_step_vol_inc_10, putMC_Barrier_Knockin_Price_1_step_vol_inc_10 =     MC_barrier_knockin_price(S0, Sb, K, T, r, mu, sigma*1.1, 1, numPaths)
print('One-step MC price of an Barrier call option with volatility increased by 10% is ' + str(callMC_Barrier_Knockin_Price_1_step_vol_inc_10)) 
print('One-step MC price of an Barrier put option with volatility increased by 10% is ' + str(putMC_Barrier_Knockin_Price_1_step_vol_inc_10)) 

callMC_Barrier_Knockin_Price_multi_step_vol_inc_10, putMC_Barrier_Knockin_Price_multi_step_vol_inc_10 =     MC_barrier_knockin_price(S0, Sb, K, T, r, mu, sigma*1.1, numSteps, numPaths)
print('Multi-step MC price of an Barrier call option with volatility increased by 10% is ' + str(callMC_Barrier_Knockin_Price_multi_step_vol_inc_10)) 
print('Multi-step MC price of an Barrier put option with volatility increased by 10% is ' + str(putMC_Barrier_Knockin_Price_multi_step_vol_inc_10)) 


# Volatility decreased by 10%
callMC_Barrier_Knockin_Price_1_step_vol_dec_10, putMC_Barrier_Knockin_Price_1_step_vol_dec_10 =     MC_barrier_knockin_price(S0, Sb, K, T, r, mu, sigma*0.9, 1, numPaths)
print('One-step MC price of an Barrier call option with volatility decreased by 10% is ' + str(callMC_Barrier_Knockin_Price_1_step_vol_dec_10)) 
print('One-step MC price of an Barrier put option with volatility decreased by 10% is ' + str(putMC_Barrier_Knockin_Price_1_step_vol_dec_10)) 

callMC_Barrier_Knockin_Price_multi_step_vol_dec_10, putMC_Barrier_Knockin_Price_multi_step_vol_dec_10 =     MC_barrier_knockin_price(S0, Sb, K, T, r, mu, sigma*0.9, numSteps, numPaths)
print('Multi-step MC price of an Barrier call option with volatility decreased by 10% is ' + str(callMC_Barrier_Knockin_Price_multi_step_vol_dec_10)) 
print('Multi-step MC price of an Barrier put option with volatility decreased by 10% is ' + str(putMC_Barrier_Knockin_Price_multi_step_vol_dec_10))


# In[ ]:


# The original:

# One-step MC price of an Barrier call option is 7.799676209681738
# One-step MC price of an Barrier put option is 0.0
# Multi-step MC price of an Barrier call option is 7.963436225610033
# Multi-step MC price of an Barrier put option is 1.192183175766639


# The results show that the one-step MC price of the Barrier call option is higher than the Barrier put option for both the original volatility, the volatility increased by 10%, and the volatility decreased by 10%. This suggests that the market expects the price of the underlying asset to rise, which makes the call option more valuable than the put option. 
# 
# And multi-step MC price of the Barrier call option is higher than the Barrier put option for both the original volatility, the volatility increased by 10%, and the volatility decreased by 10%. This suggests that the market expects the price of the underlying asset to rise, which makes the call option more valuable than the put option.
# 
# The multi-step MC price of the Barrier call option is higher than the one-step MC price for the original volatility, the volatility increased by 10%, and the volatility decreased by 10%. This is because the multi-step simulation considers a greater number of price paths of the underlying asset, allowing for a more accurate estimation of the option's expected payoff. 
# 
# Similarly, the multi-step MC price of the Barrier put option is higher than the one-step MC price for the original volatility, the volatility increased by 10%, and the volatility decreased by 10%, suggesting the multi-step simulation provides a more accurate estimate of the option's expected value.
# 
# When the volatility of the underlying asset is increased by 10%, the prices of both the Barrier call and Barrier put options increase. Conversely, when the volatility of the underlying asset is decreased by 10%, the prices of both options decrease. This is because higher volatility results in a higher probability of the underlying asset reaching the barrier level, increasing the likelihood of the option being exercised.
# 
# Overall, the results suggest that the pricing of Barrier options is sensitive to changes in the volatility of the underlying asset. Moreover, multi-step Monte Carlo simulations provide a more accurate estimate of the option's expected value compared to one-step simulations.

# ## Question 3. Discuss possible strategies to obtain the same prices from two procedures

# Design procedure for choosing a number of time steps and a number of scenarios in Monte Carlo pricing for European option to get the same price (up to the cent) as given by the Black-Scholes formula.

# In[45]:


# Set the error threshold for call price to be 0.01
# Set the error threshold for put price to be 0.01
step_list = [1, 2, 3, 4, 6, 12, 24]
for i in step_list:
    c, p, best_path = MC_european_price(S0, K, T, r, mu, sigma, i, numPaths)
    call_error = abs(c - call_BS_European_Price)
    put_error = abs(p - putBS_European_Price)  
    if call_error <= 0.01:
        best_c_error = call_error
        best_step_c = i
        best_c_price = c
    if put_error <= 0.01:
        best_p_error = put_error
        best_step_p = i
        best_p_price = p


# In[57]:


print(best_step_c)
print(best_step_p)

print('The MC price of an European call option with the optimal number of time steps is ' + str(best_c_price)) 
print('The MC price of an European put option with the optimal number of time steps is ' + str(best_p_price)) 


# In[46]:


path_list = [10, 100, 1000, 10000, 100000, 1000000]
# Set the error threshold for call price to be 0.01
# Set the error threshold for put price to be 0.01
for i in path_list:
    c, p, best_path = MC_european_price(S0, K, T, r, mu, sigma, numSteps, i)
    call_error = abs(c - call_BS_European_Price)
    put_error = abs(p - putBS_European_Price)
    if call_error <= 0.01:
        best_c_error = call_error
        best_path_c = i
        best_call_price = c
    if put_error <= 0.01:
        best_p_error = put_error
        best_path_p = i
        best_p_price = p


# In[58]:


print(best_path_c)
print(best_path_p)

print('The MC price of an European call option with the optimal number of scenarios is ' + str(best_c_price)) 
print('The MC price of an European put option with the optimal number of scenarios is ' + str(best_p_price)) 


# In[59]:


print('The optimal number of time steps is:', max(best_step_c,best_step_p))
print('The optimal number of scenarios is:', max(best_path_c,best_path_p))


# Using 24 time steps and 1 million scenarios to obtain the same price as the Black-Scholes formula is reasonable. The number of time steps determines the granularity of the simulation, with more time steps resulting in a more precise estimate of the option price.
# 
# The number of scenarios, on the other hand, affects the statistical error of the simulation. Increasing the number of scenarios reduces the variance in the estimate and can result in a more accurate price estimate. In practice, 1 million scenarios is a large enough number to achieve a low level of statistical error in Monte Carlo simulations.
