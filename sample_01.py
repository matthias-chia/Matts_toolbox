#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 00:16:17 2021

@author: mchh
"""


#Package import
import numpy as np #to work with multidimensional arrays
import pandas as pd #PANel DAta, organise arrays in tables and attach descriptive labels, time series and big database
import matplotlib.pyplot as plt #2D plotting visuals of numpy computation
from pandas_datareader import data as wb
from scipy.stats import norm
#%matplotlib inline

#Data import
#initialise tickers
ticker_1 = 'TSLA'
#ticker_2 = '^GSPC'
#ticker_3 = 'PFE'
#ticker_4 = 'XOM'

#ticker in data frame
assets = [ticker_1] #, ticker_2, ticker_3, ticker_4]
data = pd.DataFrame()

for a in assets:
    data[a] = wb.DataReader(a, data_source = 'yahoo', start = '2018-1-1')['Adj Close']
#data.tail

#Calculate historical model
log_returns = np.log(1+data.pct_change()) #pct_change() obtains simple returns from a provided dataset

tickerMu = log_returns.mean()
print('Ticker mean return price is: (US$)', tickerMu)

tickerVariance = log_returns.var()
print('Ticker return price cariance is: (US$)', tickerVariance)

#log_returns.tail
data.plot(figsize=(15,6))
plt.grid(alpha=0.5)
plt.ylabel("Price(US$)")

log_returns.plot(figsize=(15,6))
plt.grid(alpha=0.5)
plt.ylabel("log returns(trading days)")


#Drift
drift = tickerMu - (0.5*tickerVariance)
drift

# Historical Volatility
stdev = log_returns.std()
stdev

#Check data type = series and convert to arrays
print(type(drift))
print(type(stdev))

np.array(drift) #or drift.values
np.array(stdev) #or stdev.values

Z = norm.ppf(np.random.rand(10,2)) #norm.ppf converts the % into standard deviation #np.random.rand(10,2) creates a 10x2 matrix 
Z
t_intervals = 1000 #next 1000 evolutions
iterations = 100 #10 iterations of the 1000evolutions

daily_returns = np.exp(drift.values + stdev.values * norm.ppf(np.random.rand(t_intervals, iterations))) #what is the matrix multiplication here?
daily_returns #1000X10 array

#last extracted stock price to start movement of daily returns
S0 = data.iloc[-1]
S0

#Create data array loop randomised data and initialise start data
price_list = np.zeros_like(daily_returns) # creates an array of similiar size to another array e.g. daily returns

price_list[0] = S0
#price_list

for t in range(1, t_intervals):
    price_list[t] = price_list[t - 1] * daily_returns[t]
    
plt.figure(figsize=(15,6))
plt.ylabel("Simulated stock price movement using brownian motion")
plt.xlabel("Iterations")
plt.grid() 
plt.plot(price_list);

#Define bin size
k = 999
plt.figure(figsize=(15,6))
plt.hist(price_list[k], bins = 100)
plt.xlabel("Monte-Carlo Sim: security distribution @ k-th iteration, k = "+ str(k+1))
plt.ylabel('Count')
plt.show