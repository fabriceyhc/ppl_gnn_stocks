import torch
import numpy as np
import os

def best_return(pct_returns, daily_investment=100, skip_n_steps=0):
    """Calculates the optimal return possible for the given parameters. The output 
    reflects a policy of buying the `daily_investment` amount of the stock with 
    the highest return each timestep and selling it at the same timestep to realize 
    the return. This removes the effect of compounding returns. 

    Keyword arguments:
    pct_returns      -- np.array of dim: (num_companies, num_timesteps), containing
                        the percent returns for companies at various timesteps.
    daily_investment -- constant dollar amount to be invested at each timestep.
    skip_n_steps     -- skips the first n steps when calculating total return,
                        used to account for the correlation tensor truncation.
    """
    return np.sum(np.multiply(daily_investment, np.max(pct_returns.T, axis=1)[skip_n_steps:]))

def avg_return(pct_returns, daily_investment=100, skip_n_steps=0):
    """Calculates the average return (i.e. the return of holding the entire index).
    The output reflects a policy of buying the `daily_investment` amount of the 
    index average at each timestep and selling it at the same timestep to realize 
    the return. This removes the effect of compounding returns. 

    Keyword arguments:
    pct_returns      -- np.array of dim: (num_companies, num_timesteps), containing
                        the percent returns for companies at various timesteps.
    daily_investment -- constant dollar amount to be invested at each timestep.
    skip_n_steps     -- skips the first n steps when calculating total return,
                        used to account for the correlation tensor truncation.
    """
    return np.sum(np.multiply(daily_investment, np.average(pct_returns.T, axis=1)[skip_n_steps:]))
    
def get_dataset(price_data,n,train_size = 200, test_size = 20):
    
    start_point = n*test_size
    split_point = start_point + train_size
    final_point = split_point + test_size
    if final_point > len(price_data[0]):
        return len(price_data[0]) - start_point
    train = price_data[:,start_point:split_point]
    test = price_data[:,split_point:final_point]
    return train,test

def profit(test,truth):
    total = 0
    for i in range(0,len(test[0])-1):
        percent = (test[:,1:] - truth[:,:-1])/truth[:,:-1]
        true_percent = (truth[:,1:] - truth[:,:-1])/truth[:,:-1]
        best_buy = np.argmax(percent[:,i])
        prophet = 100*(true_percent[best_buy,i])
        total += prophet
    return total
