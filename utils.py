import numpy as np

def best_return(pct_returns, daily_investment=100, skip_n_steps=0):
    """Calculates the optimal return possible for the given parameters. 

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

    Keyword arguments:
    pct_returns      -- np.array of dim: (num_companies, num_timesteps), containing
                        the percent returns for companies at various timesteps.
    daily_investment -- constant dollar amount to be invested at each timestep.
    skip_n_steps     -- skips the first n steps when calculating total return,
                        used to account for the correlation tensor truncation.
    """
    return np.sum(np.multiply(daily_investment, np.average(pct_returns.T, axis=1)[skip_n_steps:]))