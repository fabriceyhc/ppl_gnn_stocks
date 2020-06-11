import tensorflow as tf
import numpy as np
import os
from scipy.stats import pearsonr
from tqdm import tqdm
import pickle

from load_data import load_EOD_data, load_relation_data

import argparse

parser = argparse.ArgumentParser(description='Initialize temporal correlational tensor for stock price prediction.')
parser.add_argument('-market_name', '-m', default='NASDAQ', help="options: ['NASDAQ', 'NYSE']")
args = parser.parse_args()

# params
data_path = '../data/2013-01-01'
market_name = args.market_name
tickers_fname = market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
steps = 1

# load stock data
tickers = np.genfromtxt(os.path.join(data_path, '..', tickers_fname),
                              dtype=str, delimiter='\t', skip_header=False)

print('#tickers selected:', len(tickers))
eod_data, mask_data, gt_data, price_data = \
    load_EOD_data(data_path, market_name, tickers, steps)

# corr tensor
corr_size = 30
num_companies, num_timesteps = gt_data.shape
correlation_matrix_shape = (num_timesteps - corr_size, num_companies, num_companies)
corr = np.ones(correlation_matrix_shape)
print(corr.shape)

iu = np.triu_indices(num_companies,k=1)
il = (iu[1],iu[0])

for t in tqdm(range(num_timesteps - corr_size)):
    for c1 in tqdm(range(num_companies)):
        for c2 in range(1, num_companies - c1):
            c1_movements = gt_data.T[t:t+corr_size][:,c1]
            c2_movements = gt_data.T[t:t+corr_size][:,c1 + c2]
            c1_c2_corr = pearsonr(c1_movements, c2_movements)[0]
            if np.isnan(c1_c2_corr):
                c1_c2_corr = 1e-10
            corr[t][c1][c1 + c2] = c1_c2_corr        
    corr[t][il] = corr[t][iu]

# save
save_file_path = '../data/correlation_init.pkl'
with open(save_file_path, 'wb') as handle:
    pickle.dump(corr, handle, protocol=pickle.HIGHEST_PROTOCOL)