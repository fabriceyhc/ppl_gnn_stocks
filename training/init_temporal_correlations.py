import tensorflow as tf
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm
import argparse
import os

from load_data import load_EOD_data, load_relation_data

# CLI args parser
parser = argparse.ArgumentParser(description='Initialize temporal correlational tensor for stock \
                                              price prediction.')
parser.add_argument('--market_name', default='NASDAQ', help="options: ['NASDAQ', 'NYSE']")
parser.add_argument('--corr_size', default=30, help="Input range for calculating correlations. \
                                                    Note that this reduces the effective size \
                                                    of the dataset by the selected amount.")
parser.add_argument('--data_path', default='../data/2013-01-01', help="load path for output files")
parser.add_argument('--save_path', default='../data/relation/correlations', help="save path for output")
parser.add_argument('--split', '-s', default=False, help="options: [True, False]")
args = parser.parse_args()

# params
market_name = args.market_name
corr_size = args.corr_size
data_path = args.data_path
save_path = os.path.join(args.save_path, market_name)
split = args.split

tickers_fname = market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
steps = 1

# setup save directory
if not os.path.exists(save_path):
    os.makedirs(save_path)

# load stock data
tickers = np.genfromtxt(os.path.join(data_path, '..', tickers_fname),
                              dtype=str, delimiter='\t', skip_header=False)

print('# tickers selected:', len(tickers))
eod_data, mask_data, gt_data, price_data = \
    load_EOD_data(data_path, market_name, tickers, steps)

# corr tensor
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
    if split:
        save_file_path = os.path.join(save_path, market_name + 
                                      '_correlation_init_' + str(t) + '.npy')
        np.save(save_file_path, corr[t])

if not split:
    save_file_path = os.path.join(save_path, market_name + '_correlation_init.npy')
    np.save(save_file_path, corr)

print('completed init_temporal_correlations.py')