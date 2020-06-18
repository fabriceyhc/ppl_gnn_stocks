# ppl_gnn_stocks
Final project repo for UCLA CS 267A

[scratch_pad.ipynb (colab)](https://colab.research.google.com/drive/1BYBE7WeGu4jv2cFyD6LZR5u5ToVV1cok?usp=sharing) # use this notebook to test functions collaboratively

[gnn_tutorial.ipynb (colab)](https://colab.research.google.com/drive/11kPl_81fmaIqoUH48Ozl3N83uXqL7xXO) # use this notebook for examples of GNN applications

[pyro_stocks.ipynb (colab)](https://colab.research.google.com/drive/1f8UDNfQdb_fGI3jcMfwl-70rRr-aSOCA#scrollTo=c20pAMkoOo1H) # use this notebook to check out stock modeling with Pyro

# Datasets

## Relational Data

The current Wiki & Industry relation matrices from [TRSR](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking) must be extracted with the following command within the `data` directory:

```
tar zxvf relation.tar.gz
```

```
data
     relation
          sector_industry
               NASDAQ_industry_ticker.json
               NYSE_industry_ticker.json
               NASDAQ_industry_relation.npy
               NYSE_industry_relation.npy
          wikidata
               NASDAQ_wiki_relation.npy
               NYSE_wiki_relation.npy
               NYSE_connections.json
               NASDAQ_connections.json
               selected_wiki_connections.csv
```

In order to initialize our temporal relational tensors, you'll need to execute this command from the `training` directory, passing either "NYSE" or "NASDAQ" as an argument:

```
python init_temporal_correlations.py -market_name "NASDAQ"
```

NOTE: This will take a long time to execute and will result in massive tensors (~8hrs + 10GB for NASDAQ, ~24hrs + 29GB for NYSE).

## High-level Statistics

| index  | num_companies | num_timesteps (T) | 
|--------|---------------|-------------------|
| NASDAQ | 1026          | 1245              | 
| NYSE   | 1737          | 1245              | 

# Pretrained Weights

The pretrained weights were too large to store in this repository, but they can be accessed [here](https://drive.google.com/file/d/1HpAsHH4oGdLrWeOby17pjVv3uIMe1TGh/view) or [here](https://drive.google.com/file/d/1fyNCZ62pEItTQYEBzLwsZ9ehX_-Ai3qT/view).

Extract the file into the `data` directory:

```
tar zxvf pretrain.tar.gz
```

```
data
     pretrain
          NASDAQ_rank_lstm_seq-16_unit-64_2.csv.npy
          NYSE_rank_lstm_seq-8_unit-32_0.csv.npy
```

# Target Returns

We present the target returns for two values of `skip_n_steps` because our relational tensor truncates the first `n` points for the correlations starting at `T-n` where `T` is the total number of timesteps under evaluation. The amounts reflect a policy of buying the `daily_investment` amount of the target stock(s) at each timestep and selling them at the same timestep. This removes the effect of compounding returns. 

## Optimal

| skip_n_steps | daily_investment | NASDAQ | NYSE |
|---|---|---|---|
| 0 | 100.00 | 20102.418 | 16884.516 | 
| 30 | 100.00 | 19622.406 | 16482.426 |

## Average

| skip_n_steps | daily_investment | NASDAQ | NYSE |
|---|---|---|---|
| 0 | 100.00 | 57.427822 | 35.447433 | 
| 30 | 100.00 | 55.723846 | 31.19346 |
