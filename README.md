# ppl_gnn_stocks
Final project repo for UCLA CS 267A


[colab](https://colab.research.google.com/drive/1f8UDNfQdb_fGI3jcMfwl-70rRr-aSOCA#scrollTo=c20pAMkoOo1H) # use this notebook to test functions collaboratively

[tutorials_colab](https://colab.research.google.com/drive/11kPl_81fmaIqoUH48Ozl3N83uXqL7xXO) # use this notebook for examples of GNN applications

# Datasets

## Relational Data

The current Wiki & Industry relation matrices from [TRSR](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking) must be extracted with the following command within the `data` directory:

```
tar zxvf relation.tar.gz
```

```
data
     relation
          correlations_trained
               NASDAQ
                    ...1215 files uploaded
               NYSE
                    ...1215 files not uploaded (too big)
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

# Run

Below are the commands used to generate the results for this project.

```
python pytorch_relational_rank_model.py -m "NASDAQ" -rn "sector_industry" -ep 100 -up 0
python pytorch_relational_rank_model.py -m "NASDAQ" -rn "wikidata" -ep 100 -up 0
python pytorch_relational_rank_model.py -m "NASDAQ" -rn "correlational" -ep 100 -up 0

python pytorch_relational_rank_model.py -m "NYSE" -rn "sector_industry" -ep 100 -up 0 -u 32
python pytorch_relational_rank_model.py -m "NYSE" -rn "wikidata" -ep 100 -up 0 -u 32
python pytorch_relational_rank_model.py -m "NYSE" -rn "correlational" -ep 100 -up 0 -u 32
```

Note that training works using rolling windows --- `train_size=200, val_size=20, test_size=20` --- and the number of windows is dynamically calculated by `num_steps \ train_size`. This results in each timestep being included in no more than 1 sliding window for `ep=100` epochs each. 
