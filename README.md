# ppl_gnn_stocks
Final project repo for UCLA CS 267A


[colab](https://colab.research.google.com/drive/1f8UDNfQdb_fGI3jcMfwl-70rRr-aSOCA#scrollTo=c20pAMkoOo1H) # use this notebook to test functions collaboratively

[tutorials_colab](https://colab.research.google.com/drive/11kPl_81fmaIqoUH48Ozl3N83uXqL7xXO) # use this notebook for examples of GNN applications

# Target Returns

We present the target returns for two values of `skip_n_steps` because our relational tensor truncates the first `n` points for the correlations starting at `T-n` where `T` is the total number of timesteps under evaluation.

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