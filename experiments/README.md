This contains scripts with the specific runs for experiments in papers.

For running the plotting scripts in R, you need a recent R installation with ```ggplot2```, ```dplyr```, ```purrr```, and ```rhdf5``` installed.

Note: To run the knockoff comparison ("knockoff_comparison.r"), you first need to run "convert_hrt_null_to_h5.py".

Example calling syntax for plots:
```
Rscript analyze_linear_comparison.R -k mcmc_n10000_s1000,ddlk,gan,deepknockoff,modelx,hrt -v fast_rf_10000,khrt_10000_fast_rf,fast_lasso_10000,khrt_10000_fast_lasso
```
