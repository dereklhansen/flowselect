library(tidyverse)
library(stringi)
root <- "output/experiments/"

dataset_tbl <- tribble(
    ~dataset, ~n_obs, ~path,
    "gaussian", 1000, "gaussian_ar_mixture_linear_d100_n1000_rho0.98_",
    "gaussian", 10000, "gaussian_ar_mixture_linear_d100_n10000_rho0.98_",
    "gaussian", 100000, "gaussian_ar_mixture_linear_d100_n100000_rho0.98_",
    "rnasq", 1000, "rnasq_cor_linear_a40_n10_nv100_nvc5_nt1000_",
    "rnasq", 10000, "rnasq_cor_linear_a40_n10_nv100_nvc5_nt10000_",
    "rnasq", 100000, "rnasq_cor_linear_a40_n10_nv100_nvc5_"
)

read_runs <- function(path) {
    runs <- 1:10
    csv_paths <- root %s+% path %s+% sprintf("/run%03d/gaussmaf_5/valid_loglik.csv", runs)
    map(csv_paths, read_csv) %>% 
        bind_rows(.id="run")
}

loglik_tbl <- plyr::adply(dataset_tbl, 1, function(x) read_runs(x$path)) %>% tibble

loglik_tbl %>% group_by(dataset, n_obs) %>% summarize(loglik_mean = mean(loglik_mean))
