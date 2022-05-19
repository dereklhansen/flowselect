#!/bin/zsh
OUTDIR=./output/experiments
GPU=0
NRUNS=10
for RUN in {1..$NRUNS}
do
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python ./flowselect/main.py --multirun \
    output=$OUTDIR\
    dgp_covariate=gaussian_ar_mixture\
    dgp_response.signal_n=20\
    dgp_covariate.d=100 dgp_covariate.n_obs=100000 dgp_covariate.rho_base=0.98 \
    data.run_id=$RUN \
    dgp_response=linear \
    joint=truth \
    knockoff=modelx \
    variable_selection=khrt \
    variable_selection.feature_statistic=fast_ridge,fast_rf,fast_lasso \
    variable_selection.train_n_obs=10000

MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python ./flowselect/main.py --multirun \
    output=$OUTDIR\
    dgp_covariate=gaussian_ar_mixture\
    dgp_response.signal_n=20\
    dgp_covariate.d=100 dgp_covariate.n_obs=100000 dgp_covariate.rho_base=0.98 \
    data.run_id=$RUN \
    dgp_response=sin_cos5 \
    joint=truth \
    knockoff=modelx \
    variable_selection=khrt \
    variable_selection.feature_statistic=fast_rf \
    variable_selection.train_n_obs=10000
done
