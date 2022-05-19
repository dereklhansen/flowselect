#!/bin/zsh
OUTDIR=./output/experiments
NRUNS=10
NOBS=$1
CUDAGPU=$2
GPU=0

export CUDA_VISIBLE_DEVICES=$CUDAGPU


for RUN in {1..$NRUNS}
do

###
# Deep knockoffs
###

MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python ./flowselect/main.py --multirun \
    output=$OUTDIR\
    dgp_covariate=gaussian_ar_mixture\
    dgp_response.signal_n=20\
    dgp_covariate.d=100 dgp_covariate.n_obs=$NOBS dgp_covariate.rho_base=0.98 \
    data.run_id=$RUN \
    dgp_response=linear \
    joint=truth \
    knockoff=deepknockoff \
    variable_selection=khrt \
    variable_selection.feature_statistic=fast_lasso \
    variable_selection.train_n_obs=10000 \
    variable_selection.test_n_obs=2000

MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python ./flowselect/main.py --multirun \
    output=$OUTDIR\
    dgp_covariate=gaussian_ar_mixture\
    dgp_response.signal_n=20\
    dgp_covariate.d=100 dgp_covariate.n_obs=$NOBS dgp_covariate.rho_base=0.98 \
    data.run_id=$RUN \
    dgp_response=sin_cos5 \
    joint=truth \
    knockoff=deepknockoff \
    variable_selection=khrt \
    variable_selection.feature_statistic=fast_rf \
    variable_selection.train_n_obs=10000 \
    variable_selection.test_n_obs=2000
done

