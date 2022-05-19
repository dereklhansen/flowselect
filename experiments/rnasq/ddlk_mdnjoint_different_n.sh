#!/bin/zsh
OUTDIR=./output/experiments
GPU=0
NRUNS=10
NOBS=$1
CUDAGPU=$2
GPU=0

export CUDA_VISIBLE_DEVICES=$CUDAGPU
#####
# Comparizon of DDLK to our method with ridge regression on true density
#####
for RUN in {1..$NRUNS}
do

MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python ./flowselect/main.py --multirun gpu=\'$GPU\' \
    output=$OUTDIR\
    data=rnasq \
    data.n_threshold=$NOBS \
    dgp_response.signal_n=10\
    dgp_response.signal_a=40.0\
    data.run_id=$RUN \
    dgp_response=linear \
    joint=ddlk \
    joint.batch_size_train=5000 \
    knockoff=ddlk \
    knockoff.batch_size_train=5000 \
    variable_selection=khrt \
    variable_selection.feature_statistic=fast_lasso \
    variable_selection.train_n_obs=10000

MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python ./flowselect/main.py --multirun gpu=\'$GPU\' \
    output=$OUTDIR\
    data=rnasq \
    data.n_threshold=$NOBS \
    dgp_response.signal_n=10\
    dgp_response.signal_a=40.0\
    data.run_id=$RUN \
    dgp_response=sin_cos5 \
    joint=ddlk \
    joint.batch_size_train=5000 \
    knockoff=ddlk \
    knockoff.batch_size_train=5000 \
    variable_selection=khrt \
    variable_selection.feature_statistic=fast_rf \
    variable_selection.train_n_obs=10000

done
