#!/bin/zsh
OUTDIR=./output/experiments
GPU=0
NRUNS=10
NOBS=$1
CUDAGPU=$2
GPU=0

export CUDA_VISIBLE_DEVICES=$CUDAGPU

for RUN in {1..$NRUNS}
do
    echo $RUN
###
# With flow-estimated density
###
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python ./flowselect/main.py --multirun gpu=\'$GPU\' \
    output=$OUTDIR \
    data=rnasq \
    data.n_threshold=$NOBS \
    dgp_response.signal_n=10\
    dgp_response.signal_a=40.0\
    data.run_id=$RUN \
    dgp_response=linear \
    joint=gaussmaf \
    joint.n_layers=5 \
    joint.lr=1e-3 \
    varsel_method=crt \
    crt.model_name=mcmc \
    crt.mcmc_steps=1000 \
    crt.n_jobs=20 \
    crt.device="cuda:$GPU" \
    crt.batch_size=1000 \
    crt.n_obs=10000\
    crt.feature_statistic=fast_lasso

MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python ./flowselect/main.py --multirun gpu=\'$GPU\' \
    output=$OUTDIR \
    data=rnasq \
    data.n_threshold=$NOBS \
    dgp_response.signal_n=10\
    dgp_response.signal_a=40.0\
    data.run_id=$RUN \
    dgp_response=sin_cos5 \
    joint=gaussmaf \
    joint.n_layers=5 \
    joint.lr=1e-3 \
    varsel_method=crt \
    crt.model_name=mcmc \
    crt.mcmc_steps=1000 \
    crt.n_jobs=20 \
    crt.device="cuda:$GPU" \
    crt.batch_size=1000 \
    crt.n_obs=10000\
    crt.feature_statistic=fast_rf
done
