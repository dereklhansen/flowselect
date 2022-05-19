#!/bin/zsh
OUTDIR=./output/experiments
# GPU=1
TRAIT=$1
GPU=$2
MCMC=$3

echo "Running FlowSelect on Soybean Data for trait=$TRAIT"
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python ./flowselect/main.py --multirun gpu=\'$GPU\' \
    output=$OUTDIR \
    data=soybean \
    data.trait=$TRAIT \
    joint=discrete_flow \
    varsel_method=crt \
    crt.model_name=mcmc \
    crt.mcmc_steps=$MCMC \
    crt.feature_statistic=fast_nn \
    crt.n_jobs=20 \
    crt.device="cuda:$GPU" \
    crt.batch_size=1000 \
    crt.n_obs=10000
