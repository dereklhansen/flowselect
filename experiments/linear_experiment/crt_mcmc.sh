#!/bin/zsh
OUTDIR=./output/experiments
GPU=0
NRUNS=10

for RUN in {1..$NRUNS}
do
    echo $RUN

###
# With flow-estimated density
###
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python ./flowselect/main.py --multirun gpu=\'$GPU\' \
    output=$OUTDIR\
    dgp_covariate=gaussian_ar_mixture\
    dgp_response.signal_n=20\
    dgp_covariate.d=100 dgp_covariate.n_obs=100000 dgp_covariate.rho_base=0.98 \
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
    crt.batch_size=1024 \
    crt.n_obs=2000,10000\
    crt.feature_statistic=fast_lasso
###
# Sin_cos
###
###
# With flow-estimated density
###
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python ./flowselect/main.py --multirun gpu=\'$GPU\' \
    output=$OUTDIR\
    dgp_covariate=gaussian_ar_mixture\
    dgp_response.signal_n=20\
    dgp_covariate.d=100 dgp_covariate.n_obs=100000 dgp_covariate.rho_base=0.98 \
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
    crt.batch_size=1024 \
    crt.n_obs=2000,10000\
    crt.feature_statistic=fast_rf
done
