#!/bin/zsh
OUTDIR=./output/experiments
GPU=0
NRUNS=10

for RUN in {1..$NRUNS}
do
    echo $RUN

    MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python ./flowselect/main.py --multirun gpu=\'$GPU\' \
        output=$OUTDIR\
        dgp_covariate=gaussian_ar_mixture\
        dgp_response.signal_n=20\
        dgp_covariate.d=100 dgp_covariate.n_obs=100000 dgp_covariate.rho_base=0.98 \
        data.run_id=$RUN \
        dgp_response=linear \
        joint=truth \
        varsel_method=hrt \
        hrt.feature_statistic=fast_ridge,fast_rf,fast_lasso\
        hrt.n_obs=10000 \
        hrt.nbootstraps=1 \
        hrt.n_jobs=20

    MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python ./flowselect/main.py --multirun gpu=\'$GPU\' \
        output=$OUTDIR\
        dgp_covariate=gaussian_ar_mixture\
        dgp_response.signal_n=20\
        dgp_covariate.d=100 dgp_covariate.n_obs=100000 dgp_covariate.rho_base=0.98 \
        data.run_id=$RUN \
        dgp_response=sin_cos5 \
        joint=truth \
        varsel_method=hrt \
        hrt.feature_statistic=fast_rf\
        hrt.n_obs=10000 \
        hrt.nbootstraps=1 \
        hrt.n_jobs=20
done
