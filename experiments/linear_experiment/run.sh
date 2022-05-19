#!/bin/zsh
SCRIPT_DIR=experiments/linear_experiment
zsh $SCRIPT_DIR/crt_mcmc.sh
zsh $SCRIPT_DIR/ddlk_mdnjoint.sh
zsh $SCRIPT_DIR/hrt.sh
# zsh $SCRIPT_DIR/ddlk.sh
zsh $SCRIPT_DIR/knockoffgan.sh
zsh $SCRIPT_DIR/deepknockoff.sh
zsh $SCRIPT_DIR/modelx.sh
# zsh $SCRIPT_DIR/crt_mcmc_lasso.sh
