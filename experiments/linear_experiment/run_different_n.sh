#!/bin/zsh
SCRIPT_DIR=experiments/linear_experiment
# zsh $SCRIPT_DIR/crt_mcmc.sh $1
zsh $SCRIPT_DIR/knockoffgan_different_n.sh $1 $2
zsh $SCRIPT_DIR/deepknockoff_different_n.sh $1 $2
# zsh $SCRIPT_DIR/modelx_different_n.sh $1 $2
zsh $SCRIPT_DIR/hrt_different_n.sh $1 $2
zsh $SCRIPT_DIR/ddlk_mdnjoint_different_n.sh $1 $2
