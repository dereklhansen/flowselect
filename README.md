Code used in experiments for "Normalizing Flows for Knockoff-free Controlled Feature Selection" ([arXiv](https://arxiv.org/abs/2106.01528)).

To run, first make sure you have poetry installed ([https://python-poetry.org/]).

Then, initialize in the repo (one-time only)
1. ```poetry install```

Then activate poetry shell
``` poetry shell ```

The main script is called ```main.py```.
1. ```cd knockoffs/```
1. ```python main.py```

Configuration is done via Hydra command-line arguments.  See ```experiments/``` for shell scripts to recreate the experiments and plots in the paper. These scripts should be run from the top-level directory using zsh; for example,
```
zsh experiments/linear_experiment/run.sh
```

If you don't have a GPU, you can disable it by removing the arguments in each experiment script.

## Instructions for Model-X
The Model-X knockoffs in this repo are a wrapper around the R package ```knockoff```. Thus, you need a working installation with the ```knockoff``` package installed in order to use it.

## DeepKnockoffs and CVXPY
The installation of cvxpy may fail on some machiens. Only DeepKnockoffs relies on cvxpy - the rest of the models can be run without it.
