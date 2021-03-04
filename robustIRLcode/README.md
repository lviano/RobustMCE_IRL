# Robust Inverse Reinforcement Learning Under Transition Dynamic Mismatch

This is the code implementing the experiments in the paper.

## Getting Started

### Prerequisites

You need a recent python installation (we used python 3.7). The required packages are Numpy, Matplotlib, Pytorch and dependencies.

In addition, we saved the results using Pickle.

### Reproducing the results

Considering that our comparison plots relies on many execution of IRL on different combination of the parameters we parallelized the execution using `run_experiment.py`. See section command line interface for the available options.

Once successfully executed, the scripts `CompareAlphas.py`,`CompareAlphas_ow.py` or `CompareAlphas_tdw.py` generates the plot shown in the paper for the 3 considered environments.

### `run_experiment.py` command line interface

`run_experiment.py` takes the following arguments:

* `--script_name`: `.sh` file to parallelize the jobs 
* `--logs_folder`: path to the log folder 
* `--job_name`: job name
* `--mode`: it allows to select one of the different configurations of the environment. For GridWorld the available options are `0, 1, 2, 9, 10`. For TwoDangerWorld it is `1` while for ObjectWorld this arguments has no impact.
* `--lr`: learning rates for the reward update at each iteration of the IRL algorithm
* `--softQ_lr`: learning rate for the TD updates in the model free soft Q learning scheme.
* `--dim`: side dimension of the considered Gridworld
* `--seed`: Fix the random seed. The default value is used to reproduce the result
* `--beta`: Inverse Temperature for the player soft policy. 
* `--beta_op`: Inverse Temperature for the opponent soft policy
* `--alphaE`: Values for the expert parameter `alpha` that determines the opponent power within the two players scheme
* `--alphaL`: Values for the learner parameter `alpha` that determines the opponent power within the two players scheme
* `--noiseE`: Environment Noises for the expert MDP
* `--noiseL`: Environment Noises for the learner MDP
* `--gw`: `True` to use the GridWorld environment, `False` to use the ObjectWorld environment.
* `--tdw`: `True` to use the Two Danger environment.
* `--linear`: `True` to use a linear combination of the feature to represent the reward function. `False` to use a neural-network based representation of the reward function as a function of the environment features.
* `--reg_opp`: `True` to use the use a tunable regularization for player and opponent as determined by the arguments `beta` and `beta_opp`.
* `--fix_start`: `True` to select as start state the bottom right corner of the GridWorld, `False` to use a uniform distribution over the state space as initial probability distribution.
* `--fix_horizon`: For the ObjectWorld case, `True` to use a policy propagation algorithms as in Algorithm 3 of https://arxiv.org/abs/1507.04888. `False` to use an infinite horizon propagation as usually done in infinite horizon IRL.
* `-h`: Help message

Run for example:
* The simulation for GridWorld type 1 with noiseL 0.0 reported in the paper can be reproduced on a cluster with the command
`python run_experiment.py --lr 0.5 --mode 1 --dim 10 --seed 0 --alphaE 1.0 --alphaL 1.0 0.95 0.9 0.85 0.8 --noiseE 0.0 0.05 0.1 0.15 0.2 --noiseL 0.0 --gw True --tdw False --linear True --reg_opp False --fix_start False --fix_horizon False`

followed by:

`python CompareAlphas.py --dim 10 --mode 1 --noiseL 0.0`

change mode and noiseL accordingly to reproduce the other plots. 

* The simulation for TwoDangerWorld with noiseL 0.0 reported in the paper can be reproduced on a cluster with the command
`python run_experiment.py --lr 0.5 --dim 10 --seed 0 --alphaE 1.0 --alphaL 1.0 0.95 0.9 0.85 0.8 --noiseE 0.0 0.05 0.1 0.15 0.2 --noiseL 0.0 --gw True --tdw True --linear True --reg_opp False --fix_start False --fix_horizon False`

followed by:

`python CompareAlphas_tdw.py --dim 10 --mode 1 --noiseL 0.0`

The other plots are obtained simply by changing the value for noiseL.

* The simulation for ObjectWorld with noiseL 0.0 reported in the paper can be reproduced on a cluster with the command
`python run_experiment.py --lr 1e-3 --dim 10 --seed 10 --alphaE 1.0 --alphaL 1.0 0.95 0.9 0.85 0.8 --noiseE 0.0 0.05 0.1 0.15 0.2 --noiseL 0.0 --gw False --tdw False --linear False --reg_opp False --fix_start False --fix_horizon False`

followed by:

`python CompareAlphas_ow.py --dim 10 --env_seed 10 --noiseL 0.0`

The other plots are obtained simply by changing the value for noiseL. Note that changing the arguments --fix_horizon to `True` could run simulations for finite horizion IRL.

### Saving system

We saved the recover reward, the recovered soft optimal policy with respect to the recovered reward, the opponent optimal policy with respect to the recovered reward, the linear combination of the 2 previous policies as determined by the parameter alpha and finally the state occupancy mismatch during training as pickle files.

In **bold** text we indicate the parsed quantity from the `run_experiment.py` interface.

The path is determined by the arguments as follows:


* For GridWorld: /models_gw/fix_start**fix_start**/env_type**mode**/dim**dim**/Linear**linear**/RegOpp**reg_opp**/alphaE_**alphaE**_alphaL_**alphaL**/beta_**beta**_beta_op_**beta_op**/sofQ_lr_**softQ_lr**/noiseE_**noiseE**_noiseL_**noiseL**/seed_**seed**/run_index/

run_index is incremented if the saving path already exists in order not to overwrite previous results.

* For TwoDangerWorld: /models_tdw/fix_start**fix_start**/env_type**mode**/dim**dim**/Linear**linear**/RegOpp**reg_opp**/alphaE_**alphaE**_alphaL_**alphaL**/beta_**beta**_beta_op_**beta_op**/sofQ_lr_**softQ_lr**/noiseE_**noiseE**_noiseL_**noiseL**/seed_**seed**/run_index/

* Finally, for ObjectWorld: /models_ow/fix_horizon**fix_horizon**/fix_start**fix_start**/dim**dim**/Linear**linear**/RegOpp**reg_opp**/alphaE_**alphaE**_alphaL_**alphaL**/beta_**beta**_beta_op_**beta_op**/sofQ_lr_**softQ_lr**/noiseE_**noiseE**_noiseL_**noiseL**/seed_**seed**/run_index/


### Generate the Comparison between different value of alpha

The functions `CompareAlphas.py`,`CompareAlphas_tdw.py` and `CompareAlphas_ow.py` evaluate the accumulated future reward of each policy output by a IRL routine. The prompt interface is as follows for `CompareAlphas.py`,`CompareAlphas_tdw.py`:

* `--mode`: it allows to select one of the different configurations of the environment. For GridWorld the available options are `0, 1, 2, 9, 10`. 
* `--dim`: side dimension of the considered Gridworld
* `--noiseL`: Environment Noises for the learner MDP

While for `CompareAlphas_ow.py`, `env_seed` replaces `mode`. Consequently the arguments are:

* `--env_seed`: Seed for the random placement of objects in the Gridworld. Results are provided for the `env_seed = 10`. 
* `--dim`: side dimension of the considered Gridworld
* `--noiseL`: Environment Noises for the learner MDP

### Provided CompareAlphas results and results visualization

Since the evaluation over a statistically significant number of trajectories can be quite long, we already provide the results in the folder `results/`.

In addition we semplify the access to them through the Jupyter Notebooks `GridWorld plots.ipynb`, `Two Danger World Plots.ipynb`, `Object World Finite Horizon Plots.ipynb`, `Object World Infinite Horizon Plots.ipynb`.




