# Robust_REIRL

Implementation of robust IRL. Reproduce the results in three steps:

**Step 1**

Train a PPO agent on the domain GaussianGridworld

```
python run_ppo.py --env-name GaussianGridworld-v0 --max-iter-num 100 --save-model-interval 10 --grid-type 1 --noiseE 0.2
```

**Step 2**

Save trajectories as follows

```
python gail/save_expert_traj.py --env-name GaussianGridworld-v0 --grid-type 2 --model-path assets/envGaussianGridworld-v0type2noiseE0.0/learned_models/GaussianGridworld-v0_ppo_99.p --noiseE 0.0
```


**Step 3**

Run RE IRL

```
python run_reirl.py --env-name GaussianGridworld-v0 --expert-traj-path assets/envGaussianGridworld-v0type2noiseE0.0/expert_traj/GaussianGridworld-v0_expert_traj.p --alpha 1.0 0.99 0.97 0.95 0.9 0.85  --seed 0 1 2 3 4 --max-iter-num 150 --save-model-interval 10 --noiseE 0.0 --noiseL 0.2 --grid-type 2
```


