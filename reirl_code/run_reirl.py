import argparse
import os
import numpy as np
import ast

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default="GaussianGridworld-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--expert-traj-path', metavar='G',
                    help='path of the expert trajectories')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='gae (default: 3e-4)')
parser.add_argument("--alpha", nargs='+', default=["0"])
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 1)')
parser.add_argument("--seed", nargs='+', default=["0"])
parser.add_argument('--max-iter-num', type=int, default=100, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument("--noiseL", nargs='+', default=["0"])
parser.add_argument("--noiseE", type=float, default = 0.0)
parser.add_argument('--grid-type', type=int, default=1, metavar='N')
parser.add_argument('--scheduler-lr', action='store_true', default=False,
                    help='Use discriminator lr scheduler')
parser.add_argument("--script_name", default='submit.sh')
parser.add_argument("--logs_folder", default='./logs')
parser.add_argument("--job_name", default='')
args = parser.parse_args()

# If submit script does not exist, create i
if not os.path.isfile(args.script_name):
    with open(args.script_name, 'w') as file:
        file.write(f'''#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=1000
./staskfarm ${{1}}\n''')
if args.env_name == "GaussianGridworld-v0":
    for seed in args.seed:
        for alpha in args.alpha:
            for noiseL in args.noiseL:
                folder = f'{args.logs_folder}/gw'

                path = f'env_name_{args.env_name}/' \
                       f'grid_type_{args.grid_type}/alpha_{alpha}/' \
                       f'lr_{args.learning_rate}/' \
                       f'noiseE_{args.noiseE}/' \
                       f'noiseL_{noiseL}/' \
                       f'seed_{seed}'


                if not os.path.isdir(f'{folder}/{path}'):
                    os.makedirs(f'{folder}/{path}')

                file = f'gail/reirl_gym.py'
                command = f'python {file} --env-name {args.env_name} ' \
                          f'--grid-type {args.grid_type} ' \
                          f'--expert-traj-path {args.expert_traj_path} ' \
                          f'--num-threads {args.num_threads} ' \
                          f'--log-interval {args.log_interval} ' \
                          f'--save-model-interval {args.save_model_interval} ' \
                          f'--max-iter-num {args.max_iter_num} ' \
                          f'--learning-rate {args.learning_rate} --alpha {alpha} ' \
                          f'--seed {seed} ' \
                          f'--noiseE {args.noiseE} ' \
                          f'--noiseL {noiseL}'

                experiment_path = f'{folder}/{path}/command.txt'

                with open(experiment_path, 'w') as file:
                    file.write(f'{command}\n')

                print(command)

                if not args.job_name:
                    job_name = path
                else:
                    job_name = args.job_name

                os.system(f'sbatch --job-name={job_name} {args.script_name} {experiment_path}')
