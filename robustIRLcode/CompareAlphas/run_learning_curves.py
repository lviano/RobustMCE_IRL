import argparse
import os
import numpy as np
import ast

parser = argparse.ArgumentParser()
parser.add_argument("--script_name", default='submit.sh')
parser.add_argument("--logs_folder", default='./logs')
parser.add_argument("--job_name", default='')
parser.add_argument("--mode", type=int)
parser.add_argument("--lr", nargs='+', default=["0"])
parser.add_argument("--dim", nargs='+', default=["0"])
parser.add_argument("--seed", nargs='+', default=["0"])
parser.add_argument("--noiseE", nargs='+', default=["0"])
parser.add_argument("--noiseL", nargs='+', default=["0"])
parser.add_argument("--gw", type=ast.literal_eval)
args = parser.parse_args()

# If submit script does not exist, create it
if not os.path.isfile(args.script_name):
    with open(args.script_name, 'w') as file:
        file.write(f'''#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=10000

./staskfarm ${{1}}\n''')

for seed in args.seed:
    for lr in args.lr:
        for dim in args.dim:
            for noiseE in args.noiseE:
                for noiseL in args.noiseL:
                    if (args.gw):
                        folder = f'{args.logs_folder}/learning_curves_gw/'
                    else:
                        folder = f'{args.logs_folder}/learning_curves_ow/'
                    path = f'mode_{args.mode}/dim_{dim}/lr_{lr}/noiseE_{noiseE}_noiseL_{noiseL}/seed_{seed}'

                    if not os.path.isdir(f'{folder}/{path}'):
                        os.makedirs(f'{folder}/{path}')

                    print(path)
                    if (args.gw):
                        file = f'PlotLearningCurves.py'
                        command = f'python {file} --dim {dim} --mode {args.mode} --noiseE {noiseE} --noiseL {noiseL} --alphaL {str(1 - float(noiseL))} --include_iil {True}'

                    else:
                        file = f'PlotLearningCurves_ow.py'
                        command = f'python {file} --dim {dim} --env_seed {args.mode} --noiseE {noiseE} --noiseL {noiseL} --alphaL {str(1 - float(noiseL))} --include_iil {True}'

                    experiment_path = f'{folder}/{path}/command.txt'

                    with open(experiment_path, 'w') as file:
                        file.write(f'{command}\n')

                    print(command)

                    if not args.job_name:
                        job_name = path
                    else:
                        job_name = args.job_name

                    os.system(f'sbatch --job-name={job_name} {args.script_name} {experiment_path}')
