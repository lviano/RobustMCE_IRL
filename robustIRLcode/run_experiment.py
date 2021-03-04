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
parser.add_argument("--softQ_lr", nargs='+', default=["0"])
parser.add_argument("--dim", nargs='+', default=["0"])
parser.add_argument("--seed", nargs='+', default=["0"])
parser.add_argument("--beta", nargs='+', default=["0"])
parser.add_argument("--beta_op", nargs='+', default=["0"])
parser.add_argument("--alphaE", nargs='+', default=["0"])
parser.add_argument("--alphaL", nargs='+', default=["0"])
parser.add_argument("--noiseE", nargs='+', default=["0"])
parser.add_argument("--noiseL", nargs='+', default=["0"])
parser.add_argument("--gw", type=ast.literal_eval)
parser.add_argument("--tdw", type=ast.literal_eval)
parser.add_argument("--linear", type=ast.literal_eval)
parser.add_argument("--reg_opp", type=ast.literal_eval)
parser.add_argument("--fix_start", type=ast.literal_eval)
parser.add_argument("--fix_horizon", type=ast.literal_eval)

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
            for softQ_lr in args.softQ_lr:
                for alphaE in args.alphaE:
                    for alphaL in args.alphaL:
                        for beta in args.beta:
                            for beta_op in args.beta_op:
                                for noiseE in args.noiseE:
                                    for noiseL in args.noiseL:
                                        if(args.gw):
                                            folder = f'{args.logs_folder}/gw'
                                        else:
                                            folder = f'{args.logs_folder}/ow'

                                        path = f'mode_{args.mode}/fix_horizon_{args.fix_horizon}/fix_start_{args.fix_start}/linear_{args.linear}/reg_opp_{args.reg_opp}/dim_{dim}/alphaE_{alphaE}_alphaL_{alphaL}/lr_{lr}/beta_{beta}_betaop_{beta_op}/softQ_lr_{softQ_lr}/noiseE_{noiseE}_noiseL_{noiseL}/seed_{seed}'

                                        if not os.path.isdir(f'{folder}/{path}'):
                                            os.makedirs(f'{folder}/{path}')

                                        print(path)
                                        if (args.gw) and (args.tdw):
                                            file = f'main_tdw.py'
                                        elif (args.gw) and not (args.tdw):
                                            file = f'main_gw.py'
                                        else:
                                            file = f'main_ow.py'

                                        command = f'python {file} --lr {lr} --dim {dim} --mode {args.mode} --seed {seed} --beta {beta} --beta_op {beta_op} --alphaE {alphaE} --alphaL {alphaL} --softQ_lr {softQ_lr} --noiseE {noiseE} --noiseL {noiseL} --linear {args.linear} --reg_opp {args.reg_opp} --fix_start {args.fix_start} --fix_horizon {args.fix_horizon}'

                                        experiment_path = f'{folder}/{path}/command.txt'

                                        with open(experiment_path, 'w') as file:
                                            file.write(f'{command}\n')

                                        print(command)

                                        if not args.job_name:
                                            job_name = path
                                        else:
                                            job_name = args.job_name

                                        os.system(f'sbatch --job-name={job_name} {args.script_name} {experiment_path}')
