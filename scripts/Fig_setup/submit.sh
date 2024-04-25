#!/bin/bash
#SBATCH --job-name=Fig_setup
#SBATCH --output=Fig_setup.out
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

time micromamba run -n rls python run_array_IC.py
time micromamba run -n rls python run_random_IC.py
