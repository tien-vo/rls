#!/bin/bash
#SBATCH --job-name=plot
#SBATCH --output=plot.out
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8gb

micromamba run -n rls python process.py
micromamba run -n rls python plot.py
