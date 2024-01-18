#!/bin/bash
#SBATCH --job-name=Bw_##IB
#SBATCH --output=Bw_##IB.out
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

time micromamba run -n rls python run.py --ib=##IB
