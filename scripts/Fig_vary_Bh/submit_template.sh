#!/bin/bash
#SBATCH --job-name=Bh_##IH
#SBATCH --output=Bh_##IH.out
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

time micromamba run -n rls python run.py --ih=##IH
