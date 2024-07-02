#!/bin/bash
#SBATCH --job-name=w_##WWCE_Bw_##BWB0_Bh_##BHB0
#SBATCH --output=w_##WWCE_Bw_##BWB0_Bh_##BHB0.out
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8gb

time micromamba run -n rls python run.py \
    --Bw-B0 ##BWB0 \
    --Bh-B0 ##BHB0 \
    --w-wce ##WWCE
