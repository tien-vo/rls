#!/bin/bash


for w in 0.05 0.1 0.15;
do
    for bw in 0.01;
    do
        for bh in 0.3 0.5 0.8;
        do
            cp submit_template.sh submit.sh
            sed -i "s/##WWCE/$w/g" submit.sh
            sed -i "s/##BWB0/$bw/g" submit.sh
            sed -i "s/##BHB0/$bh/g" submit.sh
            sbatch submit.sh
            rm submit.sh
        done
    done
done
