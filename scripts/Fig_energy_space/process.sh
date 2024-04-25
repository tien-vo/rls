#!/bin/bash


for w in 0.05 0.1 0.15;
do
    for bw in 0.01;
    do
        for bh in 0.3 0.5 0.8;
        do
            micromamba run -n rls python process.py \
                --w-wce $w \
                --Bw-B0 $bw \
                --Bh-B0 $bh
        done
    done
done
