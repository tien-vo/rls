#!/bin/bash

# for ih in $(seq 0 1 49);
for ih in 0 1 2 3 37 38 39 40 41;
do
    submit_file=submit_Bh_"$ib".sh;
    cp submit_template.sh $submit_file;
    sed -i "s/##IH/$ih/g" $submit_file;
    sbatch $submit_file;
    rm $submit_file;
done
