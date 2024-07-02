#!/bin/bash

#for ih in $(seq 0 1 99);
# for ih in $(seq 53 1 74);
for ih in 55 56 58 59 71 72;
do
    submit_file=submit_Bh_"$ih".sh;
    cp submit_template.sh $submit_file;
    sed -i "s/##IH/$ih/g" $submit_file;
    sbatch $submit_file;
    rm $submit_file;
done
