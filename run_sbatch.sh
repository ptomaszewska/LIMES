#!/bin/bash

path_to_files=$1
path_results=$2

for METHOD in LIMES incremental random ensemble restart
do
for SUBSET in 0_5 10_15
do
for DATA in tweet location tweet_location
do
for SEED in `seq 0 9`
do

sbatch run_training.sh "${METHOD}" "${SUBSET}" "${DATA}" "${SEED}" "${path_to_files}" "${path_results}"
done
done
done
done



