#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 4G
#SBATCH --time 18:00:00

METHOD=${1:-limes}

export CUDA_VISIBLE_DEVICES=${2}

for SUBSET in 0_5 10_15
do
for DATA in tweet location tweet_location
do
for SEED in `seq 0 9`
do
FILENAME=hourly-${METHOD}-${DATA}-${SUBSET}-seed${SEED}-bias
python -u training.py -d ${SUBSET} -S ${DATA} -m ${METHOD} -s ${SEED} > ${FILENAME}.res 2>${FILENAME}.stderr

done
done
done
