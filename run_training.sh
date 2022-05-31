#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 4G
#SBATCH --time 18:00:00

conda activate LIMES_env


METHOD=$1
SUBSET=$2 
DATA=$3
SEED=$4
path_to_files=$5
path_results=$6


FILENAME=hourly-${METHOD}-${DATA}-${SUBSET}-seed${SEED}
python -u training.py -d ${SUBSET} -S ${DATA} -m ${METHOD} -s ${SEED} -p ${path_to_files}> ${path_results}/${FILENAME}.res 2>${path_results}/${FILENAME}.stderr


