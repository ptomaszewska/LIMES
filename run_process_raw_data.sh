#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 4G
#SBATCH --time 18:00:00
#SBATCH -o ./outputs/process_raw_%j.out
#SBATCH -e ./errors/process_raw_%j.err


conda activate LIMES


path_to_files = $1
save_path = $2


for month in $(seq -f "%02g" 3 9)
do
	for day in $(seq -f "%02g" 10 15)
	do 
		for hour in $(seq -f "%02g" 0 23)
		do
       			python process_raw_data.py "${path_to_files}_geo-2020_${month}_${day}_${hour}.jsonl" ${save_path}
		done
	done
done


