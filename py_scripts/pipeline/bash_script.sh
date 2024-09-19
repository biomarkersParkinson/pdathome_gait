#!/bin/bash
#Set job requirements
#SBATCH -N 1
#SBATCH -t 00:15:00
#SBATCH -p genoa
#SBATCH -o ./slurm/output.%j.out # STDOUT

#Loading modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

#Run same program over many different inputs
nproc=3
input_ids=$(cat ../id_files/all.txt)

python -u 0.preparing_data.py $nproc $input_ids