#!/bin/bash
#Set job requirements
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -p genoa
#SBATCH -o ./slurm/output.%j.out # STDOUT

#Loading modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Activate Poetry environment
source $(poetry env info --path)/bin/activate

#Run same program over many different inputs
nproc=3
steps=2345
input_ids=$(cat ../../id_files/all.txt)

python -u run_pipeline.py $nproc $steps $input_ids