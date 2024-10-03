#!/bin/bash
#Set job requirements
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -p genoa

# Get the current date and time for the output file
timestamp=$(date +%Y%m%d_%H%M%S)

# Set output file with timestamp
#SBATCH -o ../../slurm/output_%x_%j_${timestamp}.out

#Loading modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Activate Poetry environment
source $(poetry env info --path)/bin/activate

#Run same program over many different inputs
nproc=8
steps=5
input_ids=$(cat ../../id_files/all.txt)

python -u run_pipeline.py $nproc $steps $input_ids