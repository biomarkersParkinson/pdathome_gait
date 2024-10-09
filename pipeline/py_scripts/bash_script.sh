#!/bin/bash
#Set job requirements
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -p genoa

# Set the output directory and filename directly in the SBATCH directive
#SBATCH --output=../../slurm/output_%x_%j.out

# Load modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Activate Poetry environment
source $(poetry env info --path)/bin/activate

# Run the program
nproc=8
steps=67
input_ids=$(cat ../../id_files/all.txt)

python -u run_pipeline.py $nproc $steps $input_ids