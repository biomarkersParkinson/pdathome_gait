#!/bin/bash
#Set job requirements
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -p genoa

# Get the current date and time for the output file
timestamp=$(date +%Y%m%d_%H%M%S)

# Define the output file with the timestamp
output_file="../../slurm/output_${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${timestamp}.out"

# Run the sbatch command and specify the output file
#SBATCH --output=$output_file 

# Load modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Activate Poetry environment
source $(poetry env info --path)/bin/activate

# Run the program
nproc=8
steps=5
input_ids=$(cat ../../id_files/all.txt)

python -u run_pipeline.py $nproc $steps $input_ids