#!/bin/bash
#SBATCH --job-name=humanoid_training_new  # Job name
#SBATCH --output=training_output_new.txt  # Standard output log file
#SBATCH --error=training_error_new.txt    # Standard error log file
#SBATCH --partition=general               # Specify the correct partition
#SBATCH --gres=gpu:a100:2                 # Request two A100 GPUs
#SBATCH --cpus-per-task=48                # Number of CPUs per task
#SBATCH --mem=256GB                       # Increased memory allocation
#SBATCH --time=48:00:00                   # Maximum run time

# Load the necessary modules
module load cuda/11.8.0-gcc-12.1.0
module load nccl-2.11.4-1-gcc-11.2.0

# Run the training script
python popo3_.p
