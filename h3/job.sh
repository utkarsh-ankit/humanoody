#!/bin/bash
#SBATCH --job-name=humanoid_training_new
#SBATCH --output=/home/uankit/training_output_new.txt
#SBATCH --error=/home/uankit/training_error_new.txt
#SBATCH --partition=general       # Specify the correct partition
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=1                # Number of tasks (1 task per node)
#SBATCH --cpus-per-task=48        # Number of CPUs per task
#SBATCH --gres=gpu:a100:2         # Number of GPUs per node
#SBATCH --mem=128G                # Memory per node
#SBATCH --time=7-00:00:00         # Maximum run time
#SBATCH --qos=public              # Quality of Service

# Load the necessary modules
module load cuda/11.8.0-gcc-12.1.0
module load nccl-2.11.4-1-gcc-11.2.0

# Activate your Python environment
source activate py310

# Run the training script
python /home/uankit/h2.py
