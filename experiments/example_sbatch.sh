#!/bin/bash

# Parameters
#SBATCH --account=das
#SBATCH --partition=das
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=1
#SBATCH --job-name=w2v2_tiny_many
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --error=/home/nvaessen/slurm/%J.err
#SBATCH --output=/home/nvaessen/slurm/%J.out

## Bail out on errors
set -e
echo "Start of my script"
echo "The time is $(date)"
echo "I am running on machine $(hostname)"
echo "I am running this from the folder $(pwd)"
echo "I know the following environment variables:"
printenv
echo "This is the ouput of nvidia-smi:"
nvidia-smi
echo "Pretending to be busy for a while"
sleep 10
echo "This is enough, the time is now $(date)"
exit 0

