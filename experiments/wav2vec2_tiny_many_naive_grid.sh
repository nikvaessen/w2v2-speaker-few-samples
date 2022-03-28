#!/bin/bash

# Parameters
#SBATCH --account=das
#SBATCH --partition=das
#SBATCH --array=0-5%6
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=1
#SBATCH --job-name=w2v2_tiny_many
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --error=~/slurm/w2v2_many_%A_%a_log.err
#SBATCH --output=~/slurm/w2v2_many_%A_%a_log.out

LR_ARRAY=(1E-7 1E-6 1E-5 1E-4 1E-3 1E-2)
LR_INDEX=$(( $SLURM_ARRAY_TASK_ID % 6))
LR=${LR_ARRAY["$LR_INDEX"]}

cd /home/nvaessen/repo/w2v2-speaker-few-samples || exit
poetry run python run.py \
+experiment=wav2vec2_tiny_many \
optim.algo.lr="$LR" \
tag=naive_grid