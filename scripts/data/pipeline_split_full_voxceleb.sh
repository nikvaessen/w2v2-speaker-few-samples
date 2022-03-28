#! /usr/bin/env bash
set -e

# set environment variables from environment file
source ../../.env 2> /dev/null || source .env

# define root destination directories
if [ $# -eq 3 ]; then
  VOX2_SPLIT=$1
  VOX2_SPLIT_ALL=$2
  VOX2_SPLIT_HARD=$3
else
  echo "script requires 3 input paths (VOX2_SPLIT,VOX2_SPLIT_ALL,VOX2_SPLIT_HARD)"
  exit 1
fi

# delete splits if they exist
if [ -d "$VOX2_SPLIT" ]; then
  echo "removing $VOX2_SPLIT"
  rm -r "$VOX2_SPLIT"
fi
if [ -d "$VOX2_SPLIT_ALL" ]; then
  echo "removing $VOX2_SPLIT_ALL"
  rm -r "$VOX2_SPLIT_ALL"
fi
if [ -d "$VOX2_SPLIT_HARD" ]; then
  echo "removing $VOX2_SPLIT_HARD"
  rm -r "$VOX2_SPLIT_HARD"
fi

# full voxceleb2 with original test set
echo "creating train/val/test split"
poetry run python split_voxceleb.py \
  --root_folder "$TEMP_FOLDER" \
  --output_folder "$VOX2_SPLIT" \
  --test_trials_path "$DATA_FOLDER"/meta/veri_test2.txt \
  --train_voxceleb1_dev false \
  --train_voxceleb2_dev true \
  --val_split_mode equal \
  --val_ratio 0.01

# only extended test set
echo "creating extended test split"
poetry run python split_voxceleb.py \
  --root_folder "$TEMP_FOLDER" \
  --output_folder "$VOX2_SPLIT_ALL" \
  --test_trials_path "$DATA_FOLDER"/meta/list_test_all2.txt \
  --train_voxceleb1_dev false \
  --train_voxceleb2_dev true \
  --val_split_mode equal \
  --val_ratio 0.01

# only hard test set
echo "creating hard test split"
poetry run python split_voxceleb.py \
  --root_folder "$TEMP_FOLDER" \
  --output_folder "$VOX2_SPLIT_HARD" \
  --test_trials_path "$DATA_FOLDER"/meta/list_test_hard2.txt \
  --train_voxceleb1_dev false \
  --train_voxceleb2_dev true \
  --val_split_mode equal \
  --val_ratio 0.01
