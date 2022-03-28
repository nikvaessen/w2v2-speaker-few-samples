#! /usr/bin/env bash
set -e

SCRIPT_DIR=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
cd "$SCRIPT_DIR"

# set environment variables from environment file
source ../../.env 2> /dev/null || source .env

# download metadata
echo "downloading meta data"
poetry run ../download/download_voxceleb_meta.sh

# extract archives
echo "unzipping archives"
poetry run ./pipeline_unzip.sh "$TEMP_FOLDER"

# create train/val/test split(s) of data
VOX2_SPLIT="$TEMP_FOLDER"/vox2_split
VOX2_SPLIT_ALL="$TEMP_FOLDER"/vox2_split_all
VOX2_SPLIT_HARD="$TEMP_FOLDER"/vox2_split_hard
TINY_MANY_LOW="$TEMP_FOLDER"/tiny_many_low
TINY_MANY_HIGH="$TEMP_FOLDER"/tiny_many_high
TINY_FEW="$TEMP_FOLDER"/tiny_few

# full voxceleb
poetry run ./pipeline_split_full_voxceleb.sh "$VOX2_SPLIT" "$VOX2_SPLIT_ALL" "$VOX2_SPLIT_HARD"

# tiny voxceleb
poetry run ./pipeline_split_tiny_datasets.sh "$VOX2_SPLIT" "$TINY_MANY_LOW" "$TINY_MANY_HIGH" "$TINY_FEW"

# prepare writing shards
SHARD_FOLDER_EVAL=$DATA_FOLDER/shards_eval
SHARD_FOLDER_FULL=$DATA_FOLDER/shards_vox2_full
SHARD_FOLDER_TINY_MANY_LOW=$DATA_FOLDER/shards_tiny_many_low
SHARD_FOLDER_TINY_MANY_HIGH=$DATA_FOLDER/shards_tiny_many_high
SHARD_FOLDER_TINY_FEW=$DATA_FOLDER/shards_tiny_few

mkdir -p "$SHARD_FOLDER_EVAL" "$SHARD_FOLDER_FULL" "$SHARD_FOLDER_TINY_MANY_LOW" \
         "$SHARD_FOLDER_TINY_MANY_HIGH" "$SHARD_FOLDER_TINY_FEW"

# generate val trials
echo "generating val trial list for full voxceleb2"
poetry run python generate_trials.py \
    --data_folder "$VOX2_SPLIT"/val \
    --vox_meta_path "$DATA_FOLDER/meta/vox2_meta.csv" \
    --save_path "$DATA_FOLDER"/meta/val_trials.txt \
    --num_pairs 30000 \
    --ensure_same_sex_trials true \
    --ensure_diff_recording_trials false # only 1 to 2 recording per speaker

echo "generating val trial list for tiny-voxceleb with few speakers"
poetry run python generate_trials.py \
    --data_folder "$TINY_FEW"/val \
    --vox_meta_path "$DATA_FOLDER/meta/vox2_meta.csv" \
    --save_path "$DATA_FOLDER"/meta/val_trials_tiny_few.txt \
    --num_pairs 2000 \
    --ensure_same_sex_trials true \
    --ensure_diff_recording_trials false # only 1 to 2 recording per speaker

# write train/val shards for tiny-voxceleb with many speakers and low session variation
poetry run ./pipeline_write_shards_train_val.sh "$TINY_MANY_LOW" "$SHARD_FOLDER_TINY_MANY_LOW"

# write train/val shards for tiny-voxceleb with many speakers and high session variation
poetry run ./pipeline_write_shards_train_val.sh "$TINY_MANY_HIGH" "$SHARD_FOLDER_TINY_MANY_HIGH"

# write train/val shards for tiny-voxceleb with few speakers
poetry run ./pipeline_write_shards_train_val.sh "$TINY_FEW" "$SHARD_FOLDER_TINY_FEW"

# write train/val shards for full voxceleb2
poetry run ./pipeline_write_shards_train_val.sh "$VOX2_SPLIT" "$SHARD_FOLDER_FULL"

# write eval shards
poetry run ./pipeline_write_shards_eval.sh "$VOX2_SPLIT" "$VOX2_SPLIT_ALL" "$VOX2_SPLIT_HARD" "$SHARD_FOLDER_EVAL"
