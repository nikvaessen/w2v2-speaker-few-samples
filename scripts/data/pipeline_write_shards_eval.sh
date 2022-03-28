#! /usr/bin/env bash
set -e

SCRIPT_DIR=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
cd "$SCRIPT_DIR"

# set environment variables from environment file
source ../../.env 2> /dev/null || source .env

# define root destination directories
if [ $# -eq 4 ]; then
  VOX2_SPLIT=$1
  VOX2_SPLIT_ALL=$2
  VOX2_SPLIT_HARD=$3
  SHARD_FOLDER_EVAL=$4
else
  echo "script requires 4 input paths (VOX2_SPLIT,VOX2_SPLIT_ALL,VOX2_SPLIT_HARD,SHARD_FOLDER_EVAL)"
  exit 1
fi

# write test (original) shards
echo "writing test shards"
poetry run python write_voxceleb_shards.py \
    --root_data_path "$VOX2_SPLIT"/test \
    --output_folder "$SHARD_FOLDER_EVAL"/original \
    --name test \
    --compress "$COMPRESS_SHARDS" \
    --samples_per_shard 5000 \
    --ensure_all_data_in_shards true \
    --discard_partial_shards false 

# write test (extended) shards
echo "writing extended test shards"
poetry run python write_voxceleb_shards.py \
    --root_data_path "$VOX2_SPLIT_ALL"/test \
    --output_folder "$SHARD_FOLDER_EVAL"/extended \
    --name test \
    --compress "$COMPRESS_SHARDS" \
    --samples_per_shard 5000 \
    --ensure_all_data_in_shards true \
    --discard_partial_shards false 

# write test (hard) shards
echo "writing hard test shards"
poetry run python write_voxceleb_shards.py \
    --root_data_path "$VOX2_SPLIT_HARD"/test \
    --output_folder "$SHARD_FOLDER_EVAL"/hard \
    --name test \
    --compress "$COMPRESS_SHARDS" \
    --samples_per_shard 5000 \
    --ensure_all_data_in_shards true \
    --discard_partial_shards false 
