#! /usr/bin/env bash
set -e

SCRIPT_DIR=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
cd "$SCRIPT_DIR"

# set environment variables from environment file
source ../../.env 2> /dev/null || source .env

# define root destination directories
if [ $# -eq 2 ]; then
  SPLIT=$1
  SHARD_FOLDER=$2
else
  echo "script requires 2 input paths (SPLIT,SHARD_FOLDER)"
  exit 1
fi

# collect all file IDs in this file
UTTERANCE_LIST_FILE=$SHARD_FOLDER/utterance_list.csv
touch "$UTTERANCE_LIST_FILE"
# write csv header
printf "utterance_key;shard_dix;set_name\n" > "$UTTERANCE_LIST_FILE"

# write train shards
echo "writing train shards"
poetry run python write_voxceleb_shards.py \
    --root_data_path "$SPLIT"/train \
    --output_folder "$SHARD_FOLDER"/train \
    --name train \
    --compress "$COMPRESS_SHARDS" \
    --samples_per_shard 5000 \
    --ensure_all_data_in_shards true \
    --discard_partial_shards false \
    --log_csv_file "$UTTERANCE_LIST_FILE"

# write val shards
echo "writing val shards"
poetry run python write_voxceleb_shards.py \
    --root_data_path "$SPLIT"/val \
    --output_folder "$SHARD_FOLDER"/val \
    --name val \
    --compress "$COMPRESS_SHARDS" \
    --samples_per_shard 5000 \
    --ensure_all_data_in_shards true \
    --discard_partial_shards false \
    --log_csv_file "$UTTERANCE_LIST_FILE"

