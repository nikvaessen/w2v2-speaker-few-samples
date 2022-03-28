#! /usr/bin/env bash
set -e

# set environment variables from environment file
source ../../.env 2> /dev/null || source .env

# define root destination directories
if [ $# -eq 4 ]; then
  VOX2_SPLIT=$1
  TINY_MANY_LOW=$2
  TINY_MANY_HIGH=$3
  TINY_FEW=$4
else
  echo "script requires 4 input paths (VOX2_SPLIT, TINY_MANY_LOW,TINY_MANY_HIGH.TINY_FEW)"
  exit 1
fi

# delete splits if they exist
if [ -d "$TINY_MANY_LOW" ]; then
  echo "removing $TINY_MANY_LOW"
  rm -r "$TINY_MANY_LOW"
fi
if [ -d "$TINY_MANY_HIGH" ]; then
  echo "removing $TINY_MANY_HIGH"
  rm -r "$TINY_MANY_HIGH"
fi
if [ -d "$TINY_FEW" ]; then
  echo "removing $TINY_FEW"
  rm -r "$TINY_FEW"
fi

# tiny voxceleb with many speakers
echo "creating tiny dataset with many speakers and few files with low session variation"
poetry run python tinyfy_voxceleb.py \
  --train_folder "$VOX2_SPLIT"/train/ \
  --val_folder "$VOX2_SPLIT"/val/ \
  --output_folder "$TINY_MANY_LOW" \
  --mode many \
  --num_files_per_speaker 8 \
  --num_files_per_recording 8

# tiny voxceleb with many speakers
echo "creating tiny dataset with many speakers and few files with high session variation"
poetry run python tinyfy_voxceleb.py \
  --train_folder "$VOX2_SPLIT"/train/ \
  --val_folder "$VOX2_SPLIT"/val/ \
  --output_folder "$TINY_MANY_HIGH" \
  --mode many \
  --num_files_per_speaker 8 \
  --num_files_per_recording 1

# tiny voxceleb with few speakers
echo "creating tiny dataset with few speakers and many files"
poetry run python tinyfy_voxceleb.py \
  --train_folder "$VOX2_SPLIT"/train/ \
  --val_folder "$VOX2_SPLIT"/val/ \
  --output_folder "$TINY_FEW" \
  --mode few \
  --meta_file "$DATA_FOLDER"/meta/vox2_meta.csv \
  --num_speakers_per_gender 50
