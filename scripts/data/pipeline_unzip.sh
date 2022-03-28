#! /usr/bin/env bash
set -e

# set environment variables from environment file
source ../../.env 2> /dev/null || source .env

# set and create root destination dir
if [ $# -eq 1 ]; then
  echo "using given directory $1 as root unzip directory"
  UNZIP_DIR=$1
else
  echo "using default directory TEMP_FOLDER=$TEMP_FOLDER as root unzip directory"
  UNZIP_DIR="$TEMP_FOLDER"
fi

mkdir -p "$UNZIP_DIR"

# general function for unzipping
unzip_voxceleb () {
  SRC=$1
  DST=$2

  if [ -d "$DST" ]; then
    echo "$SRC is already extracted to $DST"
  elif [ -f "$SRC" ]; then
    echo "unzipping $SRC to $DST"
    NUM_FILES=$(zipinfo -h "$SRC" | grep -oiP '(?<=entries: )[[:digit:]]+')
    mkdir -p "$DST"
    unzip -n "$SRC" -d "$DST" | tqdm --total "$NUM_FILES" >> /dev/null
  else
     echo "file $SRC does not exist."
  fi
}

# voxceleb1 train
VOX1_TRAIN="$DATA_FOLDER"/archives/vox1_dev_wav.zip
VOX1_TRAIN_DEST="$UNZIP_DIR"/voxceleb1/train
unzip_voxceleb "$VOX1_TRAIN" "$VOX1_TRAIN_DEST"

# voxceleb1 test
VOX1_TEST="$DATA_FOLDER"/archives/vox1_test_wav.zip
VOX1_TEST_DEST="$UNZIP_DIR"/voxceleb1/test
unzip_voxceleb "$VOX1_TEST" "$VOX1_TEST_DEST"

# voxceleb2 train
VOX2_TRAIN="$DATA_FOLDER"/archives/vox2_dev_wav.zip
VOX2_TRAIN_DEST="$UNZIP_DIR"/voxceleb2/train
unzip_voxceleb "$VOX2_TRAIN" "$VOX2_TRAIN_DEST"

# voxceleb2 test
VOX2_TEST="$DATA_FOLDER"/archives/vox2_test_wav.zip
VOX2_TEST_DEST="$UNZIP_DIR"/voxceleb2/test
unzip_voxceleb "$VOX2_TEST" "$VOX2_TEST_DEST"

