#! /usr/bin/env bash
set -e

### set environment variables
source ../../.env 2> /dev/null || source .env

# default directory to save files in
DIR="$DATA_FOLDER"/meta
mkdir -p "$DIR"

## download files
curl -C - https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt --output "$DIR"/iden_split.txt
curl -C - https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt --output "$DIR"/veri_test.txt
curl -C - https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt --output "$DIR"/veri_test2.txt
curl -C - https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_hard.txt --output "$DIR"/list_test_hard.txt
curl -C - https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_hard2.txt --output "$DIR"/list_test_hard2.txt
curl -C - https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_all.txt --output "$DIR"/list_test_all.txt
curl -C - https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_all2.txt --output "$DIR"/list_test_all2.txt
curl -C - https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox1_meta.csv --output "$DIR"/vox1_meta.csv
curl -C - https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox2_meta.csv --output "$DIR"/vox2_meta.csv

# normalize CSV files to have same header and separator
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
python "$SCRIPT_DIR"/fix_csv_file.py \
  --vox1_meta_path "$DIR"/vox1_meta.csv \
  --vox2_meta_path "$DIR"/vox2_meta.csv
