#! /usr/bin/env bash
set -e

# set environment variables from environment file
source ../.env 2> /dev/null || source .env

# first remove if there's already a venv
echo "CREATION OF NEW CLEAN VIRTUAL ENVIRONMENT"
if [[ $(poetry env info -p) ]]; then
  echo "removing stale $(poetry env info -p)"
  rm -rf "$(poetry env info -p)"
else
  echo 'no existing virtual environment to delete'
fi

# install dependencies
echo "UPDATING PIP TO LATEST VERSION"
poetry run pip install --upgrade pip
echo "INSTALLING DEPENDENCIES"
poetry update
echo "INSTALLING LOCAL PACKAGE"
poetry install
echo "INSTALLING CUDA DEPENDENCIES"
poetry run pip install -r requirements/requirements_torch110_cuda113.txt