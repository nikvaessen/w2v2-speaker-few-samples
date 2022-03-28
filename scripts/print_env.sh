#! /usr/bin/env bash
set -e

echo "### System environment variables ###"
printenv

# set environment variables from environment file
source ../.env 2> /dev/null || source .env

echo "### Project environment variables ###"
echo "DATA_FOLDER=$DATA_FOLDER"
echo "LOG_FOLDER=$LOG_FOLDER"
echo "TEMP_FOLDER=$TEMP_FOLDER"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "USE_WANDB=$USE_WANDB"
echo "WANDB_API_KEY=$WANDB_API_KEY"
