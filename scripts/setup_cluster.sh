#! /usr/bin/env bash

echo "### INSTALLING DEPENDENCIES ON CN99 ###"
./scripts/setup_dependencies.sh

echo "### INSTALLING DEPENDENCIES ON CN104 ###"
ssh cn104 "
  source .profile
  cd $PWD;
  ./scripts/setup_dependencies.sh
"
echo "### INSTALLING DEPENDENCIES ON CN105 ###"
ssh cn105 "
  source .profile
  cd $PWD;
  ./scripts/setup_dependencies.sh
"