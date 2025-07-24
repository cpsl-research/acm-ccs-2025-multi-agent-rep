#!/usr/bin/env bash

set -e 

DATAFOLDER=${1:-/data/$(whoami)/}
MODELFOLDER=${2:-/data/$(whoami)/models}

DATAFOLDER=${DATAFOLDER%/}  # remove trailing slash
MODELFOLDER=${MODELFOLDER%/}  # remove trailing slash

# (optional) Add symbolic links to data in api folder
# ./submodules/avstack-api/data/add_custom_symlinks.sh $DATAFOLDER

# Add symbolic links to perception models
mkdir -p ./data
ln -sfnT "${DATAFOLDER}/multi-agent-intersection" "./data/multi-agent-intersection"

# download camera-based and lidar-based perception models
bash download_models.sh $MODELFOLDER