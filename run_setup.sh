#!/usr/bin/env bash

set -e 

DATAFOLDER=${1:-/data/$(whoami)/}
MODELFOLDER=${2:-/data/$(whoami)/models}

DATAFOLDER=${DATAFOLDER%/}  # remove trailing slash
MODELFOLDER=${MODELFOLDER%/}  # remove trailing slash

# (optional) Add symbolic links to data in api folder
# ./submodules/lib-avstack-api/data/add_custom_symlinks.sh $DATAFOLDER

# Add symbolic links to perception models
ln -sfnT "${DATAFOLDER}/multi-agent-aerial-dense" "./data/multi-agent-aerial-dense"

# download camera-based perception models
bash download_models.sh $MODELFOLDER