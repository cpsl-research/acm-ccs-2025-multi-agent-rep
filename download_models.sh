#!/usr/bin/env bash

set -e

SAVEFOLDER=${1:-"/data/test-ccs/models"}
SAVEFOLDER=${SAVEFOLDER%/}  # remove trailing slash
SAVEFOLDER="$SAVEFOLDER/mmdet"

THISDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Downloading models and saving to $SAVEFOLDER"


MMDET_CKPT="${SAVEFOLDER}/checkpoints"
MMDET_WKDIR="${SAVEFOLDER}/work_dirs"
mkdir -p "$MMDET_CKPT"
mkdir -p "$MMDET_WKDIR"

download_models () {
    MODEL_PATH="https://download.openmmlab.com/mmdetection"
    SUBFOLDER=$1  # Input 1: subfolder
    VERSION=$2    # Input 2: mmdet version
    MODEL_TYPE=$3 # Input 3: model type
    MODEL=$4      # Input 4: model name
    MNAME="${MODEL##*/}"
    if [ -f "${MMDET_CKPT}/${SUBFOLDER}/${MNAME}" ]; then
        echo -e "$MODEL exists.\n"
    else 
        echo "Downloading models"
        wget -P "${MMDET_CKPT}/${SUBFOLDER}" "${MODEL_PATH}/${VERSION}/${MODEL_TYPE}/${MODEL}"

    fi
}

download_custom_models () {
    CUSTOM_MODEL_PATH="https://g-b0ef78.1d0d8d.03c0.data.globus.org/models/mmdet"

    SUBFOLDER=$1  # Input 1: subfolder (e.g., "carla")
    MODEL_TYPE=$2 # Input 2: model type
    MODEL=$3      # Input 3: model name
    MNAME="${MODEL##*/}"
    if [ -f "${MMDET_WKDIR}/${SUBFOLDER}/${MNAME}.pth" ]; then
        echo -e "$MODEL exists.\n"
    else 
        echo "Downloading model and configuration for $MODEL"
        MODPATH="$CUSTOM_MODEL_PATH"
        wget -P "${MMDET_WKDIR}/${SUBFOLDER}" "${MODPATH}/work_dirs/${MODEL_TYPE}/${MODEL}.pth"
        wget -P "${MMDET_WKDIR}/${SUBFOLDER}" "${MODPATH}/work_dirs/${MODEL_TYPE}/${MODEL}.py"
    fi
}


# lidar point cloud models
CARLAJOINT_PILLARS=""



# faster rcnn models
CARLAJOINT_FRCNN="faster_rcnn_r50_fpn_1x_carla_joint"
download_custom_models "faster_rcnn_r50_fpn_1x_carla_joint" "carla" "$CARLAJOINT_FRCNN"

# add last_checkpoint file
echo "Add last checkpoint file"
CKPT_PATH="$MMDET_WKDIR/faster_rcnn_r50_fpn_1x_carla_joint"
echo "$CKPT_PATH/faster_rcnn_r50_fpn_1x_carla_joint.pth" > "$CKPT_PATH/last_checkpoint"

# make symbolic links
echo "Adding symbolic link to mmdet directory"
ln -sfnT $(realpath "$MMDET_CKPT") "$THISDIR/submodules/avstack-core/third_party/mmdetection/checkpoints"
ln -sfnT $(realpath "$MMDET_WKDIR") "$THISDIR/submodules/avstack-core/third_party/mmdetection/work_dirs"
