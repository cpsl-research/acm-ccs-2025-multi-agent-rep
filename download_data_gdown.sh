#!/usr/bin/env bash

set -e

VERSION=${1:-"multi-agent-intersection"}
DATAFOLDER=${2:-"/data/test-ccs/data"}
MAXFILES=${3:-10}

DATAFOLDER=${DATAFOLDER%/}  # remove trailing slash
# DATAFOLDER="${DATAFOLDER}"

DOWNLOAD="https://g-b0ef78.1d0d8d.03c0.data.globus.org/datasets/carla"

if [ "$VERSION" = "multi-agent-intersection" ]; then
    echo "Preparing to download multi-agent-intersection dataset..."
    SAVESUB="multi-agent-intersection"
    SUBDIR="multi-agent-intersection"
    files=(
        "run-2024-04-23_22:32:13-intersection,1e6xaX-5yQHwQ3JCxDrHZ-m4d3dT9fiKm"
        "run-2024-04-23_23:04:15-intersection,1ANXPb-n-0Mxuybqvc5HQTyjIyYiJdZ0D"
        "run-2024-04-23_23:06:00-intersection,1rfqNAJf0_mDkjZd90AFOzaDSZcdoMptO"
    )
else
    echo "Cannot understand input version ${VERSION}! Currently can only use 'multi-agent-intersection'"
fi

SAVEFULL="${DATAFOLDER}/${SAVESUB}"
mkdir -p $SAVEFULL

echo "Downloading files"
COUNT=0

for FILE in ${files[@]}; do
    # split the tuple
    OLDIFS=$IFS;
    IFS=',';
    set -- $FILE  # Convert the "tuple" into the param args $1 $2...
    NAME=$1
    FID=$2
    IFS=$OLDIFS
    
    # get file/folder names
    shortname="${NAME}.tar.gz"
    fullname="${SAVEFULL}/${shortname}"
    # F_REP="${NAME//-/:}"
    evidence="${SAVEFULL}/${NAME}/.full_download"

    # -- check for evidence of full download
    if [ -f "$evidence" ]; then
        echo -e "$shortname exists and already unzipped\n"
    # -- check for existing tar.gz file
    elif [ -f "$DOWNLOAD/$SAVESUB/$shortname" ]; then
        echo -e "$shortname exists...unzipping\n"
        tar -xvf "$fullname" -C "$SAVEFULL" --force-local
        mv "$DATAFOLDER/$SAVESUB/$SUBDIR/$NAME" "$DATAFOLDER/$SAVESUB/$NAME"  # this is a result of a saving error previously
        rm -r "$DATAFOLDER/$SAVESUB/$SUBDIR"
        rm "$DATAFOLDER/$SAVESUB/${NAME}.tar.gz"
    # -- otherwise, download again
    else
        echo "Downloading ${shortname} to output at ${fullname}"
        # wget -P "$SAVEFULL" "$DOWNLOAD/$SAVESUB/$shortname"
        # wget -P "$SAVEFULL" "https://drive.usercontent.google.com/download?export=download&confirm=t&id=${FID}"
        gdown --id $FID -O $fullname
        tar -xvf "$fullname" -C "$SAVEFULL" --force-local
        mv "$DATAFOLDER/$SAVESUB/$SUBDIR/$NAME" "$DATAFOLDER/$SAVESUB/$NAME"  # this is a result of a saving error previously
        rm -r "$DATAFOLDER/$SAVESUB/$SUBDIR"
        rm "$DATAFOLDER/$SAVESUB/${NAME}.tar.gz"
    fi
    
    # -- add evidence of successful download
    touch "$evidence"

    # -- check downloads
    COUNT=$((COUNT+1))
    echo "Downloaded $COUNT files!"
    if [[ $COUNT -ge $MAXFILES ]]; then
            echo "Finished downloading $COUNT files"
            break
    fi
done
