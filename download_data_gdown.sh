#!/usr/bin/env bash

set -e

VERSION=${1:-"multi-agent-aerial-dense"}
DATAFOLDER=${2:-"/data/test-iccps/data"}
MAXFILES=${3:-10}

DATAFOLDER=${DATAFOLDER%/}  # remove trailing slash
# DATAFOLDER="${DATAFOLDER}"

DOWNLOAD="https://g-b0ef78.1d0d8d.03c0.data.globus.org/datasets/carla"

if [ "$VERSION" = "multi-agent-aerial-dense" ]; then
    echo "Preparing to download multi-agent-aerial-dense dataset..."
    SAVESUB="multi-agent-aerial-dense"
    SUBDIR="multi-agent-aerial-dense"
    files=(
        "run-2024-11-12_20:41:57,1vHw2Bm3K6ClBoor0XdEPLXxCun2t2ze0"
        "run-2024-11-12_20:46:41,17RIwCSdYX2onhurMo0lRlCzuICb1kxsi"
        "run-2024-11-12_20:51:25,1vH9VDHIGToA_qcYxksN2J0Q_v_fnOg3q"
    )
else
    echo "Cannot understand input version ${VERSION}! Currently can only use 'multi-agent-v1'"
fi

SAVEFULL="${DATAFOLDER}/${SAVESUB}"
mkdir -p $SAVEFULL

echo "Downloading up to $MAXFILES files"
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
    F_REP="${NAME//-/:}"
    evidence="${SAVEFULL}/${F_REP}/.full_download"

    # -- check for evidence of full download
    if [ -f "$evidence" ]; then
        echo -e "$shortname exists and already unzipped\n"
    # -- check for existing tar.gz file
    elif [ -f "$DOWNLOAD/$SAVESUB/$shortname" ]; then
        echo -e "$shortname exists...unzipping\n"
        tar -xvf "$fullname" -C "$SAVEFULL" --force-local
        mv "$DATAFOLDER/$SAVESUB/$SUBDIR/$FILE" "$DATAFOLDER/$SAVESUB/$FILE"  # this is a result of a saving error previously
        rm -r "$DATAFOLDER/$SAVESUB/$SUBDIR"
        rm "$DATAFOLDER/$SAVESUB/${FILE}.tar.gz"
    # -- otherwise, download again
    else
        echo "Downloading ${shortname}"
        # wget -P "$SAVEFULL" "$DOWNLOAD/$SAVESUB/$shortname"
        gdown --id $FID -O $SAVEFULL
        tar -xvf "$fullname" -C "$SAVEFULL" --force-local
        mv "$DATAFOLDER/$SAVESUB/$SUBDIR/$F_REP" "$DATAFOLDER/$SAVESUB/$F_REP"  # this is a result of a saving error previously
        rm -r "$DATAFOLDER/$SAVESUB/$SUBDIR"
        rm "$DATAFOLDER/$SAVESUB/${FILE}.tar.gz"
    fi
    
    # -- add evidence of successful download
    touch "$evidence"

    # -- check downloads
    COUNT=$((COUNT+1))
    echo "Downloaded $COUNT / $MAXFILES files!"
    if [[ $COUNT -ge $MAXFILES ]]; then
            echo "Finished downloading $COUNT files"
            break
    fi
done
