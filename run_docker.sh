#!/usr/bin/env bash

set -e 

DATAFOLDER=${1:-"/data/test-iccps/data"}
MODELFOLDER=${2:-"/data/test-iccps/models"}

DATAFOLDER=${DATAFOLDER%/}  # remove trailing slash
MODELFOLDER=${MODELFOLDER%/}  # remove trailing slash

xhost local:root

# this for remote docker image
# DOCKERNAME="roshambo919/iccps25:aerial-trust"
# this for local docker image
DOCKERNAME="aerial-trust"

# pull docker image (if using local image, comment out)
# docker pull $DOCKERNAME

# check if the data folders are populated before continuing
if [ -z "$(ls -A $DATAFOLDER)" ]; then
   echo "\nERROR: nothing found in the data directory at $DATAFOLDER. Ensure you have downloaded the data before starting container"
   exit
else
   echo "Data directory not empty...assuming it is populated with data!"
fi
if [ -z "$(ls -A $MODELFOLDER)" ]; then
   echo "\nERROR: nothing found in the model directory at $MODELFOLDER. Ensure you have downloaded models before starting container"
   exit
else
   echo "Model directory not empty...assuming it is populated with models!"
fi


# function to start docker
start_docker () {
    DF=$1
    MF=$2sw
    echo "Starting fresh docker container"
    docker run \
      --name aerial-trust \
      --privileged \
      --runtime=nvidia \
      --gpus 'all,"capabilities=graphics,utility,display,video,compute"' \
      --mount type=bind,src="$DATAFOLDER",target=/data \
      --mount type=bind,src="$MODELFOLDER",target=/models \
      -p 8888:8888 \
      $DOCKERNAME \
       /bin/bash -c "bash run_setup.sh /data /models && poetry run jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"
}


# Remove if there is existing container
CONT_ID=$(docker ps -aqf "name=^aerial-trust")
if [ "$CONT_ID" == "" ];
then
	:
else
	echo "Stopping and removing existing docker container"
	docker stop $CONT_ID
	docker rm $CONT_ID
fi

# Start up a docker container
echo "Starting up docker container"
start_docker $DATAFOLDER $MODELFOLDER


exit 0