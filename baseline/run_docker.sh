#! /bin/bash

DOCKER_IMAGE="simmc2_baseline"
DATASET_PATH=/data/simmc2/

docker run -it --runtime=nvidia -v ${DATASET_PATH}:/mnt/simmc2 ${DOCKER_IMAGE} bash

