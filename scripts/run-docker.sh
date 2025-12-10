#!/bin/bash

WORKSPACE="$(dirname ${PWD})"


# docker run --shm-size=16G --gpus all --rm -it -p 8889:8888 --mount type=bind,src=${WORKSPACE},dst=/workspace change-detection /bin/bash
docker run --shm-size=16G --gpus '"device=1"' --rm -it -p 8889:8888 --mount type=bind,src=${WORKSPACE},dst=/workspace change-detection /bin/bash
