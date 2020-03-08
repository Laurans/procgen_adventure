#!/bin/bash

image_name="alaurans/procgen_adventure"
container_name="workspace"
if [ ! "$(docker ps -q -f name=$container_name)" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=$container_name)" ]; then
        if [ "$1" == "rm" ]; then
            docker rm $container_name
        else
            docker start -i $container_name
        fi

    else
        docker run --gpus all -it -e DISPLAY=$DISPLAY --mount="src=/tmp/.X11-unix,dst=/tmp/.X11-unix,type=bind" --mount="src=${PWD},dst=/home/developer/workdir,type=bind" --name $container_name $image_name
    fi
else
    docker exec -it $container_name bash
fi
