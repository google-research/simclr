#!/bin/bash

# Get the current dir.
if [ -n "$BASH_VERSION" ]; then
    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
elif [ -n "$ZSH_VERSION" ]; then
    DIR=${0:a:h}  # https://unix.stackexchange.com/a/115431
else
	echo "Error: Unknown shell; cannot determine path to merantix/core local repository"
fi
export SIMCLR_REPO_DIR="$(dirname $DIR)"

# for ipython notebooks pushed to repository, automatically strip output
git --git-dir="SIMCLR_REPO_DIR/.git" config --local core.autocrlf input
git --git-dir="SIMCLR_REPO_DIR/.git" config filter.prepare_notebook_for_repository.clean 'developer_env/prepare_notebook_for_repository.py'

alias cd_simclr="cd $SIMCLR_REPO_DIR"

DOCKER_BASH_HISTORY="$SIMCLR_REPO_DIR/data/docker.bash_history"
touch $DOCKER_BASH_HISTORY

DOCKER_IMAGE="simclr"

# docker aliases
alias simclr_docker_build="docker build -t $DOCKER_IMAGE $SIMCLR_REPO_DIR/developer_env/simclr"

alias simclr_docker_build_gpu="docker build --build-arg base_image=tensorflow/tensorflow:2.6.0-gpu-jupyter -t $DOCKER_IMAGE $SIMCLR_REPO_DIR/developer_env/simclr"

alias simclr_run="docker run -it --rm \
    -v $SIMCLR_REPO_DIR:/simclr \
    -v $HOME/.config/gcloud:/root/.config/gcloud \
    -v $SIMCLR_REPO_DIR/data/docker.bash_history:/root/.bash_history \
    $DOCKER_IMAGE"

alias simclr_jupyter="docker run -it --rm \
    --hostname localhost \
    -v $SIMCLR_REPO_DIR:/simclr \
    -v $HOME/.config/gcloud:/root/.config/gcloud \
    -v $SIMCLR_REPO_DIR/data/docker.bash_history:/root/.bash_history \
    -p 0.0.0.0:8888:8888 \
    $DOCKER_IMAGE \
    jupyter notebook \
        --port=8888 \
        --ip=0.0.0.0 \
        --allow-root \
        --no-browser \
        --NotebookApp.custom_display_url=http://localhost:8888"

alias simclr_run_gpu="docker run -it --rm \
    -v $SIMCLR_REPO_DIR:/simclr \
    -v $HOME/.config/gcloud:/root/.config/gcloud \
    -v $SIMCLR_REPO_DIR/data/docker.bash_history:/root/.bash_history \
    --privileged=true \
    --gpus all \
    $DOCKER_IMAGE"

alias simclr_jupyter_gpu="docker run -it --rm \
    --hostname localhost \
    -v $SIMCLR_REPO_DIR:/simclr \
    -v $HOME/.config/gcloud:/root/.config/gcloud \
    -v $SIMCLR_REPO_DIR/data/docker.bash_history:/root/.bash_history \
    -p 0.0.0.0:8888:8888 \
    --gpus all \
    $DOCKER_IMAGE \
    jupyter notebook \
        --port=8888 \
        --ip=0.0.0.0 \
        --allow-root \
        --no-browser \
        --NotebookApp.custom_display_url=http://localhost:8888"
