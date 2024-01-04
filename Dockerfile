## Dockerfile

# start from the PyTorc base image for our version of PT + CUDA
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
# FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# set cwd
WORKDIR /usr/src/app

# copy the repo into the container at this directory\
COPY . .

RUN apt-get update && apt-get install -y git \
                                         vim \
                                         rsync \
                                         tmux

COPY .vimrc /root/.vimrc

ENV CUDA_HOME=/usr/local/cuda

# do pip installs
RUN pip install --no-cache-dir -r docker_requirements.txt

# Start the container
SHELL ["/bin/bash", "--login", "-c"]
CMD [ "sleep infinity" ]
