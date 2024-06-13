FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
MAINTAINER zetang "dg20320005@smail.nju.edu.cn"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    software-properties-common && \
    rm -rf /var/lib/apt/lists/*
RUN add-apt-repository ppa:deadsnakes/ppa

############################################################
# Common steps (must be the same in the CPU and GPU images)

RUN apt-get update && apt-get install -y \
    apt-transport-https \
    iputils-ping \
    git \
    curl \
    build-essential \
    cmake \
    libhdf5-dev \
    swig \
    wget \
    unzip \
    python3.7 \
    python3.7-venv \
    python3.7-dev \
    python3-pip \
    python3-software-properties \
    openjdk-8-jdk && \
    rm -rf /var/lib/apt/lists/*

# Set Python3.7 as the default python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

RUN curl https://bootstrap.pypa.io/get-pip.py | python3.7

### Without this Python thinks we're ASCII and unicode chars fail
ENV LANG C.UTF-8

## Python packages
RUN pip3 install -U pip

## Torch 1.12.1 + CUDA10.2
RUN python3 -m pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102

## install libraries from the requirements.txt
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

## Download Code Language Models from Huggingface.
COPY huggingface .

## COPY PYTHON REPO FILES.
COPY Python.zip .
RUN unzip Python.zip && rm Python.zip

## DOWNLOAD NLTK PUNK.
COPY download_punkt.py .
RUN python3 download_punkt.py