FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV PYTHONPATH /home/adityac/lsm:$PYTHONPATH

RUN apt-get update && apt-get install -y \
    git \
    python2.7 \
    python-pip \
    python3-pip \
    tar \
    vim

RUN pip2 install virtualenv && \
    virtualenv -p python2.7 /home/env && \
    source /home/env/bin/activate && \
    pip2 install numpy \
    scipy \
    tqdm \
    tabulate \
    scikit-image \
    matplotlib \
    plyfile \
    tensorflow-gpu==1.15 \
    jupyter
