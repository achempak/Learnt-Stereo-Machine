#!/bin/bash

sudo docker run -it --rm -p 8888:8888 \
--gpus all \
-v "$(pwd)":/home/adityac/lsm \
nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
