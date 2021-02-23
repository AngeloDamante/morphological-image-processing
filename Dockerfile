FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 as base
WORKDIR /parallelVersion
RUN apt-get update && apt-get install -y nano
RUN apt-get install -y cmake
RUN apt-get install -y libx11-dev && apt-get install -y libpng-dev
