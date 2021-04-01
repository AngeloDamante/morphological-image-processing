FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 as base
RUN apt-get update && apt-get install -y nano && apt-get install -y wget

# examples images by @AngeloDamante and @Fabian57Fabian
COPY ["download_files.sh", "download_files.sh"]
RUN chmod +x download_files.sh
COPY ["expand_images.py", "expand_images.py"]

# Python
ENV PYTHONPATH "${PYTHONPATH}:/parallel_CUDA:/images:/results_CUDA"
RUN apt-get update && apt install -y python3-pip && pip3 install numpy
RUN pip3 install Pillow

# CPP
WORKDIR /parallel_CUDA
RUN apt-get update && apt-get install -y cmake
RUN apt-get install -y libx11-dev && apt-get install -y libpng-dev
