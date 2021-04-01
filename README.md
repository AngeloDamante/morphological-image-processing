# Morphological Image Processing in Parallel Version.
Mathematical morphology (<a href="https://en.wikipedia.org/wiki/Mathematical_morphology">MM</a>) technique implemented in sequential and parallel methods for study purposes.

## Directories Layout
```bash 
├── images
│   ├── examples            # images to processing
├── parallel_CUDA           # handled by @Angelo D'Amante
│   ├── CMakeLists.txt
│   ├── sequential_CPP
│   │   ├── main.cpp        # sequential main
│   ├── CUDA
│   │   ├── naive.cu        # parallel main v1
│   │   ├── sharedOpt.cu    # parallel main v2
│   ├── tool                
│   │   ├── Image.h
│   │   ├── utils.cuh
│   │   ├── ...
│   ├── include         
│   │   ├── MathematicalMorphology.h    # src for CPU processing 
│   │   ├── MM.cuh                      # src for GPU processing
│   │   ├── Probe.h                     # src for probe element
│   │   ├── ...
├── results_CUDA            # csv files with speedup for each parallel version
│   ├── evaluate_speedup.py # script to compute speedups
│   ├── ...
├── parallel_JAVA           # handled by @Fabian Greavu
```

## Requirements 
For parallel computing with CUDA:
- Ubuntu >= 16.04
- NVIDIA Container Toolkit (<a href="https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker"> installation guide</a>)
- Docker Compose

## CUDA Computing with Docker
```
git clone https://github.com/AngeloDamante/morphological-image-processing-in-parallel.git
cd morphological-image-processing-in-parallel
docker-compose build
```

## Threads and Blocks managing for mapping image in CUDA
<p align="center">
  <img src="https://user-images.githubusercontent.com/62060759/113345604-f996f580-9332-11eb-8df0-9f7e1cd3db48.png" width="600"/>
</p>

## RUN MM operations with container in CUDA
To evaluate speedup with **default images**, run MM operations with automatic Kernel that produce results in csv on results_CUDA directory. You can choose number of threads to launch per block setting the number of thread per axis as an environment variable passed as argument at the container. Default value for this environment variable is 32 (1024 thread per block).  
```bash
# Launch 256 threads per block
docker-compose run --rm -e NUM_THREAD_PER_AXIS=16 autoKernelMM

# Launch 576 threads per block
docker-compose run --rm -e NUM_THREAD_PER_AXIS=24 autoKernelMM

# Launch 1024 threads per block
docker-compose run --rm autoKernelMM
```
In according with CUDA policy, the number of threads per block must be a multiple of warp-size (32).

This service produce:
- timings csv for sequential and parallel version. 
- speedups csv with chosen number of threads per block.

Ctrl+D to exit from the container.

## Alternative services in CUDA
It's possible to process not default images and then evaluate speedup. But remember to set number threads per block (32 as default).
```bash
# After loading your own images in 'images' folder

# 256 thread per block for example
docker-compose run --rm -e NUM_THREAD_PER_AXIS=16 kernelMM
mkdir -p build
cd build
cmake .. && make

# After run the three main builded, start container to evaluate speedup
docker-compose run --rm evaluateSpeedup
```
At the end, will be the speedups file.csv in results_CUDA folder.

Ctrl+D to exit from the container.

## Authors
+ <a href="https://github.com/AngeloDamante"> Angelo D'Amante </a>
+ <a href="https://github.com/fabian57fabian"> Fabian Greavu </a>
