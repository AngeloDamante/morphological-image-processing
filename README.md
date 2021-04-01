# Morphological Image Processing in Parallel Version.
Mathematical morphology (MM) technique implemented in sequential and parallel methods for study purposes.

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

## CUDA Computing
```
git clone https://github.com/AngeloDamante/morphological-image-processing-in-parallel.git
cd morphological-image-processing-in-parallel
docker-compose build
```
<!-- devo inserire l'immagine-->

## Authors
+ <a href="https://github.com/AngeloDamante"> Angelo D'Amante </a>
+ <a href="https://github.com/fabian57fabian"> Fabian Greavu </a>
