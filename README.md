# Morphological Image Processing in Parallel Version.
Mathematical morphology (MM) technique implemented in sequential and parallel methods for study purposes.

## Directory Structure
```bash 
├── images
│   ├── examples            # images to processing
│   │   ├── ...
│   ├── seqResults          # results by CPU
│   │   ├── ...
│   ├── gpuResults          # results by GPU
│   │   ├── ...
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
├── parallel_JAVA           # handled by @Fabian Greavu
```

# Authors
+ <a href="https://github.com/AngeloDamante"> Angelo D'Amante </a>
+ <a href="https://github.com/fabian57fabian"> Fabian Greavu </a>
