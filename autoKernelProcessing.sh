#!/bin/bash
echo "PREPARATION..."
./download_files.sh
cd parallel_CUDA/build
cmake .. -DCMAKE_FLAGS=VALUES
make
echo "...all ready!"

echo "seuquential version"
./mm_sequential

echo "parallel NAIVE version"
./mm_naive

echo "parallel OPTIMIZED version"
./mm_sharedOpt
