#!/bin/bash

GREEN='\033[0;32m'
NC='\033[0m'

# Preparing
echo -e "${GREEN}PREPARATION...${NC}"
./download_files.sh
cd parallel_CUDA/build
cmake .. -DCMAKE_FLAGS=VALUES
make
echo -e "${GREEN}...all ready!${NC}"

# Produce timings
echo -e "${GREEN}seuquential version${NC}"
./mm_sequential

echo -e "${GREEN}parallel NAIVE version${NC}"
./mm_naive

echo -e "${GREEN}parallel OPTIMIZED version${NC}"
./mm_sharedOpt

echo -e "${GREEN} ---- TIMINGS PRODUCED ---- ${NC}"

# Produce speedups
cd ..
cd ..
cd results_CUDA
python3 evaluate_speedup.py

echo -e "${GREEN} ---- SPEEDUP OF ${NUM_THREAD_PER_AXIS} PRODUCED ---- ${NC}"
