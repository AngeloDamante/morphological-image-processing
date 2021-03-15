/*
 * MM.cuh
 * Header to declare functions for parallel version of MM.
 *
 * Versions implemented:
 *      - NAIVE: without any optimization
 *      - SHAREDOPT: optimization with shared memory and padding
 *
 * A Parallel Program Structure:
 *      1. Alloc GPU memories
 *      2. Copy data from host to device
 *      3. Compute kernel function
 *      4. Copy data from device to host
 *      5. Destroy all GPU memories allocated
 *
 *  Created on: 01/mar/2021
 *      Author: AngeloDamante
 */

#ifndef MM_MM_CUH
#define MM_MM_CUH

#include "Image.h"
#include "Probe.h"
#include <iostream>

enum MMop { DILATATION, EROSION, OPENING, CLOSING };
enum Version { NAIVE, SHAREDOPT };

const int MASK_RADIUS = 1;
const int MASK_WIDTH = MASK_RADIUS * 2 + 1;  // PROBE 3x3
const int TILE_WIDTH = 32;  // WARP-SIZE
__constant__ float probeDataD[MASK_WIDTH*MASK_WIDTH];  // To alloc probe

/// Interface
__host__ Image* mm(Image* image, Probe* probe, MMop mmOp, Version vrs);

/// Operations
__host__ Image* erosion(Image* image, Probe* probe, Version vrs);
__host__ Image* dilatation(Image* image, Probe* probe, Version vrs);
__host__ Image* opening(Image* image, Probe* probe, Version vrs);
__host__ Image* closing(Image* image, Probe* probe, Version vrs);

__global__ void __foo(float* imgData, float* outData, int imgH, int imgW);

namespace naive{
    __global__
    void __process(float* imgData, const float* __restrict__ prbData,
        float* outData, int imgH, int imgW, int prbH, int prbW, MMop mmOp);
} // naive

namespace sharedOpt{
    __global__
    void __process(float* imgData, const float* __restrict__ prbData,
        float* outData, int imgH, int imgW, int prbH, int prbW, MMop mmOp);
} // sharedOpt

namespace paddingOpt{
    __global__
    void __process(float* imgData, const float* __restrict__ prbData,
        float* outData, int imgH, int imgW, int prbH, int prbW, MMop mmOp);
} // paddingOpt

#endif // MM_MM_CUH
