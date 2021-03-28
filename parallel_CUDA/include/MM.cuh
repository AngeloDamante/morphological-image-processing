/*
 * MM.cuh
 * Header to declare functions for parallel version of MM.
 *
 * Versions implemented:
 *      - NAIVE: without any optimization forms.
 *      - SHAREDOPT: optimization with shared memory and padding for tailing.
 *
 * Parallel Program Structure:
 *      1. Alloc GPU memories
 *      2. Copy data from host to device
 *      3. Compute kernel function
 *      4. Copy data from device to host
 *      5. Destroy all GPU memories allocated
 *
 * @Author: AngeloDamante
 * @mail: angelo.damante16@gmail.com
 * @GitHub: https://github.com/AngeloDamante
 */

#ifndef MM_MM_CUH
#define MM_MM_CUH

#include "Image.h"
#include "Probe.h"
#include <iostream>
#include <map>

const int MASK_RADIUS = 1;
const int MASK_WIDTH = MASK_RADIUS * 2 + 1;  // PROBE 3x3
const int TILE_WIDTH = 32;
const int W = TILE_WIDTH + MASK_WIDTH - 1;
__constant__ float probeDataD[MASK_WIDTH*MASK_WIDTH];  // To alloc probe

enum MMop { DILATATION, EROSION, OPENING, CLOSING };
enum Version { NAIVE, SHAREDOPT };
const std::map<MMop, std::string> MMoperations = \
{
    {DILATATION, "Dilatation"},
    {EROSION, "Erosion"},
    {CLOSING, "Closing"},
    {OPENING, "Opening"}

};


/// Interface
__host__ Image* mm(Image* image, Probe* probe, MMop mmOp, Version vrs);

/// Operations
__host__ Image* erosion(Image* image, Probe* probe, Version vrs);
__host__ Image* dilatation(Image* image, Probe* probe, Version vrs);
__host__ Image* opening(Image* image, Probe* probe, Version vrs);
__host__ Image* closing(Image* image, Probe* probe, Version vrs);

__global__ void __foo(float* imgData, float* outData, int imgH, int imgW);

namespace naive{

    /**
     * Process to compute naive parallel solution without any optimized forms.
     *
     * This solution consists of two for-cycle to read values of probe
     * element's mask. Thus, only neighborhood of pixel are considered.
     * The max and min value are computed at run-time.
     *
     * This solution presents delays due to global memory accesses for
     * imgData element.
     *
     * @param float* imgData: input image buffer stored in global memory.
     * @param float* outData: output image buffer stored in global memory.
     * @param int imgH, imgW, prbH, prbW are properties of image and probe.
     * @param MMop mmOp: enum type of MM operation.
     *
     * @return void in accord with CUDA rules for __global__ function.
    */
    __global__
    void __process(float* imgData, const float* __restrict__ prbData,
        float* outData, int imgH, int imgW, int prbH, int prbW, MMop mmOp);

} // naive

namespace sharedOpt{
    __global__
    void __process(float* imgData, const float* __restrict__ prbData,
        float* outData, int imgH, int imgW, int prbH, int prbW, MMop mmOp);
} // sharedOpt

#endif // MM_MM_CUH
