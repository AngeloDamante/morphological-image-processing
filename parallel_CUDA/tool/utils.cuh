/*
 * utils.cuh
 * Definitions of macro for CUDA compiler.
 *
 *  Created on: 01/mar/2021
 *      Author: AngeloDamante
 *      Special thanks: bertini (my teacher)
 */

#include <iostream>

#ifndef MM_UTILS_CUH
#define MM_UTILS_CUH

static void CheckCudaErrorAux(const char *, unsigned, const char *,
                              cudaError_t);
#define CUDA_CHECK_RETURN(value)                                               \
  CheckCudaErrorAux(__FILE__, __LINE__, #value, value)

static void CheckCudaErrorAux(const char *file, unsigned line,
                              const char *statement, cudaError_t err) {
  if (err == cudaSuccess)
    return;
  std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
            << err << ") at " << file << ":" << line << std::endl;
  exit(1);
}

#endif // MM_UTILS_CUH
