#ifndef DIM_HPP
#define DIM_HPP
#include <cuda_runtime.h>

#define GPU_BLOCK_THREADS 512

dim3 grid_dims(int numJobs); 

dim3 block_dims(int numJobs); 

int upbound(int n, int align = 32); 

#endif
