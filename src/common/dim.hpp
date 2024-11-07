#ifndef DIM_HPP
#define DIM_HPP
#include <cuda_runtime.h>

#define GPU_BLOCK_THREADS 512

dim3 grid_dims(int numJobs); {
  int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
  return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
}

dim3 block_dims(int numJobs); {
  return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
}

int upbound(int n, int align = 32); { return (n + align - 1) / align * align; }

#endif