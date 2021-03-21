#include <thread>
#include <iostream>
#include <vector>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>  
#include <cublas_v2.h>
#include <cublasLt.h>
#include "../third_party/cuda/gpu.h"

int main() {

  
  float alpha=1.0f;
  float beta=0.0f;

  const int N = 10000;
  const int split = 4;

  float* a[split]; 
  checkCudaErrors(cudaMallocManaged(&a[0], N * N * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(&a[1], N * N * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(&a[2], N * N * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(&a[3], N * N * sizeof(float)));

  float* b[split][split];
  for(int r = 0; r < split; r++) {
    for(int c = 0; c < split; c++) {
      checkCudaErrors(cudaMallocManaged(&b[r][c], N * N * sizeof(float)));
    }
  }

  for(int c = 0; c < split; c++) {
    for(int r = 0; r < split; r++) {
      for(auto i = 0; i < N * N; ++i) {
        b[r][c][i] = (float) (i + r * split + c);
      }
    }

    for(auto i = 0; i < N * N; ++i) {
        a[c][i] = (float) (i + 1);
    }
  }

  std::vector<std::tuple<int, int>> muls;
  for(int t = 0; t < split; t++) {
    for(int c = 0; c < split; c++) {
      muls.push_back({t, c});
    }
  }

  std::random_shuffle ( muls.begin(), muls.end() );

  // create the front stream (at the current step fetch the matrix)
  cudaStream_t s_front;
  cudaStreamCreate(&s_front);
  cublasHandle_t h_front; cublasCreate(&h_front); cublasSetStream(h_front, s_front);

  cudaEvent_t e_front;
  cudaEventCreate(&e_front);

  float *o_front = nullptr;
  for(int32_t i = 0; i < muls.size(); ++i) {

    auto t = std::get<0>(muls[i]);
    auto c = std::get<1>(muls[i]);

    // allocate the output memory
    checkCudaErrors(cudaMallocManaged(&o_front, N * N * sizeof(float)));

    // do the multiply
    cublasSgemm(h_front, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, a[t], N, b[t][c], N, &beta, o_front, N);
    cudaEventRecord(e_front, s_front); 

    // sync to wait for the prefetch
    cudaEventSynchronize(e_front);
  }

  cudaProfilerStop();

  return 0;
}