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


// kernel definition
__global__ void dense_add_kernel(float *a, float *b, float *c, int n) {

  // get our global thread ID
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  // make sure we do not go out of bounds
  if (id < n)
      c[id] = a[id] + b[id];
}


void rotate_streams(cudaStream_t &s_front, cublasHandle_t &h_front, 
                    cudaStream_t &s_mid,   cublasHandle_t &h_mid,
                    cudaStream_t &s_back,  cublasHandle_t &h_back) {

  cudaStream_t s_tmp = s_back;
  s_back = s_mid;
  s_mid = s_front;
  s_front = s_tmp;

  cublasHandle_t h_tmp = h_back;
  h_back  = h_mid;
  h_mid   = h_front;
  h_front = h_tmp;
}

void rotate_events(cudaEvent_t &e_front, 
                   cudaEvent_t &e_mid,
                   cudaEvent_t &e_back) {

  cudaEvent_t e_tmp = e_back;
  e_back  = e_mid;
  e_mid   = e_front;
  e_front = e_tmp;
}

void rotate_tensors(float* &o_front, 
                    float* &o_mid,
                    float* &o_back) {

  float* o_tmp = o_back;
  o_back  = o_mid;
  o_mid   = o_front;
  o_front = o_tmp;
}

int main() {

  
  float alpha=1.0f;
  float beta=0.0f;

  const int N = 10000;
  const int split = 4;
  //const int thread_num = split * split;

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

  auto t = std::get<0>(muls[0]);
  auto c = std::get<1>(muls[0]);

  // create the front stream (at the current step fetch the matrix)
  cudaStream_t s_front;
  cudaStreamCreate(&s_front);
  cublasHandle_t h_front; cublasCreate(&h_front); cublasSetStream(h_front, s_front);

  // create the mid strea (at the current step do the multiply)
  cudaStream_t s_mid;
  cudaStreamCreate(&s_mid);
  cublasHandle_t h_mid; cublasCreate(&h_mid); cublasSetStream(h_mid, s_mid);

  // create the back stream (at the current step does unloading)
  cudaStream_t s_back;
  cudaStreamCreate(&s_back);
  cublasHandle_t h_back; cublasCreate(&h_back); cublasSetStream(h_back, s_back);

  cudaEvent_t e_front;
  cudaEventCreate(&e_front);

  cudaEvent_t e_mid;
  cudaEventCreate(&e_mid);

  cudaEvent_t e_back;
  cudaEventCreate(&e_back);

  // prefetch on front
  cudaMemPrefetchAsync(a[t],    N * N * sizeof(float), 0, s_front);
  cudaMemPrefetchAsync(b[t][c], N * N * sizeof(float), 0, s_front);
  cudaEventRecord(e_front, s_front); 

  // rotate the streams
  rotate_streams(s_front, h_front, s_mid, h_mid, s_back, h_back);
  rotate_events(e_front, e_mid, e_back);

  float *o_front = nullptr;
  float *o_mid = nullptr;
  float *o_back = nullptr;

  for(int32_t i = 0; i < muls.size(); ++i) {

    // allocate the output memory
    checkCudaErrors(cudaMallocManaged(&o_mid, N * N * sizeof(float)));

    // sync to wait for the prefetch
    cudaEventSynchronize(e_front);
    cudaEventSynchronize(e_mid);
    cudaEventSynchronize(e_back);

    // do the multiply
    cublasSgemm(h_mid, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, a[t], N, b[t][c], N, &beta, o_mid, N);
    cudaEventRecord(e_mid, s_mid); 
    
    if(i > 0) {
      
      // load out on back
      cudaMemPrefetchAsync(o_back, N * N * sizeof(float), cudaCpuDeviceId, s_back); 
      cudaEventRecord(e_back, s_back); 
    }

    if(i + 1 < muls.size()) {

      // get the index
      t = std::get<0>(muls[i + 1]);
      c = std::get<1>(muls[i + 1]);

      // prefetch on front
      cudaMemPrefetchAsync(a[t],    N * N * sizeof(float), 0, s_front);
      cudaMemPrefetchAsync(b[t][c], N * N * sizeof(float), 0, s_front);
      cudaEventRecord(e_front, s_front); 
    }

    // rotate the streams
    rotate_streams(s_front, h_front, s_mid, h_mid, s_back, h_back);
    rotate_events(e_front, e_mid, e_back);
    rotate_tensors(o_front, o_mid, o_back);
  }

  cudaProfilerStop();

  return 0;
}