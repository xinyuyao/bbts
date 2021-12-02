#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <map>
#include <mkl/mkl_types.h>
#include <mutex>
#include <pthread.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "../third_party/cuda/gpu.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

using namespace std::chrono;

// kernel definition
__global__ void add_kernel(float *a, float *b, int n) {

  // get our global thread ID
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  // make sure we do not go out of bounds
  if (id < n)
    a[id] += b[id];
}

std::mutex m;

const int32_t num_devices = 4;
const int32_t matrix_size = 20000;
const int32_t block_split = 2;
const int32_t block_size = matrix_size / block_split;

struct block_info_t {

  uint64_t location;
  float *cpu;
  std::vector<float *> gpu;
};

std::map<std::tuple<int32_t, int32_t>, block_info_t> a;
std::map<std::tuple<int32_t, int32_t>, block_info_t> b;
std::map<std::tuple<int32_t, int32_t, int32_t>, block_info_t> c;

float *copy_cpu_to_gpu(block_info_t blk) {
  float *tmp;
  checkCudaErrors(cudaMalloc(&tmp, block_size * block_size * sizeof(float)));
  cudaMemcpyAsync(tmp, blk.cpu, block_size * block_size * sizeof(float),
                  cudaMemcpyHostToDevice);
  return tmp;
}

float *copy_gpu_to_gpu(block_info_t blk) {
  float *tmp;
  checkCudaErrors(cudaMalloc(&tmp, block_size * block_size * sizeof(float)));
  if (blk.location != 0) {
    auto src_dev = __builtin_ctz(blk.location);
    cudaMemcpyAsync(tmp, blk.gpu[src_dev],
                    block_size * block_size * sizeof(float),
                    cudaMemcpyDeviceToDevice);
  } else {
    // this is a fallback in case something went wrong
    cudaMemcpyAsync(tmp, blk.cpu, block_size * block_size * sizeof(float),
                    cudaMemcpyHostToDevice);
  }
  return tmp;
}

void product_thread(int32_t device, int32_t rowID, int32_t colID) {

  cudaSetDevice(device);
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);

  float *a_blk, *b_blk, *c_blk;
  for (auto k = 0; k < block_split; ++k) {

    // if this is the first block
    if (k == 0) {

      // copy the stuff
      {
        std::unique_lock<std::mutex> lck(m);
        auto it = a.find({rowID, k});
        auto jt = b.find({k, colID});

        a_blk = copy_cpu_to_gpu(it->second);
        b_blk = copy_cpu_to_gpu(jt->second);
      }

      // sync
      cudaDeviceSynchronize();

      {
        std::unique_lock<std::mutex> lck(m);

        auto it = a.find({rowID, k});
        auto jt = b.find({k, colID});

        // set the location for a
        it->second.location |= (1 << device);
        it->second.gpu[device] = a_blk;

        // set the location for b
        jt->second.location |= (1 << device);
        jt->second.gpu[device] = b_blk;
      }
    } else {

      // sync the gpu copy from the previous iteration
      cudaDeviceSynchronize();

      {
        std::unique_lock<std::mutex> lck(m);

        auto it = a.find({rowID, k});
        auto jt = b.find({k, colID});

        // set the location for a
        it->second.location |= (1 << device);
        it->second.gpu[device] = a_blk;

        // set the location for b
        jt->second.location |= (1 << device);
        jt->second.gpu[device] = b_blk;
      }
    }

    // do the multiply
    float alpha = 1.0f;
    float beta = 0.0f;
    checkCudaErrors(cudaMalloc(&c_blk, block_size * block_size * sizeof(float)));
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, block_size, block_size,
                block_size, &alpha, a_blk, block_size, b_blk, block_size, &beta,
                c_blk, block_size);

    {
      std::unique_lock<std::mutex> lck(m);
      auto it = c.find({rowID, k, colID});
      it->second.gpu[device] = c_blk;
      it->second.location |= (1 << device);
    }

    {
      // copy the stuff
      {
        std::unique_lock<std::mutex> lck(m);
        auto it = a.find({rowID, k});
        auto jt = b.find({k, colID});

        a_blk = copy_gpu_to_gpu(it->second);
        b_blk = copy_gpu_to_gpu(jt->second);
      }
    }
  }

  cudaDeviceSynchronize();

  for (auto k = 1; k < block_split; ++k) {

    auto left = c.find({rowID, 0, colID});
    auto right = c.find({rowID, k, colID});

    // number of thread blocks in grid
    uint32_t threads_num = 1024;
    uint32_t n = block_size * block_size;
    uint32_t grid_size = (int)ceil((float)n / threads_num);

    add_kernel<<<grid_size, threads_num, 0>>>(left->second.gpu[device],
                                             right->second.gpu[device], n);
    gpuErrchk( cudaPeekAtLastError() );

    checkCudaErrors(cudaDeviceSynchronize());
  }
}

void set_to_one(float *blk) {
  for (size_t idx = 0; idx < block_size * block_size; ++idx) {
    blk[idx] = 1.0f;
  }
}

int main() {

  for (int idx = 0; idx < block_split; idx++) {
    for (int jdx = 0; jdx < block_split; jdx++) {

      float *a_blk = (float *)malloc(sizeof(float) * block_size * block_size);
      set_to_one(a_blk);
      auto &a_nfo = a[{idx, jdx}];
      a_nfo.cpu = a_blk;
      a_nfo.location = 0;
      a_nfo.gpu.resize(num_devices);

      float *b_blk = (float *)malloc(sizeof(float) * block_size * block_size);
      set_to_one(b_blk);
      auto &b_nfo = b[{idx, jdx}];
      b_nfo.cpu = b_blk;
      b_nfo.location = 0;
      b_nfo.gpu.resize(num_devices);

      // init the info for the c_blocks
      for (int kdx = 0; kdx < block_split; kdx++) {
        auto &c_nfo = c[{idx, kdx, jdx}];
        c_nfo.cpu = nullptr;
        c_nfo.location = 0;
        c_nfo.gpu.resize(num_devices);
      }
    }
  }

  auto start = high_resolution_clock::now();
  std::vector<std::thread> threads;
  for (auto dev = 0; dev < num_devices; ++dev) {
    threads.emplace_back(std::thread([dev]() {
      auto rowID = dev % 2;
      auto colID = dev / 2;
      product_thread(dev, rowID, colID);
    }));
  }

  for (auto &t : threads) {
    t.join();
  }
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  std::cout << "Time it took : " << (float)duration.count() * 1e-6f << std::endl;

  for (auto dev = 0; dev < num_devices; ++dev) {

    auto rowID = dev % 2;
    auto colID = dev / 2;
    // std::cout << "Value : " << c[{rowID, 0, colID}].gpu[dev][0] << "\n";
  }

  return 0;
}