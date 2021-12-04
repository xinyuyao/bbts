#include <atomic>
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

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
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
const int32_t matrix_size = 40000;
const int32_t block_split = 4;
const int32_t block_size = matrix_size / block_split;

// 0 1 2 3
// 0 1 2 3
// 0 1 2 3
// 0 1 2 3
std::vector<std::tuple<int32_t, int32_t>> four_gpu_schedule = {
    {0, 0}, {1, 1}, {2, 2}, {3, 3}, 
    {1, 0}, {2, 1}, {3, 2}, {0, 3},
    {2, 0}, {3, 1}, {0, 2}, {1, 3}, 
    {3, 0}, {0, 1}, {1, 2}, {2, 3}};

// 0 1 2
// 0 1 2
// 0 1 2
std::vector<std::tuple<int32_t, int32_t>> three_gpu_schedule = {
  {0, 0}, {1, 1}, {2, 2},
  {1, 0}, {2, 1}, {0, 2},
  {0, 1}, {1, 2}, {2, 0}
};

// 0 1
// 0 1
std::vector<std::tuple<int32_t, int32_t>> two_gpu_schedule = {
  {0, 0}, {1, 1},
  {1, 0}, {0, 1},
};

std::vector<std::tuple<int32_t, int32_t>> one_gpu_schedule = {
  {0, 0},
};

std::vector<std::vector<std::tuple<int32_t, int32_t>>> schedules {
  one_gpu_schedule,
  two_gpu_schedule,
  three_gpu_schedule,
  four_gpu_schedule
};

struct block_info_t {

  uint64_t location;
  float *cpu;
  std::vector<float *> gpu;
};

std::map<std::tuple<int32_t, int32_t>, block_info_t> a;
std::map<std::tuple<int32_t, int32_t>, block_info_t> b;
std::map<std::tuple<int32_t, int32_t>, float *> c;

std::atomic_int32_t num_cpu_transfers;
std::atomic_int32_t num_gpu_transfers;

std::vector<std::vector<char>> order {
  {0, 1, 2, 3},
  {1, 2, 3, 0},
  {2, 3, 0, 1},
  {3, 0, 1, 2},
};

float *copy_to_device(int32_t dev, block_info_t &blk) {
  float *tmp = nullptr;
  checkCudaErrors(cudaMalloc(&tmp, block_size * block_size * sizeof(float)));
  if (blk.location != 0) {

    int32_t src_dev;
    for(auto test_dev : order[dev]) {
      if((blk.location & (1 << test_dev)) != 0) {
        src_dev = test_dev;
        break;
      }
    }

    checkCudaErrors(cudaMemcpyPeerAsync(tmp, dev, 
                                        blk.gpu[src_dev], src_dev, 
                                        block_size * block_size * sizeof(float)));

    num_gpu_transfers++;
  } else {
    // this is a fallback in case something went wrong
    checkCudaErrors(cudaMemcpyAsync(tmp, blk.cpu,
                                    block_size * block_size * sizeof(float),
                                    cudaMemcpyHostToDevice));
    num_cpu_transfers++;
  }
  return tmp;
}



void prefetch(int32_t dev, int32_t i, int32_t j, int32_t k) {

  std::unique_lock<std::mutex> lck(m);

  auto at = a.find({i, k});
  auto bt = b.find({k, j});

  // do we already have it
  if((at->second.location & (1 << dev)) == 0) {
    auto a_blk = copy_to_device(dev, at->second);
    at->second.gpu[dev] = a_blk;  
  }

  // do we already have it
  if((bt->second.location & (1 << dev)) == 0) {
    auto b_blk = copy_to_device(dev, bt->second);
    bt->second.gpu[dev] = b_blk;
  }
}

void do_muliply(cublasHandle_t &cublas_handle, int32_t dev, int32_t i,
                int32_t j, int32_t i_next, int32_t j_next) {

  float *c_blk = nullptr;
  for (auto k = 0; k < block_split; ++k) {

    // sync so we get the previous
    checkCudaErrors(cudaDeviceSynchronize());

    if(k + 1 < block_split) {
      // prefetch the next block otherwise
      prefetch(dev, i, j, k + 1);
    }
    else if(i_next != -1 && j_next != -1) {
      // prefetch the next block if this is the last block
      prefetch(dev, i_next, j_next, 0);
    }

    // at this point the previous block is synced
    float *a_blk, *b_blk;
    {
      std::unique_lock<std::mutex> lck(m);

      auto at = a.find({i, k});
      auto bt = b.find({k, j});

      at->second.location = at->second.location | (1 << dev);
      bt->second.location = bt->second.location | (1 << dev);

      assert(at->second.location != 0);
      assert(bt->second.location != 0);

      a_blk = at->second.gpu[dev];
      b_blk = bt->second.gpu[dev];
    }

    // free the block if necessary
    if(c_blk != nullptr) {
      cudaFree(c_blk);
    }

    // do the multiply
    float alpha = 1.0f;
    float beta = 0.0f;
    checkCudaErrors(
        cudaMalloc(&c_blk, block_size * block_size * sizeof(float)));
    checkCudaErrors(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                block_size, block_size, block_size, &alpha,
                                a_blk, block_size, b_blk, block_size, &beta,
                                c_blk, block_size));

    // do we need to preform an add
    float *ct;
    {
      std::unique_lock<std::mutex> lck(m);

      ct = c[{i, j}];
      if (ct == nullptr) {
        c[{i, j}] = c_blk;
        c_blk = nullptr;
        continue;
      }
    }

    // sync here to do the add
    checkCudaErrors(cudaDeviceSynchronize());

    // number of thread blocks in grid
    uint32_t threads_num = 1024;
    uint32_t n = block_size * block_size;
    uint32_t grid_size = (int)ceil((float)n / threads_num);

    // sum the stuff
    assert(ct != nullptr);
    assert(c_blk != nullptr);
    add_kernel<<<grid_size, threads_num, 0>>>(ct, c_blk, n);
  }
}

void product_thread(int32_t dev,
                    std::vector<std::tuple<int32_t, int32_t>> &final_blks) {

  // set the device
  cudaSetDevice(dev);
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);

  for(auto peer = 0; peer < num_devices; ++peer) {
    if(peer != dev) {
      cudaDeviceEnablePeerAccess(peer, 0);
    }
  }

  prefetch(dev, std::get<0>(final_blks[0]), std::get<0>(final_blks[0]), 0);
  for (auto idx = 0; idx < final_blks.size(); ++idx) {

    int32_t i = std::get<0>(final_blks[idx]);
    int32_t j = std::get<1>(final_blks[idx]);

    //
    int32_t i_next = -1, j_next = -1;
    if (idx + 1 < final_blks.size()) {
      i_next = std::get<0>(final_blks[idx + 1]);
      j_next = std::get<1>(final_blks[idx + 1]);
    }

    // do the multiply
    do_muliply(cublas_handle, dev, i, j, i_next, j_next);
  }

  checkCudaErrors(cudaDeviceSynchronize());
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
    }
  }

  auto &schedule = schedules[num_devices - 1];
  std::vector<std::vector<std::tuple<int32_t, int32_t>>> run_these;
  run_these.resize(num_devices);
  for (int idx = 0; idx < block_split; idx += num_devices) {
    for (int jdx = 0; jdx < block_split; jdx += num_devices) {
      for (const auto &offset : schedule) {
        auto rowID = idx + std::get<0>(offset);
        auto colID = jdx + std::get<1>(offset);
        
        if(colID >= block_split || rowID >= block_split ) {
          continue;
        }

        run_these[std::get<1>(offset)].push_back({rowID, colID});
      }
    }
  }

  auto start = high_resolution_clock::now();
  std::vector<std::thread> threads;
  for (auto dev = 0; dev < num_devices; ++dev) {
    threads.emplace_back(std::thread(
        [&run_these, dev]() { product_thread(dev, run_these[dev]); }));
  }

  for (auto &t : threads) {
    t.join();
  }
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  std::cout << "Time it took : " << (float)duration.count() * 1e-6f
            << std::endl;

  std::cout << "CPU transfers in total : " << num_cpu_transfers << "\n";
  std::cout << "GPU transfers in total : " << num_gpu_transfers << "\n";

  for (int idx = 0; idx < block_split; idx++) {
    for (int jdx = 0; jdx < block_split; jdx++) {
      // std::cout << "Value : " << c[{idx, jdx}][0] << "\n";
    }
  }

  return 0;
}