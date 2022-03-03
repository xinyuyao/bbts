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
#include <memory>
#include <mkl/mkl_types.h>
#include <mutex>
#include <pthread.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <condition_variable>
#include <vector>
#include "../src/utils/concurent_queue.h"
#include <deque>

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

enum steam_type_t { MULT = 0, ADD = 1, COPY = 2 };

struct stream_t {
  cudaStream_t stream;
  cudaEvent_t event;
};

// kernel definition
__global__ void add_kernel(float *a, float *b, int n) {

  // get our global thread ID
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  // make sure we do not go out of bounds
  if (id < n)
    a[id] += b[id];
}

std::mutex m;

struct block_info_t {
  std::mutex m;
  char matrix;
  std::vector<bool> location;
  float *cpu;
  std::vector<float *> gpu;
};

const int32_t num_devices = 4;
const int32_t matrix_size = 40000;
const int32_t block_split = 10;
const int32_t block_size = matrix_size / block_split;

struct pq_entry_t {
  int32_t left;
  int32_t right;
};

// keeps track of all the tensors
std::vector<block_info_t> tensors;

// the join groups
struct join_group_t {
  int32_t num_remaining;
  int32_t left;
  int32_t right;
};
std::vector<join_group_t> join_groups;

// basically keeps a pq that ranks the entries by two criteria
// we sort them first by the number of kernels it will trigger
// and then if they trigger an equal number of kernels we sort them by the
// number of join groups what will use them
struct tensor_pq_entry_t {
  int32_t num_to_trigger = 0;
  int32_t num_use = 0;
};

struct comp {
  bool operator()(const tensor_pq_entry_t &a, const tensor_pq_entry_t &b) {
    if (a.num_to_trigger == b.num_to_trigger) {
      return a.num_use > b.num_use;
    }
    return a.num_to_trigger > b.num_to_trigger;
  };
};

using pq_t = std::multimap<tensor_pq_entry_t, int32_t, comp>;
std::multimap<tensor_pq_entry_t, int32_t, comp> tensor_pq;
std::unordered_map<int32_t, pq_t::iterator> tensor_pq_mapping;

// the mapping of the tensors to the join groups
std::unordered_map<int32_t, std::vector<int32_t>> tid_mapping;

std::vector<bbts::concurent_queue<int32_t>> to_prep;

std::atomic_int32_t num_cpu_transfers;
std::atomic_int32_t num_gpu_transfers;
std::vector<std::vector<char>> order{
    {0, 1, 2, 3},
    {1, 2, 3, 0},
    {2, 3, 0, 1},
    {3, 0, 1, 2},
};

// maps the joing group to the aggregation group it belongs to
std::map<std::tuple<int32_t, int32_t>, int> agg_groups;

// where the aggregation groups are mapped
std::vector<int32_t> agg_group_mappings;

void move_to_gpu(int32_t dev, int32_t tensor_id, cudaEvent_t &event,
                 cudaStream_t &stream) {



  // set the device
  cudaSetDevice(dev);

  // lock so we can access the the tensor structure
  std::unique_lock<std::mutex> lck(tensors[tensor_id].m);

  // get the block
  auto &blk = tensors[tensor_id];

  // malloc the block on the GPU
  float *tmp = nullptr;
  checkCudaErrors(cudaMalloc(&tmp, block_size * block_size * sizeof(float)));
  blk.gpu[dev] = tmp;

  // this is a fallback in case something went wrong
  num_cpu_transfers++;
  checkCudaErrors(cudaMemcpyAsync(tmp, blk.cpu,
                                  block_size * block_size * sizeof(float),
                                  cudaMemcpyHostToDevice, stream));

  // mark this so we know when this is done
  cudaEventRecord(event, stream);
}

float *fetch_from_gpu(stream_t &stream, int32_t dev, block_info_t &blk) {

  float *tmp = nullptr;
  checkCudaErrors(cudaMalloc(&tmp, block_size * block_size * sizeof(float)));

  int32_t src_dev;
  for (auto test_dev : order[dev]) {
    if (blk.location[test_dev]) {
      src_dev = test_dev;
      break;
    }
  }

  num_gpu_transfers++;
  checkCudaErrors(cudaMemcpyPeerAsync(tmp, dev, blk.gpu[src_dev], src_dev,
                                      block_size * block_size * sizeof(float),
                                      stream.stream));

  return tmp;
}

void prefetch(stream_t &stream, int32_t dev, int32_t left, int32_t right) {

  {
    std::unique_lock<std::mutex> lck(tensors[left].m);

    // do we already have it
    auto &at = tensors[left];
    if (!at.location[dev]) {
      auto a_blk = fetch_from_gpu(stream, dev, at);
      at.gpu[dev] = a_blk;
    }
  }

  {
    std::unique_lock<std::mutex> lck(tensors[right].m);

    // do we already have it
    auto &bt = tensors[right];
    if (!bt.location[dev]) {
      auto b_blk = fetch_from_gpu(stream, dev, bt);
      bt.gpu[dev] = b_blk;
    }
  }
  
  // mark this as a checkpoint
  cudaEventRecord(stream.event, stream.stream);
}

void host_to_dev_thread() {

  cudaStream_t streams[num_devices];
  cudaEvent_t events[num_devices];
  for (auto idx = 0; idx < num_devices; ++idx) {
    checkCudaErrors(cudaSetDevice(idx));
    checkCudaErrors(cudaEventCreate(&events[idx]));
    checkCudaErrors(cudaStreamCreate(&streams[idx]));
  }

  int32_t cur_gpu_dev = 0;
  int32_t cur_dev = 0;
  std::vector<int32_t> queued_to_run;
  while (!tensor_pq.empty()) {

    // get a new tensor to transfer
    auto it = tensor_pq.begin();

    // pick a GPU and transfer it (async)
    move_to_gpu(cur_dev, it->second, events[cur_dev], streams[cur_dev]);

    // all the join groups
    auto jt = tid_mapping.find(it->second);
    for (int32_t idx = 0; idx < jt->second.size(); ++idx) {

      // the join group
      auto jg = jt->second[idx];

      // decrement the number of requred
      join_groups[jg].num_remaining--;
      if (join_groups[jg].num_remaining == 0) {
        queued_to_run.push_back(jg);
      } else if (join_groups[jg].num_remaining == 1) {

        auto other_tid = join_groups[jg].left == it->second
                             ? join_groups[jg].right
                             : join_groups[jg].left;
        auto kt = tensor_pq_mapping.find(other_tid)->second;

        auto stored_tid = kt->second;
        auto stored_key = kt->first;
        tensor_pq.erase(kt);

        stored_key.num_to_trigger++;
        auto a = tensor_pq.insert({stored_key, stored_tid});
        tensor_pq_mapping[other_tid] = a;
      }
    }

    // sync all the events here
    checkCudaErrors(cudaEventSynchronize(events[cur_dev]));

    {
      // set the gpu location
      std::unique_lock<std::mutex> lck(tensors[it->second].m);
      tensors[it->second].location[cur_dev] = true;
    }

    // remove it
    tensor_pq.erase(it);
    cur_dev = (cur_dev + 1) % num_devices;

    // add the queued join groups so they can be run
    std::unique_lock<std::mutex> lck(m);
    for (auto &jg : queued_to_run) {
      auto agg_group = agg_groups[{join_groups[jg].left, join_groups[jg].right}];
      if(agg_group_mappings[agg_group] == -1) {
        agg_group_mappings[agg_group] = cur_gpu_dev;
        cur_gpu_dev = (cur_gpu_dev + 1) % num_devices;
      }
      // std::cout << "Queued : " << jg << '\n';
      to_prep[agg_group_mappings[agg_group]].enqueue(jg);
    }
    queued_to_run.clear();
  }

  int32_t tmp = -1;
  for(auto dev = 0; dev < num_devices; ++dev) {
    to_prep[dev].enqueue(tmp);
  }
}

std::vector<bbts::concurent_queue<int32_t>> to_run;

std::atomic_int32_t cpy_join_grp_cnt;
void gpu_copy_thread(int32_t dev) {

  // set the device
  cudaSetDevice(dev);
  for (auto peer = 0; peer < num_devices; ++peer) {
    if (peer != dev) {
      cudaDeviceEnablePeerAccess(peer, 0);
    }
  }

  stream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream.stream));
  checkCudaErrors(cudaEventCreate(&stream.event));
  checkCudaErrors(cudaEventSynchronize(stream.event));

  while (true) {

    // wait before we get something
    int join_group;
    int left, right;

    // take the join group
    to_prep[dev].wait_dequeue(join_group);
    // std::cout << "Pulled : " << join_group << '\n';

    // we are done
    if(join_group == -1) {
      to_run[dev].enqueue(join_group);
      break;
    }

    // grab the left and right tuple index
    left = join_groups[join_group].left;
    right = join_groups[join_group].right;

    // prefetch
    cpy_join_grp_cnt++;
    prefetch(stream, dev, left, right);

    // sync all the events here
    checkCudaErrors(cudaEventSynchronize(stream.event));

    // mark that we have them on the gpu
    {
      std::unique_lock<std::mutex> lck(tensors[left].m);
      tensors[left].location[dev] = true;
    }

    {
      std::unique_lock<std::mutex> lck(tensors[right].m);
      tensors[right].location[dev] = true;
    }
      
    // std::cout << "Pushed 2 : " << join_group << '\n';
    to_run[dev].enqueue(join_group);
  }
}

std::vector<std::map<int, float *>> final_aggregated;

std::atomic_int32_t mult_cnt;
void product_thread(int32_t dev) {

  // set the device
  cudaSetDevice(dev);

  // create the stream
  stream_t stream;
  cudaStreamCreate(&stream.stream);
  cudaEventCreate(&stream.event);  
  
  // create the cublas handle
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  cublasSetStream(cublas_handle, stream.stream);

  // for the multiply
  float alpha = 1.0f;
  float beta = 0.0f;

  float *c_blk = nullptr;
  while(true) {

    float *a_blk, *b_blk;
    int join_group;
    int left, right;

    // take the join group
    to_run[dev].wait_dequeue(join_group);

    // we are done
    if(join_group == -1) {
      break;
    }

    {
      std::unique_lock<std::mutex> lck(m);

      // set the left and right tuple
      left = join_groups[join_group].left;
      right = join_groups[join_group].right;
    }

    {
      // get the blocks we need to multiply
      std::unique_lock<std::mutex> lck(tensors[left].m);
      a_blk = tensors[left].gpu[dev];
    }

    {
      std::unique_lock<std::mutex> lck(tensors[right].m);
      b_blk = tensors[right].gpu[dev];
    }

    // make sure everything is synced
    checkCudaErrors(cudaEventSynchronize(stream.event));

    // run the multiply
    mult_cnt++;
    if(c_blk == nullptr) {
      checkCudaErrors(cudaMallocManaged(&c_blk, block_size * block_size * sizeof(float)));
    }
    checkCudaErrors(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                block_size, block_size, block_size, &alpha,
                                a_blk, block_size, b_blk, block_size, &beta,
                                c_blk, block_size));
    float *c_out;
    int32_t agg_group;
    {
      std::unique_lock<std::mutex> lck(m);
      agg_group = agg_groups[{left, right}];
      c_out = final_aggregated[dev][agg_group];

      if(c_out == nullptr) {
        cudaEventRecord(stream.event, stream.stream);
        final_aggregated[dev][agg_group] = c_blk;
        c_blk = nullptr;
        continue;
      }
    }
                              
    // number of thread blocks in grid
    uint32_t threads_num = 1024;
    uint32_t n = block_size * block_size;
    uint32_t grid_size = (int)ceil((float)n / threads_num);

    // sum the stuff
    assert(c_out != nullptr);
    assert(c_blk != nullptr);
    add_kernel<<<grid_size, threads_num, 0, stream.stream>>>(c_out, c_blk, n);
    cudaEventRecord(stream.event, stream.stream);
  }

  checkCudaErrors(cudaDeviceSynchronize());
}

void set_to_one(float *blk) {
  for (size_t idx = 0; idx < block_size * block_size; ++idx) {
    blk[idx] = 1.0f;
  }
}

int main() {

  std::map<std::tuple<int32_t, int32_t>, int32_t> a;
  std::map<std::tuple<int32_t, int32_t>, int32_t> b;

  tensors = std::vector<block_info_t>(2 * block_split * block_split);
  int32_t i = 0;
  for (int idx = 0; idx < block_split; idx++) {
    for (int jdx = 0; jdx < block_split; jdx++) {

      float *a_blk;
      cudaMallocHost((void**)&a_blk, sizeof(float) * block_size * block_size);
      set_to_one(a_blk);
      block_info_t &a_nfo = tensors[i];
      a_nfo.cpu = a_blk;
      a_nfo.location.resize(num_devices);
      a_nfo.gpu.resize(num_devices);
      a[{idx, jdx}] = i++;

      float *b_blk;
      cudaMallocHost((void**)&b_blk, sizeof(float) * block_size * block_size);
      set_to_one(b_blk);
      block_info_t &b_nfo = tensors[i];
      b_nfo.cpu = b_blk;
      b_nfo.location.resize(num_devices);
      b_nfo.gpu.resize(num_devices);
      b[{idx, jdx}] = i++;
    }
  }

  for (int idx = 0; idx < block_split; idx++) {
    for (int jdx = 0; jdx < block_split; jdx++) {

      int32_t agg_idx = agg_group_mappings.size();
      agg_group_mappings.push_back(-1);
      for (int kdx = 0; kdx < block_split; kdx++) {
        auto left = a[{idx, kdx}];
        auto right = b[{kdx, jdx}];
        tid_mapping[left].push_back(join_groups.size());
        tid_mapping[right].push_back(join_groups.size());
        join_groups.push_back(
            {.num_remaining = 2, .left = left, .right = right});
        agg_groups[{left, right}] = agg_idx;
      }
    }
  }

  for (auto m : tid_mapping) {

    std::pair<tensor_pq_entry_t, int32_t> t = {};

    // init this
    t.first.num_to_trigger = m.second.size() == 1 ? 1 : 0;
    t.first.num_use = m.second.size();

    // set the tid of the input
    t.second = m.first;
    auto a = tensor_pq.insert(t);
    tensor_pq_mapping[m.first] = a;
  }

  final_aggregated.resize(num_devices);
  to_prep = std::vector<bbts::concurent_queue<int32_t>>(num_devices);

  to_run = std::vector<bbts::concurent_queue<int32_t>>(num_devices);

  // just to warm-up the cuda stuff
  float *ts;
  checkCudaErrors(cudaMallocManaged(&ts, 1024));
  cudaFree(ts);

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  auto h2dt = std::thread([]() { host_to_dev_thread(); });

  std::vector<std::thread> threads;
  for(auto dev = 0; dev < num_devices; ++dev) {

    threads.push_back(std::thread([dev]() { 
      gpu_copy_thread(dev); 
    }));

    threads.push_back(std::thread([dev]() { 
      product_thread(dev); 
    }));
  }

  h2dt.join();
  for(auto &t : threads) {
    t.join();
  }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();


  for(auto i : final_aggregated) {
    for(auto j : i) {
      // std::cout << j.second[0] << '\n';
    }
  }

  std::cout << "Multiplied : " << mult_cnt << '\n';
  std::cout << "Join Group : " << cpy_join_grp_cnt << '\n';
  std::cout << "Time run = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() * 0.000001 << "[s]" << std::endl;

  return 0;
}