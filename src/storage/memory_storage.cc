#include "memory_storage.h"
#include <cassert>
#include <cstdlib>
#include <stdexcept>

#ifdef ENABLE_GPU
#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>  
#endif
#include "../server/static_config.h"
#include "../utils/terminal_color.h"
#include <iostream>
#include <sstream>
#include <sys/mman.h>
#include <jemalloc/jemalloc.h>


namespace bbts {

memory_storage_t::memory_storage_t(communicator_ptr_t com)
    : _com(std::move(com)) {

  // just empty hooks
  _tensor_create_hook = [](tid_t _) {};
  _tensor_delete_hook = [](tid_t _) {};

  // 10GB 
  ____bytes = (char*) aligned_alloc(sizeof(float), 10lu * 1024lu * 1024lu * 1024lu);
  if(mlock(____bytes, 5lu * 1024lu * 1024lu * 1024lu) == -1) {
      throw std::runtime_error("Failed to lock the memory.");
  }
// bootstrap cuda
#ifdef ENABLE_GPU
  // bootstrap managed memory
  void *ts;
  checkCudaErrors(cudaMallocManaged(&ts, 1024));
  cudaFree(ts);
#endif
}

memory_storage_t::~memory_storage_t() {

  // go through each allocated tensor and free it
  for(auto &it : _tensor_nfo) {
    free_tensor(it.second.address, it.second.num_bytes);
  }

  std::cout << "Total bytes required " << ____num_bytes << '\n';

}

memory_storage_t::tensor_ref_t memory_storage_t::_get_by_tid(tid_t _id) { 

  // try to find the tensor if we find it return the address 
  auto it = _tensor_nfo.find(_id);
  return it != _tensor_nfo.end() ? tensor_ref_t{ .id = _id, .tensor = it->second.address } : 
                                   tensor_ref_t{ .id = _id, .tensor = nullptr };
}

memory_storage_t::tensor_ref_t memory_storage_t::_create_tensor(tid_t _id, size_t num_bytes) {

  // malloc the tensor
  tensor_t *ts = _allocate_tensor(num_bytes);

  // store the info
  _tensor_nfo[_id] = sto_tensor_nfo_t{.address = ts, .num_bytes = num_bytes};

  // notify that the tensor is created
  if constexpr (static_config::enable_hooks) {
    _tensor_create_hook(_id);
  }

  // return the tensor
  return {.id = _id, .tensor = ts};
}

memory_storage_t::tensor_ref_t memory_storage_t::_create_tensor(size_t num_bytes) {

  // malloc the tensor
  tensor_t *ts = _allocate_tensor(num_bytes);

  // get a new tid for this
  auto tid = _current_anon--;
  _tensor_nfo[tid] = { .address=ts, .num_bytes=num_bytes};

  // call the hook if necessary
  if constexpr (static_config::enable_hooks) {

    // notify that the tensor is created
    _tensor_create_hook(TID_NONE);
  }

  // return the tensor
  return {.id = tid, .tensor = ts};
}

tensor_t *memory_storage_t::_allocate_tensor(size_t num_bytes) {

  // malloc the tensor
  tensor_t *ts;

  #ifdef ENABLE_GPU
    // allocate the GPU
    checkCudaErrors(cudaMallocManaged(&ts, num_bytes));
  #else
    // we can not do this
    // ts = (tensor_t*) malloc(num_bytes);
    std::cout <<  this->____num_bytes << '\n';
    assert(this->____num_bytes < 20lu * 1024lu * 1024lu * 1024lu);
    ts = (tensor_t*) (____bytes + this->____num_bytes);
    this->____num_bytes += num_bytes;
  #endif

  return ts;
}

void memory_storage_t::free_tensor(tensor_t *tensor, size_t num_bytes) {

  #ifdef ENABLE_GPU
    // free the GPU
    checkCudaErrors(cudaFree(tensor));
  #else
    // free the regular tensor
    // free(tensor);
  #endif
}

bool memory_storage_t::has_tensor(tid_t _id) {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // check if found
  return _tensor_nfo.find(_id) != _tensor_nfo.end();
}

bool memory_storage_t::remove_by_tid(tid_t _id) {
  
  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // remove the tensor
  auto it = _tensor_nfo.find(_id);
  if(it == _tensor_nfo.end()) {
    return false;
  }

  // free the tensor
  free_tensor(it->second.address, it->second.num_bytes);

  // remove the tensor
  _tensor_nfo.erase(it);

  // call the hook if necessary
  if constexpr (static_config::enable_hooks) {

    // call that the tensor is deleted
    _tensor_delete_hook(_id);
  }

  return true;
}

bool memory_storage_t::assign_tid(tid_t _anon_id, tid_t _id) {
  
  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // try to find it
  auto it = _tensor_nfo.find(_anon_id);
  if(it == _tensor_nfo.end()) {
    return false;
  }

  // rewire
  _tensor_nfo[_id] = it->second;
  _tensor_nfo.erase(it);
  
  return true;
}

size_t memory_storage_t::get_num_tensors() {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  return _tensor_nfo.size();
}

size_t memory_storage_t::get_tensor_size(tid_t _id){

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // return the size if the tensor exists
  auto it = _tensor_nfo.find(_id);
  return it == _tensor_nfo.end() ? 0 : it->second.num_bytes;
}

void memory_storage_t::print(std::stringstream &ss) {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // print all the allocated tensors
  ss << bbts::green << "TID\tSize (in bytes)\t\taddress\n" << bbts::reset;
  for(auto &t : _tensor_nfo) {
    ss << t.first << "\t" << t.second.num_bytes << "\t\t" << (void*) t.second.address << '\n';
  }
}

void memory_storage_t::clear() {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // go through each allocated tensor and free it
  for(auto &it : _tensor_nfo) {
    
    // is it gpu
    free_tensor(it.second.address, it.second.num_bytes);
  }
  _tensor_nfo.clear();
}

memory_storage_t::reservation_result_t memory_storage_t::_create_reserved(const std::vector<tid_t> &get,
                                                                         const std::vector<std::tuple<tid_t, size_t>> &create) {

  // get all the tensors
  std::vector<tensor_ref_t> out_get;
  out_get.reserve(get.size());
  for (auto t : get) {
    out_get.push_back(_get_by_tid(t));
  }

  // create all the tensors we need
  std::vector<tensor_ref_t> out_create;
  out_create.reserve(create.size());
  for (auto ct : create) {

    // create all the necessary tensors
    auto [id, num_bytes] = ct;
    if (id != TID_NONE) {
      out_create.push_back(_create_tensor(id, num_bytes));
    } else {
      out_create.push_back(_create_tensor(num_bytes));
    }
  }

  // return the result
  return {.get = out_get, .create = out_create};
}

std::vector<std::tuple<bbts::tid_t, bbts::tensor_meta_t>> memory_storage_t::extract_meta() {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);
  
  std::size_t idx = 0;
  std::vector<std::tuple<bbts::tid_t, bbts::tensor_meta_t>> out(_tensor_nfo.size());
  for(auto nfo : _tensor_nfo) {
    out[idx++] = {nfo.first, nfo.second.address->_meta};
  }

  return std::move(out);
}

} // namespace bbts
