#include "storage.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>  

#include "../server/static_config.h"
#include "../utils/terminal_color.h"
#include <driver_types.h>
#include <iostream>


namespace bbts {

storage_t::~storage_t() {

  // go through each allocated tensor and free it
  for(auto &it : _tensor_nfo) {
    if(it.second.is_gpu) {
      cudaFree(it.second.address);
    }
    else {
      free(it.second.address);
    }
  }
}

storage_t::tensor_ref_t storage_t::_get_by_tid(tid_t _id) { 

  // try to find the tensor if we find it return the address 
  auto it = _tensor_nfo.find(_id);
  return it != _tensor_nfo.end() ? tensor_ref_t{ .id = _id, .tensor = it->second.address } : 
                                   tensor_ref_t{ .id = _id, .tensor = nullptr };
}

storage_t::tensor_ref_t storage_t::_create_tensor(tid_t _id, size_t num_bytes, bool used_by_gpu) {

  // malloc the tensor
  tensor_t *ts;
  if(used_by_gpu) {
    checkCudaErrors(cudaMallocManaged(&ts, num_bytes));
  }
  else {
    ts = (tensor_t*) malloc(num_bytes); 
  }

  // store the info
  _tensor_nfo[_id] = sto_tensor_nfo_t{.address = ts, .num_bytes = num_bytes, .is_gpu = used_by_gpu};

  // notify that the tensor is created
  if constexpr (static_config::enable_hooks) {
    _tensor_create_hook(_id);
  }

  // return the tensor
  return {.id = _id, .tensor = ts};
}

storage_t::tensor_ref_t storage_t::_create_tensor(size_t num_bytes, bool used_by_gpu) {

  // malloc the tensor
  tensor_t *ts;
  if(used_by_gpu) {
    checkCudaErrors(cudaMallocManaged(&ts, num_bytes));
  }
  else {
    ts = (tensor_t*) malloc(num_bytes); 
  }

  // get a new tid for this
  auto tid = _current_anon--;
  _tensor_nfo[tid] = { .address=ts, .num_bytes=num_bytes, .is_gpu=used_by_gpu };

  // call the hook if necessary
  if constexpr (static_config::enable_hooks) {

    // notify that the tensor is created
    _tensor_create_hook(TID_NONE);
  }

  // return the tensor
  return {.id = tid, .tensor = ts};
}

bool storage_t::remove_by_tid(tid_t _id) {

  // remove the tensor
  auto it = _tensor_nfo.find(_id);
  if(it == _tensor_nfo.end()) {
    return false;
  }

  // free the tensor
  if(it->second.is_gpu) {
    cudaFree(it->second.address);
  }
  else {
    free(it->second.address);
  }

  // remove the tensor
  _tensor_nfo.erase(it);

  // call the hook if necessary
  if constexpr (static_config::enable_hooks) {

    // call that the tensor is deleted
    _tensor_delete_hook(_id);
  }

  return true;
}

bool storage_t::assign_tid(tid_t _anon_id, tid_t _id) {
  
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

size_t storage_t::get_num_tensors() {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  return _tensor_nfo.size();
}

size_t storage_t::get_tensor_size(tid_t _id){

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // return the size if the tensor exists
  auto it = _tensor_nfo.find(_id);
  return it == _tensor_nfo.end() ? 0 : it->second.num_bytes;
}

void storage_t::print() {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // print all the allocated tensors
  std::cout << bbts::green << "TID\tSize (in bytes)\t\taddress\n" << bbts::reset;
  for(auto &t : _tensor_nfo) {
    std::cout << t.first << "\t" << t.second.is_gpu << "\t" << t.second.num_bytes << "\t\t" << (void*) t.second.address << '\n';
  }
}

void storage_t::clear() {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // go through each allocated tensor and free it
  for(auto &it : _tensor_nfo) {
    
    // is it gpu
    if(it.second.is_gpu) {
      cudaFree(it.second.address);
    }
    else {
      free(it.second.address);
    }
  }
  _tensor_nfo.clear();
}

bool storage_t::_try_reserve(const std::vector<tid_t> &get,
                             const std::vector<std::tuple<tid_t, bool, size_t>> &create) {

  // TODO for now we always do this
  return true;
}

storage_t::reservation_result_t storage_t::_create_reserved(const std::vector<tid_t> &get,
                                                                   const std::vector<std::tuple<tid_t, bool, size_t>> &create) {

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
    auto [id, is_gpu, num_bytes] = ct;
    if (id != TID_NONE) {
      out_create.push_back(_create_tensor(id, num_bytes, is_gpu));
    } else {
      out_create.push_back(_create_tensor(num_bytes, is_gpu));
    }
  }

  // return the result
  return {.get = out_get, .create = out_create};
}

// release the reserved tensors
void storage_t::_release_reservation(const std::vector<tid_t> &get,
                                        const std::vector<std::tuple<tid_t, bool, size_t>> &create) {
  return;
}

}
