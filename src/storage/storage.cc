#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>  

#include "storage.h"
#include "../server/static_config.h"
#include "../utils/terminal_color.h"
#include <iostream>

namespace bbts {

storage_t::~storage_t() {

  // go through each allocated tensor and free it
  for(auto &it : _allocated_tensors) {
    if(it.second.is_gpu) {
      cudaFree(it.first);
    }
    else {
      free(it.first);
    }
  }
}

tensor_t *storage_t::get_by_tid(tid_t _id) { 

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // try to find the tensor if we find it return the address 
  auto it = _tensor_nfo.find(_id);
  return it != _tensor_nfo.end() ? it->second.address : nullptr;
}

tensor_t *storage_t::create_tensor(tid_t _id, size_t num_bytes, bool used_by_gpu) {
  
  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // malloc the tensor
  tensor_t *ts;
  if(used_by_gpu) {
    cudaMallocManaged(&ts, num_bytes);
  }
  else {
    ts = (tensor_t*) malloc(num_bytes); 
  }

  _tensor_nfo[_id] = sto_tensor_nfo_t{.address = ts, .num_bytes = num_bytes, .is_gpu = used_by_gpu};
  _allocated_tensors[ts] = { .id=_id, .num_bytes=num_bytes, .is_gpu= used_by_gpu };

  lck.unlock();

  // notify that the tensor is created
  _tensor_create_hook(_id);

  // return the tensor
  return ts;
}

tensor_t *storage_t::create_tensor(size_t num_bytes, bool used_by_gpu) {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // malloc the tensor
  tensor_t *ts;
  if(used_by_gpu) {
    cudaMallocManaged(&ts, num_bytes);
  }
  else {
    ts = (tensor_t*) malloc(num_bytes); 
  }

  _allocated_tensors[ts] = { .id=TID_NONE, .num_bytes=num_bytes, .is_gpu=used_by_gpu };

  lck.unlock();

  // call the hook if necessary
  if constexpr (static_config::enable_hooks) {

    // notify that the tensor is created
    _tensor_create_hook(TID_NONE);
  }

  // return the tensor
  return ts;
}

bool storage_t::remove_by_tensor(tensor_t &_tensor) {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // try to find the tensor
  auto it = _allocated_tensors.find(&_tensor);
  if(it == _allocated_tensors.end()) {
    return false;
  }

  // free the tensor
  if(it->second.is_gpu) {
    cudaFree(it->first);
  }
  else {
    free(it->first);
  }

  // remove the it from the other mapping if necessary
  auto _tid = it->second.id;
  if(it->second.id != TID_NONE) {
    _tensor_nfo.erase(it->second.id);
  }

  // remove it from the allocated tensors
  _allocated_tensors.erase(it);

  // we are done with bookkeeping
  lck.unlock();

  // call the hook if necessary
  if constexpr (static_config::enable_hooks) {

    // call that the tensor is deleted
    _tensor_delete_hook(_tid);
  }

  // we are out of here...
  return true;
}

bool storage_t::assign_tid(tensor_t &_tensor, tid_t _tid) {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // try to find the tensor
  auto it = _allocated_tensors.find(&_tensor);
  if(it == _allocated_tensors.end()) {
    return false;
  }

  // make sure that we don't already have a tensor with this tid
  if(_tensor_nfo.find(_tid) != _tensor_nfo.end()) {
    return false;
  }

  // set the new id
  it->second.id = _tid;
  _tensor_nfo[_tid] = sto_tensor_nfo_t{ .address = &_tensor, .num_bytes = it->second.num_bytes };

  return true;
}

bool storage_t::remove_by_tid(tid_t _id) {
  
  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // remove the tensor
  auto it = _tensor_nfo.find(_id);
  if(it == _tensor_nfo.end()) {
    return false;
  }

  // free the tensor
  auto jt = _allocated_tensors.find(it->second.address);
  if(jt->second.is_gpu) {
    cudaFree(jt->first);
  }
  else {
    free(jt->first);
  }

  // remove the tensor
  _allocated_tensors.erase(jt);
  _tensor_nfo.erase(it);

  // unlock
  lck.unlock();

  // call the hook if necessary
  if constexpr (static_config::enable_hooks) {

    // call that the tensor is deleted
    _tensor_delete_hook(_id);
  }

  return true;
}

size_t storage_t::get_num_tensors() {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  return _tensor_nfo.size();
}

void storage_t::print() {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // print all the allocated tensors
  std::cout << bbts::green << "TID\tSize (in bytes)\t\taddress\n" << bbts::reset;
  for(auto &t : _allocated_tensors) {
    std::cout << t.second.id << "\t" << t.second.is_gpu << "\t" << t.second.num_bytes << "\t\t" << (void*) t.first << '\n';
  }
}

void storage_t::clear() {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // go through each allocated tensor and free it
  for(auto &it : _allocated_tensors) {
    
    // is it gpu
    if(it.second.is_gpu) {
      cudaFree(it.first);
    }
    else {
      free(it.first);
    }
  }
  _allocated_tensors.clear();
  _tensor_nfo.clear();
}

}