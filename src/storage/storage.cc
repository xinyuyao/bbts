#include "storage.h"
#include <iostream>

namespace bbts {

storage_t::~storage_t() {

  // go through each allocated tensor and free it
  for(auto &it : _allocated_tensors) {
    free(it.first);
  }
}

tensor_t *storage_t::get_by_tid(tid_t _id) { 

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // try to find the tensor if we find it return the address 
  auto it = _tensor_nfo.find(_id);
  return it != _tensor_nfo.end() ? it->second.address : nullptr;
}

tensor_t *storage_t::create_tensor(tid_t _id, size_t num_bytes) {
  
  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // malloc the tensor
  auto ts = (tensor_t*) malloc(num_bytes);
  _tensor_nfo[_id] = sto_tensor_nfo_t{.address = ts, .num_bytes = num_bytes};
  _allocated_tensors[ts] = _id;

  return ts;
}

tensor_t *storage_t::create_tensor(size_t num_bytes) {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // malloc the tensor
  auto ts = (tensor_t*) malloc(num_bytes);
  _allocated_tensors[ts] = TID_NONE;

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
  free(it->first);

  // remove the it from the other mapping if necessary
  if(it->second != TID_NONE) {
    _tensor_nfo.erase(it->second);
  }

  // remove it from the allocated tensors
  _allocated_tensors.erase(it);

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
  free(it->second.address);

  // remove the tensor
  _allocated_tensors.erase(it->second.address);
  _tensor_nfo.erase(it);

  return true;
}

}