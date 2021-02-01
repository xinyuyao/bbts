#pragma once

#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <mutex>
#include "../tensor/tensor.h"

namespace bbts {

// this class is responsible for memory managment like getting new 
// memory for tensors, getting already existing tensors and in the future evicting
// tensors to disk or moving them to GPU
struct storage_t {

  storage_t() {

    // just empty hooks
    _tensor_create_hook = [](tid_t _) {};
    _tensor_delete_hook = [](tid_t _) {};
  }

  // destroy the storage and frees all the tensors
  ~storage_t();

  // information about the stored tensor
  struct sto_tensor_nfo_t {
    
    // if present the address will be not null if evicted
    tensor_t *address;

    // the size of the tensor in bytes
    size_t num_bytes;
  };

  // an existing tensor by tid
  tensor_t *get_by_tid(tid_t _id);

  // returns a tensor for the tid, the tensor is always not initialized
  tensor_t *create_tensor(tid_t _id, size_t num_bytes);

  // returns an anonymous tensor, the tensor is always not initialized
  tensor_t *create_tensor(size_t num_bytes);

  // remove by address true if it succeeds
  bool remove_by_tensor(tensor_t &_tensor);

  // assign a tid to an anonymous tensor
  bool assign_tid(tensor_t &_tensor, tid_t _tid);

  // remove by tid
  bool remove_by_tid(tid_t _id);

  // add the retired hook
  template<class fn>
  void add_created_hook(fn fun){ _tensor_create_hook = fun; }

  // add the retired hook
  template<class fn>
  void add_deleted_hook(fn fun){ _tensor_delete_hook = fun; }

  // print the storage
  void print();

  // get the number of tensors in the system
  size_t get_num_tensors();

  // the mutex to lock this thing as it is going to be hammered by threads
  std::mutex _m;

  // all allocated tensors
  std::unordered_map<tensor_t*, std::tuple<tid_t, size_t>> _allocated_tensors;

  // maps to the information
  std::unordered_map<tid_t, sto_tensor_nfo_t> _tensor_nfo;

  // called when a command is retired on this node
  std::function<void(tid_t id)> _tensor_create_hook;

  // called when a command is retired on this node
  std::function<void(tid_t id)> _tensor_delete_hook;
};

// we put the storage here
using storage_ptr_t = std::shared_ptr<storage_t>;

}