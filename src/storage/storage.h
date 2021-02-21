#pragma once

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <memory>
#include <mutex>
#include "../tensor/tensor.h"
#include "../communication/communicator.h"

namespace bbts {


// this class is responsible for memory managment like getting new 
// memory for tensors, getting already existing tensors and in the future evicting
// tensors to disk or moving them to GPU
struct storage_t {

  storage_t(communicator_ptr_t com) : _com(std::move(com)) {

    // just empty hooks
    _tensor_create_hook = [](tid_t _) {};
    _tensor_delete_hook = [](tid_t _) {};
  }

  // destructor
  ~storage_t();

  // is a reference of the tensor
  struct tensor_ref_t { 

    // the identifier of the tensor
    tid_t id;

    // tensor 
    tensor_t *tensor;
  };

  // the result of a reservation
  struct reservation_result_t {

    // the existing tensors we want to get
    std::vector<tensor_ref_t> get;

    // the tensors we want to create
    std::vector<tensor_ref_t> create;
  };

  // make sure that all the tensors created or requested are aquired at the same time
  template<class fn>
  void local_transaction(const std::vector<tid_t> &get, 
                         const std::vector<std::tuple<tid_t, bool, size_t>> &create,
                         const fn &fun) {
  
    for(;;) {

      // lock this thing
      std::unique_lock<std::mutex> lck (_m);

      // try to aquire a reservation
      bool val = _try_reserve(get, create);

      if(!val) {

        // maybe I will need to adjust this if it is firing to much
        lck.unlock();
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));

        // try again
        continue;
      }

      // craete the reserved
      auto c = _create_reserved(get, create);

      // run the function
      lck.unlock();
      fun(c);
      lck.lock();
      
      // release the reserved
      _release_reservation(get, create);

      // we are done here
      break;
    }
  }

  // if there is a 
  template<class fn>
  void remote_transaction(command_id_t cmd,
                          const bbts::command_t::node_list_t &nodes,
                          const std::vector<tid_t> &get, 
                          const std::vector<std::tuple<tid_t, bool, size_t>> &create,
                          const fn &fun) {
    
    for(;;) {

      // lock this thing
      std::unique_lock<std::mutex> lck (_m);

      // try to aquire a reservation
      bool val = _try_reserve(get, create);

      // sync all the nodes so that they know if the reservation was aquired or not
      lck.unlock();
      bool res = _com->sync_resource_aquisition(cmd, nodes, val);
      lck.lock();

      // did all the nodes aquire a result if so we can proceed
      if(res) {

        // craete the reserved
        auto c = _create_reserved(get, create);

        // run the function
        lck.unlock();
        fun(c);
        lck.lock();

        // release the reserved
        _release_reservation(get, create);

        return;
      }

      // release the reservation and try to reaqire again
      if(val) {
        _release_reservation(get, create);
      }

      // maybe I will need to adjust this if it is firing to much
      lck.unlock();
      std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }
  }

    // if there is a 
  template<class fn>
  void remote_transaction_p2p(command_id_t cmd,
                              node_id_t other,
                              const std::vector<tid_t> &get, 
                              const std::vector<std::tuple<tid_t, bool, size_t>> &create,
                              const fn &fun) {
    
    for(;;) {

      // lock this thing
      std::unique_lock<std::mutex> lck (_m);

      // try to aquire a reservation
      bool val = _try_reserve(get, create);
 
      // sync all the nodes so that they know if the reservation was aquired or not
      lck.unlock();
      bool res = _com->sync_resource_aquisition_p2p(cmd, other, val);
      lck.lock();

      // did all the nodes aquire a result if so we can proceed
      if(res) {

        // craete the reserved
        auto c = _create_reserved(get, create);

        // run the function
        lck.unlock();
        fun(c);
        lck.lock();

        // release the reserved
        _release_reservation(get, create);

        return;
      }

      // release the reservation and try to reaqire again
      if(val) {
        _release_reservation(get, create);
      }

      // maybe I will need to adjust this if it is firing to much
      lck.unlock();
      std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }
  }

  // remove by tid
  bool remove_by_tid(tid_t _id);

  // assign a tid ot the anonymous tensor
  bool assign_tid(tid_t _anon_id, tid_t _id);

  // print the storage
  void print();

  // clear the storage
  void clear();

  // get the number of tensors in the system
  size_t get_num_tensors();
  
  // returns the size of a tensor
  size_t get_tensor_size(tid_t _id);

  // add the retired hook
  template<class fn>
  void add_created_hook(fn fun){ _tensor_create_hook = fun; }

  // add the retired hook
  template<class fn>
  void add_deleted_hook(fn fun){ _tensor_delete_hook = fun; }

private:

  // information about the stored tensor
  struct sto_tensor_nfo_t {
    
    // if present the address will be not null if evicted
    tensor_t *address;

    // the size of the tensor in bytes
    size_t num_bytes;

    // is this the gpu 
    bool is_gpu;
  };

  // returns a tensor for the tid, the tensor is always not initialized
  tensor_ref_t _create_tensor(tid_t _id, size_t num_bytes, bool used_by_gpu);

  // returns an anonymous tensor, the tensor is always not initialized
  tensor_ref_t _create_tensor(size_t num_bytes, bool used_by_gpu);

  // an existing tensor by tid
  tensor_ref_t _get_by_tid(tid_t _id);

  // this reserves space for the tensors in get to be loaded and tensors in create to created
  bool _try_reserve(const std::vector<tid_t> &get,
                   const std::vector<std::tuple<tid_t, bool, size_t>> &create);

  // craete all the tensors we just reserved
  reservation_result_t _create_reserved(const std::vector<tid_t> &get, 
                                       const std::vector<std::tuple<tid_t, bool, size_t>> &create);

  // release the reserved tensors
  void _release_reservation(const std::vector<tid_t> &get,
                            const std::vector<std::tuple<tid_t, bool, size_t>> &create);

  // the mutex to lock this thing as it is going to be hammered by threads
  std::mutex _m;

  // init it to the first available negative number
  tid_t _current_anon = TID_NONE - 1;

  // maps to the information
  std::unordered_map<tid_t, sto_tensor_nfo_t> _tensor_nfo;

  // called when a command is retired on this node
  std::function<void(tid_t id)> _tensor_create_hook;

  // called when a command is retired on this node
  std::function<void(tid_t id)> _tensor_delete_hook;

  // communicator used for remote transactions
  bbts::communicator_ptr_t _com;
};

// we put the storage here
using storage_ptr_t = std::shared_ptr<storage_t>;

}