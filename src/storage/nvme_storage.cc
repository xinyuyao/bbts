#include "nvme_storage.h"

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>  

#include "../server/static_config.h"
#include "../utils/terminal_color.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <future>
#include <unistd.h>


namespace bbts {

nvme_storage_t::~nvme_storage_t() {}

void nvme_storage_t::request_thread() {

  while (true) {

    // lock this thing
    std::unique_lock<std::mutex> lck (_m);

    // wait until we have something
    _res_processing_cv.wait(lck, [&]{return !_scheduled_reservations.empty() || is_shutdown;});

    // break out from here if we are done
    if(is_shutdown) {
      break;
    }

    // get the reservation
    auto val = std::move(std::get<0>(_scheduled_reservations.front()));
    auto res = std::get<1>(_scheduled_reservations.front());

    // pop it
    _scheduled_reservations.pop();

    // the result
    reservation_result_t out(res.get->size(), res.create->size());

    // we load the tensors here
    std::vector<sto_tensor_nfo_t*> to_load;
    to_load.reserve(res.get->size());

    // go through all the tensor we need to get
    size_t required = 0;
    for(auto &g : *res.get) {
      
      // find the tensor
      auto it = _tensor_nfo.find(g);
      
      // make sure the tensor is not deleted
      assert(it->second.state != tensor_state_t::DELETED);

      // if the tensor is not loaded we need prep, if it is unloading 
      // we will take care of it later but just not freeing the memory
      if(it->second.state == tensor_state_t::NOT_LOADED) {

          // make a new promise
          it->second.promise = std::promise<tensor_ref_t>();

          // turn in into a new shared future
          it->second.data = std::shared_future<tensor_ref_t>(it->second.promise.get_future());
          
          // mark that we are the one loading this tensor
          it->second.state = tensor_state_t::LOADING;

          // mark that we are loading this
          to_load.push_back(&it->second);

          // increment the number of bytes reqired
          required += it->second.num_bytes; 
      }

      // store the future
      // this is valid for UNLOADING, LOADING, LOADED
      out.get.push_back(it->second.data);
    }
    
    // go through all the tensors we need to create
    for(auto &c : *res.create) {
      
      // the the info about the tensor we need to create
      auto [tid, is_gpu, num_bytes] = c;

      // make an entry
      auto &ts = _tensor_nfo[tid];

      // set the values
      ts.id = tid;
      ts.num_bytes = num_bytes;
      ts.is_gpu = is_gpu;
      ts.num_ref = 1;
      ts.file_offset = -1;
      ts.state = tensor_state_t::LOADED;
      ts.promise = std::promise<tensor_ref_t>();
      ts.data = std::shared_future<tensor_ref_t>(ts.promise.get_future());

      // we need to allocate this
      required += num_bytes;

      // add the create
      out.create.push_back(ts.data);
    }

    // evict some tensors if necessary
    if(cur_allocated + required >= max_allocated) {
      _evict_some(lck, required);
    }

    // allocate the necessary tensors
    for(auto &c : *res.create) {
      
      auto [tid, is_gpu, num_bytes] = c;

      // get an entry
      auto &ts = _tensor_nfo[tid];

      // set the value
      ts.promise.set_value(tensor_ref_t{.id = tid, 
                                        .tensor = _allocate_tensor(num_bytes, is_gpu)});

      // we just allocated
      cur_allocated += num_bytes;
    }

    // load all the tensors
    for(auto &l : to_load) {
      
      // allocate the tensor
      l->promise.set_value(tensor_ref_t{.id = l->id, 
                                        .tensor = _allocate_tensor(l->num_bytes, l->is_gpu)});

      // we just allocated
      cur_allocated += l->num_bytes;
    }

    for(auto &l : to_load) {

      // read the tensor
      lck.unlock();
      pread(fp, l->data.get().tensor, l->num_bytes, l->file_offset);
      l->state = tensor_state_t::LOADED;
      lck.lock();
    }


    // set the value
    val.set_value(out);
  }

}

bool nvme_storage_t::remove_by_tid(tid_t _id) {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // try to find it
  auto it =_tensor_nfo.find(_id);
  if(it == _tensor_nfo.end()) {
    return false;
  }

  // make sure that nobody is referencing this tensor
  assert(it->second.num_ref == 0);

  // check for special cases
  if(it->second.state == tensor_state_t::UNLOADING) {
    
    // if the tensor is unloading now just mark it as deleted and return
    it->second.state = tensor_state_t::DELETED;
    return true;
  } 
  else if(it->second.state == tensor_state_t::NOT_LOADED) {
    
    // if it is not laded, simpy removce it
    _tensor_nfo.erase(it);
    return true;
  }

  // remove it from the lru
  _lru.remove(_id);

  // it is no longer allocated 
  cur_allocated -= it->second.num_bytes;

  // free the memory allocated
  free(it->second.data.get().tensor);

  // remove it
  _tensor_nfo.erase(it);

  return true;
}

bool nvme_storage_t::assign_tid(tid_t _anon_id, tid_t _id) {

  // TODO

  return true;
}

size_t nvme_storage_t::get_num_tensors() {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // return the number of tensors in the system
  return _tensor_nfo.size();
}

size_t nvme_storage_t::get_tensor_size(tid_t id) {

  // lock this thing
  std::unique_lock<std::mutex> lck (_m);

  // try to find it
  auto it = _tensor_nfo.find(id);
  if(it == _tensor_nfo.end()) {
    return 0;
  }

  // return the tensor size
  return it->second.num_bytes;
}

void nvme_storage_t::print() {}

void nvme_storage_t::clear() {}

void nvme_storage_t::shutdown() {

  // shutdown the storage
  is_shutdown = true;
  _res_processing_cv.notify_all();
}

bool nvme_storage_t::_try_reserve(const std::vector<tid_t> &get,
                                  std::vector<std::tuple<tid_t, bool, size_t>> &create) {

  // go through all the tensor we need to get
  size_t required = 0;
  for(auto &t : get) {

    // check if anybody has already reserved this tensor
    // if so there is no need to budget memory for it as it was already done
    auto it = _tensor_nfo.find(t);
    if(it->second.num_ref == 0) {
      required += it->second.num_bytes;
    } 

    // mark tha we are using this one
    it->second.num_ref++;
  }

  // go through all the tensors we want to create
  for(auto &c : create) {

    // add it to the required
    required += std::get<2>(c);
  }

  // check if we have enough memory
  if(cur_reserved + required >= max_allocated) {

    // we need to reverse reference counts
    for(auto &t : get) {
      auto it = _tensor_nfo.find(t);
      it->second.num_ref--;
    }

    // we failed
    return false;
  }

  // assign a tid to the anonymous tensor
  for(auto &c : create) {
    if(std::get<0>(c) == TID_NONE) {
      std::get<0>(c) = _current_anon--;
    }
  }

  // mark this as reserved
  cur_reserved += required;

  // tell that we have approved it
  return true;
}

std::future<nvme_storage_t::reservation_result_t> nvme_storage_t::_create_reserved(const std::vector<tid_t> &get,
                                                                                   const std::vector<std::tuple<tid_t, bool, size_t>> &create) {
  

  // get the future that is going to return the reservation result
  std::promise<reservation_result_t> res;
  auto f = res.get_future();

  // add the reservation to the queue
  _scheduled_reservations.push({std::move(res), reservation_nfo_t{.get = &get, .create = &create}});

  // make sure the processing threads know that there is stuff to do
  _res_processing_cv.notify_all();

  // return the result
  return std::move(f);
}

void nvme_storage_t::_release_reservation(const std::vector<tid_t> &get,
                                          const std::vector<std::tuple<tid_t, bool, size_t>> &create) {
  
  // go through all the tensor we wanted to get
  for(auto &t : get) {

    // find information about the tensor
    auto it = _tensor_nfo.find(t);
    it->second.num_ref--;

    // if this tensor is no longer reserved mark that we are 
    if(it->second.num_ref == 0) {

      // ok this is not reserved anymore
      cur_reserved -= it->second.num_bytes;

      // if we already laoded it add it to eviction
      if(it->second.state == tensor_state_t::LOADED) {
        
        // add it to the lru 
        _lru.add(t);
      }
    }
  }

  // go through all the tensors we wanted to create
  for(auto &c : create) {

    // find the created tensor
    auto it = _tensor_nfo.find(std::get<0>(c));
    it->second.num_ref--;

    // should be zero as otherwise it is wrong usage
    assert(it->second.num_ref == 0);

    // remove the reserved
    cur_reserved -= std::get<2>(c);

    // add it to the lru, as it should have zero references
    _lru.add(it->second.id);
  }
}

tensor_t *nvme_storage_t::_allocate_tensor(size_t num_bytes, bool used_by_gpu) {

  // malloc the tensor
  tensor_t *ts;
  if(used_by_gpu) {
    checkCudaErrors(cudaMallocManaged(&ts, num_bytes));
  }
  else {
    ts = (tensor_t*) malloc(num_bytes); 
  }

  return ts;
}

void nvme_storage_t::free_tensor(tensor_t *tensor, bool used_by_gpu) {
  if(used_by_gpu) {
    checkCudaErrors(cudaFree(tensor));
  }
  else {
    free(tensor);
  }
}

void nvme_storage_t::_evict_some(std::unique_lock<std::mutex> &lck, size_t required) {
  
  while (cur_allocated + required >= max_allocated) {

    // evict a tensor
    auto id = _lru.evict(lck);

    // check if it is deleted
    auto it = _tensor_nfo.find(id);

    // just some checking to make sure we are cool to evict this
    assert(it->second.num_ref == 0);
    assert(it->second.state == tensor_state_t::LOADED);

    // change the state to unloading
    it->second.state = tensor_state_t::UNLOADING;

    // if it does not have an offset assigned give it one
    if(it->second.file_offset == -1) {
      it->second.file_offset = file_offset;
      file_offset += it->second.num_bytes;
    }
    
    // unlock so we can dump the data to the disk
    lck.unlock();

    pwrite(fp, it->second.data.get().tensor, it->second.num_bytes, it->second.file_offset);

    // lock again so we can update the state
    lck.lock();

    // check if somebody has used it in the mean time
    if(it->second.num_ref != 0) {

      // ok we can not release this 
      it->second.state = tensor_state_t::LOADED;
    }
    if(it->second.state == tensor_state_t::DELETED) {
      
      // free the memory
      free_tensor(it->second.data.get().tensor, it->second.is_gpu);
      cur_allocated -= it->second.num_bytes;

      // if the tensor was delted in the mean time just kill it and add the memory
      _tensor_nfo.erase(it);
    }
    else {

      // ok the tensor is still there and it is not used
      cur_allocated -= it->second.num_bytes;
      it->second.state = tensor_state_t::NOT_LOADED;

      // free the memory
      free_tensor(it->second.data.get().tensor, it->second.is_gpu);
    }
  }
}

}

