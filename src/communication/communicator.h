#pragma once

#include <mpich/mpi.h>
#include <cstdint>
#include <unordered_map>
#include <iostream>

#include "../tensor/tensor_factory.h"
#include "../ud_functions/ud_function.h"
#include "../server/node.h"
#include "../tensor/tensor.h"

namespace bbts {

// the type that identifies a node
using node_id_t = int32_t;

// identifier for any node
node_id_t ANY_NODE = MPI_ANY_SOURCE;

// defines some of the most common message request
using com_tags = int32_t;

// the request to a node to send the sending node the tensors it wants
const com_tags TENSOR_MOVE = 1;
const com_tags BCAST_TENSOR = 2;
const com_tags ANY = MPI_ANY_TAG;
const com_tags SHUTDOWN = 3;

// the mpi communicator
class mpi_communicator_t {
public:

  mpi_communicator_t(const node_config_ptr_t &_cfg) {

    // initialize the mpi
    MPI_Init(NULL, NULL);

    // get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &_num_nodes);

    // get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
  }

  ~mpi_communicator_t() {

    // shutdown the communicator
    MPI_Finalize();
  }

  // this is the request to fetch a certain number of tensors
  struct request_t {

    // the type of the message
    com_tags message_tag;
    
    // the number of bytes
    int32_t num_bytes;

    // the status of the message
    MPI_Status status;

    // the message identifier
    MPI_Message message;

    // the success 
    bool success = true;
  };

  // does the send, method is blocking
  bool send_sync(const void *_bytes, size_t num_bytes, node_id_t _node, com_tags _tag) {

    // get the number of byte to send and send the request
    return MPI_Ssend(_bytes, num_bytes, MPI_CHAR, _node, _tag, MPI_COMM_WORLD) == MPI_SUCCESS;
  }

  // waits for a request to be available from a particular node
  auto expect_request_sync(node_id_t _node, com_tags _tag) {

    // wait for a request
    request_t _req;
    auto mpi_errno = MPI_Mprobe(_node, (int32_t) _tag, MPI_COMM_WORLD, &_req.message, &_req.status);

    // check for errors
    if (mpi_errno != MPI_SUCCESS) {        
        _req.success = false;
        return _req;
    }

    // get the size
    MPI_Get_count(&_req.status, MPI_CHAR, &_req.num_bytes);

    // set the message type
    _req.message_tag = (com_tags) _req.status.MPI_TAG;

    // return the request
    return _req;
  }

  // recieves the request that we got from expect_request_sync
  auto recieve_request_sync(void *_bytes, request_t &_req) {

    // recieve the stuff
    return MPI_Mrecv (_bytes, _req.num_bytes, MPI_CHAR, &_req.message, &_req.status) == MPI_SUCCESS;
  }

  // waits for all the nodes to hit this, should only be used for initialization
  void barrier() {

    // wait for every node
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // return the rank
  int32_t get_rank() const {
    return _rank;
  }

  // return the number of nodes
  int32_t get_num_nodes() const {
    return _num_nodes;
  }

 private:

  // the rank of my node
  int32_t _rank;

  // the number of nodes in the cluster
  int32_t _num_nodes;

  // we need this to get hte tensor size
  tensor_factory_ptr_t _tensor_factory;
};

// the default communicator is the mpi communicator
using communicator = mpi_communicator_t;

}  // namespace bbts