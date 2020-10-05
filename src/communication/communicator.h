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

// defines some of the most common message request
enum class communicator_messages : int32_t {

  // the request to a node to send the sending node the tensors it wants
  TENSOR_REQUEST = 1,
  BCAST_TENSOR_REQUEST = 2,
  SHUTDOWN = 3
};

class mpi_communicator_t {
public:

  mpi_communicator_t(const node_config_ptr_t &_cfg) {
    // initialize the mpi
    MPI_Init(&_cfg->argc, &_cfg->argv);

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
    communicator_messages message_type;
    
    // the number of bytes
    int32_t num_bytes;

    // the status of the message
    MPI_Status status;

    // the message identifier
    MPI_Message message;
  };

  // does the send, method is blocking
  bool send_tensor(const tensor_t &_tensor, node_id_t _node) {

    // get the number of byte to send and send the request
    auto numBytes = _tensor_factory->get_tensor_size(_tensor._meta);
    return MPI_Ssend(&_tensor, numBytes, MPI_CHAR, _node, (int32_t) communicator_messages::BCAST_TENSOR_REQUEST, MPI_COMM_WORLD) == MPI_SUCCESS;
  }

  // does the broadcast blocking
  bool broadcast_tensor(tensor_t &_tensor, node_id_t _me) {
    auto numBytes = _tensor_factory->get_tensor_size(_tensor._meta);
    return MPI_Bcast(&_tensor, numBytes, MPI_CHAR, _me, MPI_COMM_WORLD) == MPI_SUCCESS;
  }

  // waits for a request to be available from a particular node
  auto listen_for_request(node_id_t _node) {

    // wait for a request
    request_t _req;
    MPI_Mprobe(_node, MPI_ANY_TAG, MPI_COMM_WORLD, &_req.message, &_req.status);

    // get the size
    MPI_Get_count(&_req.status, MPI_CHAR, &_req.num_bytes);

    // set the message type
    _req.message_type = (communicator_messages) _req.status.MPI_TAG;

    // return the request
    return _req;
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