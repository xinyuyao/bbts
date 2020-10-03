#pragma once

#include <cstdint>
#include <iostream>
#include <mpi.h>
#include "../tensor/tensor.h"
#include "../server/node.h"

namespace bbts {

enum class b : int32_t {
  
  

};

class mpi_communicator {
public:

  mpi_communicator(const node_config_ptr_t &_cfg) {
    
    // initialize the mpi
    MPI_Init(&_cfg->argc, &_cfg->argv);

    // get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &_num_nodes);

    // get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &_rank);

    std::cout << "I am : " << _rank << " " << _num_nodes << '\n';

  }

  ~mpi_communicator() {

    // shutdown the communicator
    MPI_Finalize();
  }

  // does the send, method is blocking
  void send_tensor(const tensor_t _tensor) {

  }

  // does the recv blocking
  void recv_tensor() {

  }

  // does the broadcast blocking
  void broadcast_tensor() {

  }

  // does the reduce blocking
  void reduce_tensor() {

  }

  // 

private:

  // the rank of my node
  int32_t _rank;

  // the number of nodes in the cluster
  int32_t _num_nodes;

};


// the default communicator is the mpi communicator
using communicator = mpi_communicator;

}