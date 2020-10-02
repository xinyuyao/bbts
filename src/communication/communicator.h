#pragma once

#include <cstdint>

namespace bbts {

class mpi_communicator {
public:

  mpi_communicator(int32_t rank, int32_t num_nodes) {

  }

  // does the send blocking
  void send() {}

  // does the recv blocking
  void recv() {}

  // does the broadcast blocking
  void broadcast() {}

  // does the reduce blocking
  void reduce() {}

};


// the default communicator is the mpi communicator
using communicator = mpi_communicator;

}