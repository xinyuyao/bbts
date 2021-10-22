#pragma once

#include <cstdint>
#include <memory>

namespace bbts {

// the type that identifies a node
using node_id_t = int32_t;

// this structure information on how the node is configured
struct node_config_t {

  // the number of arguments passed to the node
  int argc;

  // the arguments as string pointers
  char **argv;

  // the number of threads we are running
  size_t num_threads = 8;

  // the total available ram memory
  size_t total_ram = 0;

  // should we print out everything?
  bool verbose = false;

  // the cost to transfter bytes to/from the GPU per byte
  float gpu_transfer_cost_per_byte = 1.0f;

  // the cost to send bytes per byte
  float send_cost_per_byte = 1.16415322E-9;
  
};

// a nice way to reference the configuration ptr
using node_config_ptr_t = std::shared_ptr<node_config_t>;

}