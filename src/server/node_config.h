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
};

// a nice way to reference the configuration ptr
using node_config_ptr_t = std::shared_ptr<node_config_t>;

}