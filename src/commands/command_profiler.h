#pragma once

#include <memory>
#include "../server/node_config.h"

namespace bbts {

class command_profiler_t {
public:

  command_profiler_t(const node_config_ptr_t &config) : _config(config) {}

  void set_enabled(bool val) { _config->profile = val; }

  // the config of the node
  node_config_ptr_t _config;
};

using command_profiler_ptr_t = std::shared_ptr<command_profiler_t>;

}