#pragma once

#include "command.h"
#include <cassert>
#include <cstddef>
#include <unordered_map>

namespace bbts {

// this keeps track of what tensor is used by how many and what kind of commands
struct tensor_stats_t {
public:
  struct tensor_stat_t {

    // number of commands it is used in as an input
    size_t num_used_in;

    // should this tensor also be available on the GPU
    bool is_gpu;
  };

  // go through the command
  void add_command(const command_t &_cmd) {

    // is this command using a gpu
    bool is_gpu = (_cmd.is_apply() || _cmd.is_reduce()) && _cmd.nfo.is_gpu;
    assert(is_gpu == false);

    // go through all the inputs
    const auto &inputs = _cmd.get_inputs();
    for (const auto &in : inputs) {
      auto &stat = _stats[in.tid];
      stat.is_gpu = stat.is_gpu || is_gpu;
      stat.num_used_in++;
    }

    // go thorugh the outputs
    const auto &outputs = _cmd.get_outputs();
    for (const auto &out : outputs) {
      auto &stat = _stats[out.tid];
      stat.is_gpu = stat.is_gpu || is_gpu;
    }
  }

  // add the tensor
  void add_tensor(tid_t _tid, bool is_gpu) {
    auto &stat = _stats[_tid];
    stat.is_gpu = is_gpu;
  }

  // check if this tensor is used by a gpu
  bool is_gpu(tid_t id) const {
    auto [_, val] = *_stats.find(id);
    return val.is_gpu;
  }

  // reset the tensor stats
  void reset() {

    // simpy clear whatever was there
    _stats.clear();
  }

  // the stats for the tensors
  std::unordered_map<tid_t, tensor_stat_t> _stats;
};

// the shared pointer for this
using tensor_stats_ptr_t = std::shared_ptr<tensor_stats_t>;

} // namespace bbts