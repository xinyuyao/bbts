#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <vector>
#include "../server/node_config.h"
#include "command.h"

namespace bbts {

class command_profiler_t {
public:

  enum class event_t {
    START,
    END,
    STORAGE_OP_START,
    STORAGE_OP_END,
    KERNEL_START,
    KERNEL_END,
    SEND,
    SEND_END,
    RECV,
    RECV_END
  };

  struct log_entry_t {

    // the command id
    command_id_t cmd_id;

    // the event type
    event_t event;

    // the start entry 
    long ts;

    // the id of the thread
    int32_t thread_id;
  };

  struct batch_t {

    // the timestamp
    long ts;

    // the commans in the batch
    std::vector<log_entry_t> commands;
  };

  command_profiler_t(const node_config_ptr_t &config) : _config(config) {}

  // is the profilier enabled or not
  void set_enabled(bool val) { _config->profile = val; }

  // a batch of commans has started
  void batch_started() {

    // check if the profiling is even enabled
    if(!_config->profile) {
      return;
    }
 
    auto ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    // lock the command in the batch
    std::unique_lock<std::mutex> lck{m};

    // create a new batch
    batch = batch_t{.ts = ts, .commands = {}};
  }

  // the batch ended
  void batch_ended() {

    // lock the command in the batch
    std::unique_lock<std::mutex> lck{m};

    // store the previous batch
    previous_batches.push_back(std::move(batch));
  }

  // log the command
  void command_event(command_id_t cmd_id, event_t event, int32_t thread_id) {
    
    auto ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    // lock the command in the batch
    std::unique_lock<std::mutex> lck{m};

    // push the command
    batch.commands.push_back(log_entry_t{.cmd_id = cmd_id, .event = event, .ts = ts, .thread_id = thread_id});
  }

  std::mutex m;

  // the current batch
  batch_t batch;

  // the previous batches
  std::vector<batch_t> previous_batches;

  // the config of the node
  node_config_ptr_t _config;
};

using command_profiler_ptr_t = std::shared_ptr<command_profiler_t>;

}