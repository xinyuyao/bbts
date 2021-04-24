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

    // the event type
    event_t event;

    // the start entry 
    std::size_t ts;

    // the id of the thread
    int32_t thread_id;
  };

  struct command_log_t {

    // the events that happened related to this command
    std::vector<log_entry_t> events;

    // the mutex 
    std::mutex m;
  };


  struct batch_t {

    // the commans in the batch
    std::vector<command_log_t> commands;
  };

  command_profiler_t(const node_config_ptr_t &config) : _config(config) {}

  // is the profilier enabled or not
  void set_enabled(bool val) { _config->profile = val; }

  // a batch of commans has started
  void batch_started(size_t num_cmds) {
    batch = batch_t{.commands = std::vector<command_log_t>(num_cmds)};
  }

  // the batch ended
  void batch_ended() {
    command_batches.push_back(std::move(batch));
  }

  //
  void command_event(command_id_t cmd_id, event_t event, int32_t thread_id) {
    
    // lock the command in the batch
    std::unique_lock<std::mutex> lck{batch.commands[cmd_id].m};
  }

  // 
  batch_t batch;

  // 
  std::vector<batch_t> command_batches;

  // the config of the node
  node_config_ptr_t _config;
};

using command_profiler_ptr_t = std::shared_ptr<command_profiler_t>;

}