#pragma once

#include <cstddef>
#include <cstdint>
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

    // the command id
    command_id_t cmd_id;

    // the id of the thread
    int32_t thread_id;

    // the start entry 
    long ts;

    // transform event to string
    std::string event_to_string() const {

      switch (event) {

        case event_t::START : return "START";
        case event_t::END : return "END";
        case event_t::STORAGE_OP_START : return "STORAGE_OP_START";
        case event_t::STORAGE_OP_END : return "STORAGE_OP_END";
        case event_t::KERNEL_START : return "KERNEL_START";
        case event_t::KERNEL_END : return "KERNEL_END";
        case event_t::SEND : return "SEND";
        case event_t::SEND_END : return "SEND_END";
        case event_t::RECV : return "RECV";
        case event_t::RECV_END : return "RECV_END";
      }
    }

  };

  struct batch_nfo_t {

    // the id
    size_t id;

    // the start timestamp
    long start;

    // the end timestap
    long end;

  };

  struct batch_t {

    // the batch info
    batch_nfo_t nfo;

    // the commans in the batch
    std::vector<log_entry_t> commands;
  };

  command_profiler_t(const node_config_ptr_t &config) : _config(config) {}

  // return all the profiling info
  std::vector<bbts::command_profiler_t::batch_nfo_t> get_all_profiling_nfo() {

    // lock the command in the batch
    std::unique_lock<std::mutex> lck{m};

    // fill it up
    std::vector<bbts::command_profiler_t::batch_nfo_t> out; out.reserve(previous_batches.size());
    for(auto &batch : previous_batches) {
      out.push_back(batch.nfo);
    }

    // return it
    return std::move(out);
  }

  bbts::command_profiler_t::batch_t get_profile_batch(size_t id) {

    // lock the command in the batch
    std::unique_lock<std::mutex> lck{m};

    // check if we even have it
    if(id >= previous_batches.size()) {
      return {};
    }

    // return a copy of the batch
    return previous_batches[id];
  }

  void command_event(command_id_t cmd_id, command_profiler_t::event_t event, int32_t thread_id) {

    // lock the command in the batch
    std::unique_lock<std::mutex> lck{m};
    
    // log this
    batch.commands.push_back(log_entry_t{.event = event, 
                                         .cmd_id = cmd_id, 
                                         .thread_id = thread_id, 
                                         .ts = std::chrono::high_resolution_clock::now().time_since_epoch().count()});
  }

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
    batch = batch_t{.nfo = {.id = previous_batches.size(), .start = ts, .end = -1}, .commands = {}};
  }

  // the batch ended
  void batch_ended() {

    // lock the command in the batch
    std::unique_lock<std::mutex> lck{m};

    // the batch has ended
    batch.nfo.end = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    // store the previous batch
    previous_batches.push_back(std::move(batch));
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