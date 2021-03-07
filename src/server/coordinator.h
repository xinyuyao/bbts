#pragma once

#include <cstddef>
#include <memory>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "../server/coordinator_ops.h"
#include "../server/logger.h"
#include "../commands/command.h"
#include "../commands/tensor_stats.h"
#include "../communication/communicator.h"
#include "../commands/reservation_station.h"
#include "../commands/command_runner.h"
#include "../commands/tensor_notifier.h"
#include "coordinator.h"
#include "static_config.h"

namespace bbts {

class coordinator_t {
public:

  // init the scheduler
  coordinator_t(bbts::communicator_ptr_t _comm,
                bbts::reservation_station_ptr_t _rs,
                bbts::logger_ptr_t _logger,
                storage_ptr_t _storage,
                bbts::command_runner_ptr_t _command_runner,
                bbts::tensor_notifier_ptr_t _tensor_notifier,
                bbts::tensor_factory_ptr_t _tf,
                tensor_stats_ptr_t _stats);

  // accept a request
  void accept();

  // schedules all the provided commands
  std::tuple<bool, std::string> schedule_commands(const std::vector<command_ptr_t>& cmds);

  // run the commands
  std::tuple<bool, std::string> run_commands();

  // set the verbose status
  std::tuple<bool, std::string> set_verbose(bool val);

  // sets the number of threads to use
  std::tuple<bool, std::string> set_num_threads(std::uint32_t set_num_threads);

  // set the maximum storage
  std::tuple<bool, std::string> set_max_storage(size_t set_num_threads);

  // print the info abo
  std::tuple<bool, std::string> print_storage_info();

  // print info about a tensor
  std::tuple<bool, std::string> print_tensor_info(bbts::tid_t id);

  // clears the storage
  std::tuple<bool, std::string> clear();

  // shutdown the cluster
  std::tuple<bool, std::string> shutdown_cluster();

private:

  void _fail();

  void _schedule(coordinator_op_t ops, std::stringstream &ss);

  void _collect(std::tuple<bool, std::string> &out);

  void _load_cmds(const std::vector<command_ptr_t> &cmds,
                  std::stringstream &ss);

  void _run();

  void _clear();

  template<class T = storage_t>
  void _shutdown() {

    // sync
    _comm->barrier();

    // shutdown the command runner
    _command_runner->shutdown();

    // shutdown the reservation station
    _rs->shutdown();

    // shutdown the storage
    if constexpr(static_config::enable_storage) {
      std::static_pointer_cast<T>(_storage)->shutdown();
    }

    // shutdown the tensor notifier
    _tensor_notifier->shutdown();

    // mark that the coordinator is down
    _is_down = true;
  }

  void _set_verbose(bool val);

  template<class T = storage_t>
  void _set_max_storage(size_t val) {
    if constexpr(static_config::enable_storage) {
      std::static_pointer_cast<T>(_storage)->set_max_storage(val);
    }
  } 

  void _print_storage(std::stringstream &ss);

  void _print_tensor(tid_t id, std::stringstream &ss);

  // the communicator
  bbts::communicator_ptr_t _comm;

  // this stores all our tensors
  storage_ptr_t _storage;

  // the reservation station
  bbts::reservation_station_ptr_t _rs;

  // the statistics about the tensors for the current 
  bbts::tensor_stats_ptr_t _stats;

  // the logger
  bbts::logger_ptr_t _logger;

  // shutdown
  std::atomic<bool> _is_down{};

  // runs commands
  bbts::command_runner_ptr_t _command_runner;

  // the notifier
  bbts::tensor_notifier_ptr_t _tensor_notifier;

  // tensor factory
  bbts::tensor_factory_ptr_t _tf;
};

// the pointer
using coordinator_ptr_t = std::shared_ptr<coordinator_t>;

}