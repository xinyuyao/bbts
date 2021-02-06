#pragma once

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <unistd.h>
#include <dlfcn.h>
#include "../server/coordinator_ops.h"
#include "../server/logger.h"
#include "../commands/command.h"
#include "../communication/communicator.h"
#include "../commands/reservation_station.h"
#include "../commands/command_runner.h"
#include "../commands/tensor_notifier.h"
#include "../tensor/tensor_factory.h"
#include "../ud_functions/udf_manager.h"
#include "coordinator.h"

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
                bbts::tensor_factory_ptr_t _tensor_factory,
                bbts::udf_manager_ptr _udf_manager);

  // accept a request
  void accept();

  // schedules all the provided commands
  std::tuple<bool, std::string> schedule_commands(const std::vector<command_ptr_t>& cmds);

  // run the commands
  std::tuple<bool, std::string> run_commands();

  // set the verbose status
  std::tuple<bool, std::string> set_verbose(bool val);

  // print the info abo
  std::tuple<bool, std::string> print_storage_info();

  // clears the storage
  std::tuple<bool, std::string> clear();

  // shutdown the cluster
  std::tuple<bool, std::string> shutdown_cluster();

  // load the shared library contained in file_bytes
  std::tuple<bool, std::string> load_shared_library(char* file_bytes, size_t file_size);

  // shutdown the coordinator
  void shutdown();

private:

  void _fail();

  void _schedule(coordinator_op_t ops);

  void _load_cmds(const std::vector<command_ptr_t> &cmds);

  void _run();

  void _clear();

  void _shutdown();

  void _set_verbose(bool val);

  void _print_storage();

  void _register(coordinator_op_t ops);

  void _register_from_bytes(char* file_bytes, size_t file_size);

  // the communicator
  bbts::communicator_ptr_t _comm;

  // this stores all our tensors
  storage_ptr_t _storage;

  // the reservation station
  bbts::reservation_station_ptr_t _rs;

  // the logger
  bbts::logger_ptr_t _logger;

  // shutdown
  std::atomic<bool> _is_down{};

  // runs commands
  bbts::command_runner_ptr_t _command_runner;

  // the notifier
  bbts::tensor_notifier_ptr_t _tensor_notifier;

  // the tensor factory
  bbts::tensor_factory_ptr_t _tensor_factory;

  // the udf manager
  bbts::udf_manager_ptr _udf_manager;

  // This struct will handle removing temporary files opened by 
  // load_shared_library and hold onto the so_handle. 
  // It isn't strictly necessary because the os will delete files 
  // in /tmp/ periodically. 
  // When barb exits, the shared library will be closed regardless 
  // of calling dlclose.
  struct shared_library_item_t {
    shared_library_item_t(std::string const& filename, void* so_handle)
      : filename(filename), so_handle(so_handle) {}
    ~shared_library_item_t(){
      // we no longer need the temporary file
      unlink(filename.c_str());
    }

    std::string filename;
    void* so_handle;
  };
  std::vector<shared_library_item_t> shared_libs;
};

// the pointer
using coordinator_ptr_t = std::shared_ptr<coordinator_t>;

}
