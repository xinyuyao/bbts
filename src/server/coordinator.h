#pragma once

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "../server/coordinator_ops.h"
#include "../server/logger.h"
#include "../commands/command.h"
#include "../communication/communicator.h"
#include "../commands/reservation_station.h"

namespace bbts {

class coordinator_t {
public:

  // init the scheduler
  coordinator_t(bbts::communicator_ptr_t _comm,
                bbts::reservation_station_ptr_t _rs,
                bbts::logger_ptr_t _logger);

  // accept a request
  void accept();

  // schedules all the provided commands
  std::tuple<bool, std::string> schedule_commands(const std::vector<command_ptr_t>& cmds);

  // run the commands
  std::tuple<bool, std::string> run_commands();

  // set the verbose status
  std::tuple<bool, std::string> set_verbose(bool val);

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

  // the communicator
  bbts::communicator_ptr_t _comm;

  // the reservation station
  bbts::reservation_station_ptr_t _rs;

  // the logger
  bbts::logger_ptr_t _logger;

  // shutdown
  std::atomic<bool> _is_down{};
};

// the pointer
using coordinator_ptr_t = std::shared_ptr<coordinator_t>;

}