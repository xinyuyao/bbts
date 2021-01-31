#pragma once

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "../server/coordinator_ops.h"
#include "../commands/command.h"
#include "../communication/communicator.h"
#include "../commands/reservation_station.h"

namespace bbts {

class coordinator_t {
public:

  // init the scheduler
  coordinator_t(bbts::communicator_ptr_t _comm, bbts::reservation_station_ptr_t _rs);

  // accept a request
  void accept();

  // schedules all the provided commands
  std::tuple<bool, std::string> schedule_commands(const std::vector<command_ptr_t>& cmds);

  // run the commands
  std::tuple<bool, std::string> run_commands();

  // shutdown the coordinator
  void shutdown();

private:

  void _fail();

  void _schedule(coordinator_op_t ops);

  void _load_cmds(const std::vector<command_ptr_t> &cmds);

  void _run();

  void _clear();

  void _shutdown();

  // the communicator
  bbts::communicator_ptr_t _comm;

  // the reservation station
  bbts::reservation_station_ptr_t _rs;

  // shutdown
  std::atomic<bool> _is_down{};
};

// the pointer
using coordinator_ptr_t = std::shared_ptr<coordinator_t>;

}