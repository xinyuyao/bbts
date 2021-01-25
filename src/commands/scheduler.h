#pragma once

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "command.h"
#include "../communication/communicator.h"
#include "reservation_station.h"

namespace bbts {

class scheduler_t {
public:

  // init the scheduler
  explicit scheduler_t(bbts::communicator_ptr_t _comm);

  // schedule this command so it can be moved to the right node
  void schedule(command_ptr_t _cmd);

  // accept teh
  void accept();

  void forward();

  void shutdown();

private:

  // the mutex to lock
  std::mutex _m;

  // the conditional var to sync
  std::condition_variable _cv;

  // the commands we want broadcast
  std::queue<command_ptr_t> _cmds_to_bcst;

  // the communicator
  bbts::communicator_ptr_t _comm;

  // the reservation station
  bbts::reservation_station_ptr_t _rs;

  // shutdown
  std::atomic<bool> _shutdown;
};

// the pointer
using scheduler_ptr_t = std::shared_ptr<scheduler_t>;

}