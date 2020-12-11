#pragma once

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "commands.h"
#include "../communication/communicator.h"
#include "reservation_station.h"

namespace bbts {

class scheduler_t {
public:

  // init the scheduler
  scheduler_t(bbts::communicator_ptr_t _comm) : _comm(std::move(_comm)) {
    _shutdown = false;
  }

  // schedule this command so it can be moved to the right node
  void schedule(command_ptr_t _cmd) {

    std::unique_lock<std::mutex> lk(_m);

    std::cout << "Added\n";
    // store the command
    _cmds_to_bcst.push(std::move(_cmd));
    _cv.notify_one();
  }

  void accept() {

    // receive the commands
    while(true) {

      // get the next command
      auto _cmd = _comm->expect_cmd();

      std::cout << "Got\n";

      // check if we are done that is if the command is null
      if(_cmd == nullptr) {
        break;
      }

      // add the command to the reservation station
      _rs->queue_command(std::move(_cmd));
    }
  }

  void forward() {

    // process it
    while(true) {

      // the command we want to forward
      command_ptr_t _cmd;

      // get the lock
      std::unique_lock<std::mutex> lk(_m);
      _cv.wait(lk, [&]{ return _shutdown || !_cmds_to_bcst.empty(); });

      // check if we are shutdown
      if(_shutdown) {
        break;
      }

      // grab the command
      _cmd = std::move(_cmds_to_bcst.front());
      _cmds_to_bcst.pop();

      // unlock here
      lk.unlock();

      // forward the command to the right nodes
      _comm->forward_cmd(_cmd);

      // move it to the reservation station
      if(_cmd->uses_node(_comm->get_rank())) {
        _rs->queue_command(std::move(_cmd));
      }
    }
  }

  void shutdown() {

    std::unique_lock<std::mutex> lk(_m);

    // mark the we are done
    _shutdown = true;
    _cv.notify_one();
  }

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