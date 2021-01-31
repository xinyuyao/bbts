#include "coordinator.h"
#include "../utils/terminal_color.h"

bbts::coordinator_t::coordinator_t(bbts::communicator_ptr_t _comm,
                                   bbts::reservation_station_ptr_t _rs) : _comm(std::move(_comm)),
                                                                          _rs(std::move(_rs)) {
  _is_down = false;
}

void bbts::coordinator_t::accept() {

  while (!_is_down) {

    // the operation
    auto op = _comm->expect_coord_op();

    switch(op._type) {

      case coordinator_op_types_t::FAIL : {  _fail(); break; }
      case coordinator_op_types_t::RUN : { _run(); break; }
      case coordinator_op_types_t::CLEAR : { _clear(); break; }
      case coordinator_op_types_t::SCHEDULE : { _schedule(op); break; }
      case coordinator_op_types_t::SHUTDOWN : { _shutdown(); break; }
    }

    // sync all nodes
    _comm->barrier();
  }
}

std::tuple<bool, std::string> bbts::coordinator_t::schedule_commands(const std::vector<command_ptr_t>& cmds) {

  if(!_comm->send_coord_op(coordinator_op_t{._type = coordinator_op_types_t::SCHEDULE,
                                                .num_cmds = cmds.size()})) {
    return {false, "Could not schedule commands!\n"};
  }

  // send all the commands
  if(!_comm->send_coord_cmds(cmds)) {
    return {false, "Could not send the commands we were about to schedule!\n"};
  }

  // load all the commands
  _load_cmds(cmds);

  // sync all nodes
  _comm->barrier();

  // we succeeded
  return {true, "Scheduled " + std::to_string(cmds.size()) + " commands\n"};
}

std::tuple<bool, std::string> bbts::coordinator_t::run_commands() {

  if(!_comm->send_coord_op(coordinator_op_t{._type = coordinator_op_types_t::RUN, .num_cmds = 0 })) {
    return {false, "Could not run the commands!\n"};
  }

  // run everything
  _run();

  // sync everything
  _comm->barrier();

  return {true, "Finished running commands\n"};
}

void bbts::coordinator_t::shutdown() {

  // mark the we are done
  _is_down = true;
}

void bbts::coordinator_t::_fail() {
  std::cout << bbts::red << "FAIL\n" << bbts::reset;
  exit(-1);
}

void bbts::coordinator_t::_schedule(coordinator_op_t op) {

  // expect all the commands
  std::vector<command_ptr_t> cmds;
  if(!_comm->expect_coord_cmds(op.num_cmds, cmds)) {
    std::cout << bbts::red << "Could not receive the scheduled commands!\n" << bbts::reset;
    return;
  }

  // load all the commands
  _load_cmds(cmds);
}

void bbts::coordinator_t::_load_cmds(const std::vector<command_ptr_t> &cmds) {

  // schedule them all at once
  for (auto &_cmd : cmds) {

    // if it uses the node
    if (_cmd->uses_node(_comm->get_rank())) {
      _rs->queue_command(_cmd->clone());
    }
  }
}

void bbts::coordinator_t::_run() {

  // async execute the scheduled commands
  _rs->execute_scheduled_async();

  // wait for all the commands to be run
  _rs->wait_until_finished();

  // stop executing all the commands
  _rs->stop_executing();
}

void bbts::coordinator_t::_clear() {
  std::cout << "CLEAR\n";
}

void bbts::coordinator_t::_shutdown() {
  std::cout << "SHUTDOWN\n";
}

