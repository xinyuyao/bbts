#include <chrono>
#include "coordinator.h"
#include "../utils/terminal_color.h"

using namespace std::chrono;

bbts::coordinator_t::coordinator_t(bbts::communicator_ptr_t _comm,
                                   bbts::reservation_station_ptr_t _rs,
                                   bbts::logger_ptr_t _logger,
                                   storage_ptr_t _storage) : _comm(std::move(_comm)),
                                                             _rs(std::move(_rs)),
                                                             _logger(std::move(_logger)),
                                                             _storage(std::move(_storage)) {
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
      case coordinator_op_types_t::VERBOSE : { _set_verbose(static_cast<bool>(op._val)); break; }
      case coordinator_op_types_t::PRINT_STORAGE : { _print_storage(); break; }
    }

    // sync all nodes
    _comm->barrier();
  }
}

std::tuple<bool, std::string> bbts::coordinator_t::schedule_commands(const std::vector<command_ptr_t>& cmds) {

  if(!_comm->send_coord_op(coordinator_op_t{._type = coordinator_op_types_t::SCHEDULE,
                                                ._val = cmds.size()})) {
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

  auto start = high_resolution_clock::now();

  if(!_comm->send_coord_op(coordinator_op_t{._type = coordinator_op_types_t::RUN, ._val = 0 })) {
    return {false, "Could not run the commands!\n"};
  }

  // run everything
  _run();

  // sync everything
  _comm->barrier();

  auto end = high_resolution_clock::now();
  auto duration = (double) duration_cast<microseconds>(end - start).count() / (double) duration_cast<microseconds>(1s).count();

  return {true, "Finished running commands in " + std::to_string(duration) + "s \n"};
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
  if(!_comm->expect_coord_cmds(op._val, cmds)) {
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

std::tuple<bool, std::string> bbts::coordinator_t::set_verbose(bool val) {

  if(!_comm->send_coord_op(coordinator_op_t{._type = coordinator_op_types_t::VERBOSE,
                                                ._val = static_cast<size_t>(val) })) {
    return {false, "Failed to set the verbose flag!\n"};
  }

  // run everything
  _set_verbose(val);

  // sync everything
  _comm->barrier();

  return {true, "Set the verbose flag to " + std::to_string(val) + "\n"};
}

std::tuple<bool, std::string> bbts::coordinator_t::print_storage_info() {

  if(!_comm->send_coord_op(coordinator_op_t{._type = coordinator_op_types_t::PRINT_STORAGE, ._val = 0 })) {
    return {false, "Failed to set the verbose flag!\n"};
  }

  // print the storage
  _print_storage();

  // we succeded
  return {true, ""};
}

std::tuple<bool, std::string> bbts::coordinator_t::clear() {

  if(!_comm->send_coord_op(coordinator_op_t{._type = coordinator_op_types_t::CLEAR, ._val = 0 })) {
    return {false, "Failed to clear the cluster!\n"};
  }

  // print the storage
  _clear();

  // sync everything
  _comm->barrier();

  // we succeded
  return {true, "Cleared the cluster!"};
}

void bbts::coordinator_t::_clear() {

  // clear everything
  _storage->clear();
  _rs->clear();
}

void bbts::coordinator_t::_shutdown() {
  std::cout << "SHUTDOWN\n";
}

void bbts::coordinator_t::_set_verbose(bool val) {
  _logger->set_enabled(val);
}

void bbts::coordinator_t::_print_storage() {

  // each node gets a turn
  for(node_id_t node = 0; node < _comm->get_num_nodes(); ++node) {

    // check the rank of the node
    if(node == _comm->get_rank()) {
      std::cout << "<<< For Node " << _comm->get_rank() << ">>>\n";
      _storage->print();
    }

    _comm->barrier();
  }

  // final sync just in case
  _comm->barrier();
}