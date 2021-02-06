#include <chrono>
#include <utility>
#include <unistd.h>
#include "coordinator.h"
#include "../utils/terminal_color.h"

using namespace std::chrono;

bbts::coordinator_t::coordinator_t(bbts::communicator_ptr_t _comm,
                                   bbts::reservation_station_ptr_t _rs,
                                   bbts::logger_ptr_t _logger,
                                   storage_ptr_t _storage,
                                   bbts::command_runner_ptr_t _command_runner,
                                   bbts::tensor_notifier_ptr_t _tensor_notifier,
                                   bbts::tensor_factory_ptr_t _tensor_factory,
                                   bbts::udf_manager_ptr _udf_manager)
    : _comm(std::move(_comm)),
      _rs(std::move(_rs)),
      _logger(std::move(_logger)),
      _storage(std::move(_storage)),
      _command_runner(std::move(_command_runner)),
      _tensor_notifier(std::move(_tensor_notifier)), 
      _tensor_factory(std::move(_tensor_factory)),
      _udf_manager(std::move(_udf_manager)) { _is_down = false; }

void bbts::coordinator_t::accept() {

  while (!_is_down) {

    // the operation
    auto op = _comm->expect_coord_op();

    switch (op._type) {

      case coordinator_op_types_t::FAIL : {
        _fail();
        break;
      }
      case coordinator_op_types_t::RUN : {
        _run();
        break;
      }
      case coordinator_op_types_t::CLEAR : {
        _clear();
        break;
      }
      case coordinator_op_types_t::SCHEDULE : {
        _schedule(op);
        break;
      }
      case coordinator_op_types_t::SHUTDOWN : {
        _shutdown();
        break;
      }
      case coordinator_op_types_t::VERBOSE : {
        _set_verbose(static_cast<bool>(op._val));
        break;
      }
      case coordinator_op_types_t::PRINT_STORAGE : {
        _print_storage();
        break;
      }
      case coordinator_op_types_t::REGISTER : {
        _register(op);
        break;
      }

    }

    // sync all nodes
    _comm->barrier();
  }
}

std::tuple<bool, std::string> bbts::coordinator_t::schedule_commands(const std::vector<command_ptr_t> &cmds) {

  if (!_comm->send_coord_op(coordinator_op_t{._type = coordinator_op_types_t::SCHEDULE,
      ._val = cmds.size()})) {
    return {false, "Could not schedule commands!\n"};
  }

  // send all the commands
  if (!_comm->send_coord_cmds(cmds)) {
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

  if (!_comm->send_coord_op(coordinator_op_t{._type = coordinator_op_types_t::RUN, ._val = 0})) {
    return {false, "Could not run the commands!\n"};
  }

  // run everything
  _run();

  // sync everything
  _comm->barrier();

  auto end = high_resolution_clock::now();
  auto duration =
      (double) duration_cast<microseconds>(end - start).count() / (double) duration_cast<microseconds>(1s).count();

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
  if (!_comm->expect_coord_cmds(op._val, cmds)) {
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

  if (!_comm->send_coord_op(coordinator_op_t{._type = coordinator_op_types_t::VERBOSE,
      ._val = static_cast<size_t>(val)})) {
    return {false, "Failed to set the verbose flag!\n"};
  }

  // run everything
  _set_verbose(val);

  // sync everything
  _comm->barrier();

  return {true, "Set the verbose flag to " + std::to_string(val) + "\n"};
}

std::tuple<bool, std::string> bbts::coordinator_t::print_storage_info() {

  if (!_comm->send_coord_op(coordinator_op_t{._type = coordinator_op_types_t::PRINT_STORAGE, ._val = 0})) {
    return {false, "Failed to set the verbose flag!\n"};
  }

  // print the storage
  _print_storage();

  // we succeded
  return {true, ""};
}

std::tuple<bool, std::string> bbts::coordinator_t::clear() {

  if (!_comm->send_coord_op(coordinator_op_t{._type = coordinator_op_types_t::CLEAR, ._val = 0})) {
    return {false, "Failed to clear the cluster!\n"};
  }

  // print the storage
  _clear();

  // sync everything
  _comm->barrier();

  // we succeded
  return {true, "Cleared the cluster!\n"};
}

std::tuple<bool, std::string> bbts::coordinator_t::shutdown_cluster() {

  if (!_comm->send_coord_op(coordinator_op_t{._type = coordinator_op_types_t::SHUTDOWN, ._val = 0})) {
    return {false, "Failed to shutdown the cluster!\n"};
  }

  // print the storage
  _shutdown();

  // sync everything
  _comm->barrier();

  // we succeded
  return {true, "Cluster shutdown!\n"};
}

std::tuple<bool, std::string> bbts::coordinator_t::load_shared_library(char* file_bytes, size_t file_size) {
  if(!_comm->send_coord_op(coordinator_op_t{._type = coordinator_op_types_t::REGISTER, ._val = file_size})) {
    return {false, "Failed to register library!\n"};
  }

  // send the data
  if (!_comm->send_bytes(file_bytes, file_size)) {
    return {false, "Could not send file to register!\n"};
  }
  
  // do the actual registering, on this node
  _register_from_bytes(file_bytes, file_size);

  // sync everything
  _comm->barrier();

  return {true, "Registered library!\n"};
}

void bbts::coordinator_t::_clear() {

  // clear everything
  _storage->clear();
  _rs->clear();
}

void bbts::coordinator_t::_shutdown() {

  // sync
  _comm->barrier();

  // shutdown the command runner
  _command_runner->shutdown();

  // shutdown the reservation station
  _rs->shutdown();

  // shutdown the tensor notifier
  _tensor_notifier->shutdown();

  // mark that the coordinator is down
  _is_down = true;
}

void bbts::coordinator_t::_set_verbose(bool val) {
  _logger->set_enabled(val);
}

void bbts::coordinator_t::_print_storage() {

  // each node gets a turn
  for (node_id_t node = 0; node < _comm->get_num_nodes(); ++node) {

    // check the rank of the node
    if (node == _comm->get_rank()) {
      std::cout << "<<< For Node " << _comm->get_rank() << ">>>\n";
      _storage->print();
    }

    _comm->barrier();
  }

  // final sync just in case
  _comm->barrier();
}

void bbts::coordinator_t::_register(coordinator_op_t op) {

  std::vector<char> file_bytes;
  file_bytes.reserve(op._val);

  if(!_comm->expect_bytes(op._val, file_bytes)) {
    std::cout << bbts::red << "Could not recieve the library file!\n" << bbts::reset;
    return;
  }

  _register_from_bytes(file_bytes.data(), op._val);
}

void bbts::coordinator_t::_register_from_bytes(char* file_bytes, size_t file_size) {
  int rank = _comm->get_rank();
  std::string filename_template = "/tmp/register_from_bytes_" + std::to_string(rank) + "_XXXXXX";

  auto filename = std::unique_ptr<char[]>(new char[filename_template.size()]);

  std::copy(filename_template.begin(), filename_template.end(), filename.get());

  // this will modify filename
  int filedes = mkstemp(filename.get());

  if(filedes == -1) {
    std::cout << bbts::red << "Could not set temporary filename!\n" << bbts::reset;
    return;
  }
  if(-1 == write(filedes,file_bytes,file_size)) {
    std::cout << bbts::red << "Could not write shared library object!\n" << bbts::reset;
    return;
  }

  // open the newly created temporary file
  void* so_handle = dlopen(filename.get(), RTLD_LOCAL | RTLD_NOW);

  if(!so_handle) {
    std::cout << bbts::red << "Could not open temporary shared library object!\n" << bbts::reset;
    return;
  }

  // The .so should have atleast one of the two (unmangled) functions, 
  //  register_tensors, register_udfs. 
  // Call them here.
  bool did_register = false;
  void* register_tensors_ = dlsym(so_handle, "register_tensors");
  if(register_tensors_) {
    did_register = true;
    typedef void *register_tensors_f(tensor_factory_ptr_t);
    auto *register_tensors = (register_tensors_f *) register_tensors_;
    register_tensors(_tensor_factory);
  }

  void* register_udfs_ = dlsym(so_handle, "register_udfs");
  if(register_udfs_) {
    did_register = true;
    typedef void *register_udfs_f(udf_manager_ptr);
    auto *register_udfs = (register_udfs_f *) register_udfs_;
    register_udfs(_udf_manager);
  }

  if(!did_register) {
    std::cout << bbts::red << "Shared library object did not have a valid \"register_tensors\" or \"register_udfs\"!\n" << std::endl;
  }  

  // keep track of the stuff here so the system can clean it up later
  shared_libs.emplace_back(filename.get(), so_handle);
}

