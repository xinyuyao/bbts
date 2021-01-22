#pragma once

#include <memory>
#include <utility>

#include "node_config.h"
#include "../commands/scheduler.h"
#include "../ud_functions/udf_manager.h"
#include "../commands/command_runner.h"
#include "../commands/tensor_notifier.h"

namespace bbts {


class node_t {
public:

  // creates the node
  explicit node_t(bbts::node_config_ptr_t config) : _config(std::move(config)) {}

  void init() {

    // the communicator
    _comm = std::make_shared<communicator_t>(_config);

    // the scheduler
    _scheduler = std::make_shared<scheduler_t>(_comm);

    // create the storage
    _storage = std::make_shared<storage_t>();

    // init the factory
    _factory = std::make_shared<bbts::tensor_factory_t>();

    // init the udf manager
    _udf_manager = std::make_shared<bbts::udf_manager>(_factory);

    // init the reservation station
    _res_station = std::make_shared<bbts::reservation_station_t>(_comm->get_rank(), _comm->get_num_nodes());

    // this runs commands
    _command_runner = std::make_shared<bbts::command_runner_t>(_storage, _factory, _udf_manager, _res_station, _comm);

    // the tensor notifier
    _tensor_notifier = std::make_shared<bbts::tensor_notifier_t>(_comm, _res_station);
  }

  void run() {

    /// 1.0 Kick off all the stuff that needs to run

    // this will delete the tensors
    auto deleter = create_deleter_thread();

    // the command processing threads
    std::vector<std::thread> command_processing_threads;
    command_processing_threads.reserve(_comm->get_num_nodes());
    for (node_id_t t = 0; t < _config->num_threads; ++t) {
      command_processing_threads.push_back(std::move(create_command_processing_thread()));
    }

    // this will get all the notifications about tensors
    auto tsn_thread = tensor_notifier();

    // this kicks off and handles remove commands (MOVE and REDUCE)
    auto command_expect = expect_remote_command();

    // notification sender
    std::vector<std::thread> remote_notification_sender;
    remote_notification_sender.reserve(_config->num_threads);
    for(node_id_t node = 0; node < _comm->get_num_nodes(); ++node) {

      // no need to notify self so skip that
      if(node == _comm->get_rank()) {
        continue;
      }

      // create the notifier thread
      remote_notification_sender.push_back(remote_tensor_notification_sender(node));
    }

    /// 2.0 Wait for stuff to finish

    // wa
    for(auto &rns : remote_notification_sender) {
      rns.join();
    }

    command_expect.join();

    tsn_thread.join();

    for(auto &cpt : command_processing_threads) {
      cpt.join();
    }

    deleter.join();
  }

  // return the nuber of nodes
  [[nodiscard]] size_t get_num_nodes() const {
    return _comm->get_num_nodes();
  }

  // load all commands
  void load_commands(const std::vector<command_ptr_t>& commands) {

    // schedule them all at once
    for (auto &_cmd : commands) {

      // if it uses the node
      if (_cmd->uses_node(_comm->get_rank())) {
        _res_station->queue_command(_cmd->clone());
      }
    }
  }

  // sync all the nodes to know that they have all execute up to this point
  void sync() {

    _comm->barrier();
  }

// protected: for now everything is public

  // the reservation station needs a deleter thread
  std::thread create_deleter_thread() {

    // create the thread
    return std::thread([this]() {

      _command_runner->run_deleter();
    });
  }

  std::thread create_command_processing_thread() {

    // create the thread to pull
    std::thread t = std::thread([this]() {

      _command_runner->local_command_runner();
    });

    return std::move(t);
  }

  std::thread expect_remote_command() {

    // create the thread
    std::thread t = std::thread([this]() {

      _command_runner->remote_command_handler();
    });

    return std::move(t);
  }

  std::thread remote_tensor_notification_sender(node_id_t out_node) {

    // create the thread
    std::thread t = std::thread([out_node, this]() {

      // this will send notifications to out node
      _tensor_notifier->run_notification_sender_for_node(out_node);
    });

    return std::move(t);
  }

  std::thread create_notification_handler_thread() {

    // create the thread
    std::thread t = std::thread([this]() {

      // run the handler for the notifications
      _tensor_notifier->run_notification_handler();
    });

    return std::move(t);
  }

  std::thread tensor_notifier() {

    // create the thread
    std::thread t = std::thread([this]() {

      // run the handler for the notifications
      _tensor_notifier->run_notification_handler();
    });

    return std::move(t);
  }

  // the configuration of the node
  node_config_ptr_t _config;

  // the communicator, this is doing all the sending
  communicator_ptr_t _comm;

  // the reservation station
  reservation_station_ptr_t _res_station;

  // this is responsible for forwarding all the commands to the right node and receiving commands
  scheduler_ptr_t _scheduler;

  // this initializes the tensors
  tensor_factory_ptr_t _factory;

  // this stores all our tensors
  storage_ptr_t _storage;

  // the udf manager
  udf_manager_ptr _udf_manager;

  // runs commands
  command_runner_ptr_t _command_runner;

  // the notifier
  tensor_notifier_ptr_t _tensor_notifier;
};



}