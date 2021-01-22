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

  // initialize stuff like communication
  void init();

  // starts handling the commands
  void run();

  // return the nuber of nodes
  [[nodiscard]] size_t get_num_nodes() const;

  // load all commands
  void load_commands(const std::vector<command_ptr_t>& commands);

  // sync all the nodes to know that they have all execute up to this point
  void sync();

// protected: for now everything is public

  // the reservation station needs a deleter thread
  std::thread create_deleter_thread();

  std::thread create_command_processing_thread();

  std::thread expect_remote_command();

  std::thread remote_tensor_notification_sender(node_id_t out_node);

  std::thread tensor_notifier();

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