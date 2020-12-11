#pragma once

#include <memory>
#include <utility>

#include "node_config.h"
#include "../commands/scheduler.h"
#include "../ud_functions/udf_manager.h"

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
  }

  void run() {

    // run the command forwarding thread
    std::thread scheduler_f([&]() {
      _scheduler->forward();
    });

    // run the accept thread
    std::thread scheduler_a([&]() {
      _scheduler->accept();
    });

    // sync all the threads
    scheduler_f.join();
    scheduler_a.join();
  }

protected:

  // the configuration of the node
  bbts::node_config_ptr_t _config;

  // the communicator, this is doing all the sending
  communicator_ptr_t _comm;

  // this is responsible for forwarding all the commands to the right node and receiving commands
  scheduler_ptr_t _scheduler;

  // this initializes the tensors
  tensor_factory_ptr_t _factory;

  // this stores all our tensors
  storage_ptr_t _storage;

  // the udf manager
  udf_manager_ptr _udf_manager;
};



}