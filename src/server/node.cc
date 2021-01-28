#include "node.h"

void bbts::node_t::init() {

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
void bbts::node_t::run() {

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

  //
  command_expect.join();

  //
  tsn_thread.join();

  //
  for(auto &cpt : command_processing_threads) {
    cpt.join();
  }

  //
  deleter.join();
}

size_t bbts::node_t::get_num_nodes() const {
  return _comm->get_num_nodes();
}

size_t bbts::node_t::get_rank() const {
  return _comm->get_rank();
}

size_t bbts::node_t::get_physical_cores() const {
  return _config->num_threads;
}

size_t bbts::node_t::get_total_ram() const {
  return _config->total_ram;
}

void bbts::node_t::print_cluster_info(std::ostream& out) {

  out << "Cluster Information : \n";
  out << "\tNumber of Nodes : " << _comm->get_num_nodes() << " \n";
  out << "\tNumber of physical cores : " << _config->num_threads << " \n";
  out << "\tTotal RAM : " << _config->total_ram / (1024 * 1024) << " MB \n";
}

void bbts::node_t::load_commands(const std::vector<command_ptr_t> &commands) {

  // schedule them all at once
  for (auto &_cmd : commands) {

    // if it uses the node
    if (_cmd->uses_node(_comm->get_rank())) {
      _res_station->queue_command(_cmd->clone());
    }
  }
}

void bbts::node_t::sync() {

  _comm->barrier();
}

void bbts::node_t::shutdown() {

  // sync
  _comm->barrier();

  // shutdown the command runner
  _command_runner->shutdown();

  // shutdown the reservation station
  _res_station->shutdown();

  // shutdown the tensor notifier
  _tensor_notifier->shutdown();
}

std::thread bbts::node_t::create_deleter_thread() {

  // create the thread
  return std::thread([this]() {

    _command_runner->run_deleter();
  });
}
std::thread bbts::node_t::create_command_processing_thread() {

  // create the thread to pull
  std::thread t = std::thread([this]() {

    _command_runner->local_command_runner();
  });

  return std::move(t);
}
std::thread bbts::node_t::expect_remote_command() {

  // create the thread
  std::thread t = std::thread([this]() {

    _command_runner->remote_command_handler();
  });

  return std::move(t);
}
std::thread bbts::node_t::remote_tensor_notification_sender(bbts::node_id_t out_node) {

  // create the thread
  std::thread t = std::thread([out_node, this]() {

    // this will send notifications to out node
    _tensor_notifier->run_notification_sender_for_node(out_node);
  });

  return std::move(t);
}
std::thread bbts::node_t::tensor_notifier() {

  // create the thread
  std::thread t = std::thread([this]() {

    // run the handler for the notifications
    _tensor_notifier->run_notification_handler();
  });

  return std::move(t);
}


