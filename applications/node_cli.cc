#include <bits/stdint-intn.h>
#include <iostream>
#include <memory>
#include "../src/tensor/tensor.h"
#include "../src/tensor/tensor_factory.h"
#include "../src/server/node.h"

#include "../third_party/cli/include/cli/cli.h"
#include "../third_party/cli/include/cli/clifilesession.h"

using namespace cli;

std::thread loading_message(const std::string &s, std::atomic_bool &b) {

  auto t = std::thread([s, &b]() {

    // as long as we load
    int32_t dot = 0;
    while(!b) {

      std::cout << '\r' << s;
      for(int32_t i = 0; i < dot; ++i) { std::cout << '.';}
      dot = (dot + 1) % 4;
      usleep(300000);
    }

    std::cout << '\n';
  });

  return std::move(t);
}

void load_binary_command(bbts::node_t &node, const std::string &file_path) {

  // kick of a loading message
  std::atomic_bool b; b = false;
  auto t = loading_message("Loading the file", b);

  // try to deserialize
  bbts::parsed_command_list_t cmd_list;
  bool success = cmd_list.deserialize(file_path);

  // finish the loading message
  b = true; t.join();

  // did we fail
  if(!success) {
    std::cout << bbts::red << "Failed to load the file " << file_path << '\n' << bbts::reset;
    return;
  }

  // kick of a loading message
  b = false;
  t = loading_message("Scheduling the loaded commands", b);

  // load the commands we just parsed
  auto [did_load, message] = node.load_commands(cmd_list);

  // finish the loading message
  b = true; t.join();

  // did we fail
  if(!did_load) {
    std::cout << bbts::red << "Failed to schedule the loaded commands : \"" << message << "\"\n" << bbts::reset;
  }
  else {
    std::cout << bbts::green << message << bbts::reset;
  }
}

void run_commands(bbts::node_t &node) {

  // kick of a loading message
  std::atomic_bool b; b = false;
  auto t = loading_message("Running the commands", b);

  // run all the commands
  auto [did_load, message] = node.run_commands();

  // finish the loading message
  b = true; t.join();

  // did we fail
  if(!did_load) {
    std::cout << bbts::red << "Failed to run commands : \"" << message << "\"\n" << bbts::reset;
  }
  else {
    std::cout << bbts::green << message << bbts::reset;
  }
}

void print(bbts::node_t &node, const std::string &file_path) {

  // kick of a loading message
  std::atomic_bool b; b = false;
  auto t = loading_message("Loading the file", b);

  // try to deserialize
  bbts::parsed_command_list_t cmd_list;
  bool success = cmd_list.deserialize(file_path);

  // finish the loading message
  b = true; t.join();

  // did we fail
  if(!success) {
    std::cout << bbts::red << "Failed to load the file " << file_path << '\n' << bbts::reset;
  }

  // print out the commands
  cmd_list.print(std::cout);
}

void clear(bbts::node_t &node) {

  // kick of a loading message
  std::atomic_bool b; b = false;
  auto t = loading_message("Clearing", b);

  // run all the commands
  auto [did_load, message] = node.clear();

  // finish the loading message
  b = true; t.join();

  // did we fail
  if(!did_load) {
    std::cout << bbts::red << "Failed to clear : \"" << message << "\"\n" << bbts::reset;
  }
  else {
    std::cout << bbts::green << message << bbts::reset;
  }
}

void verbose(bbts::node_t &node, bool val) {

  // kick of a loading message
  std::atomic_bool b; b = false;
  auto t = loading_message("Set verbose", b);

  // run all the commands
  auto [did_load, message] = node.set_verbose(val);

  // finish the loading message
  b = true; t.join();

  // did we fail
  if(!did_load) {
    std::cout << bbts::red << "Set to fail verbose : \"" << message << "\"\n" << bbts::reset;
  }
  else {
    std::cout << bbts::green << message << bbts::reset;
  }
}

void shutdown(bbts::node_t &node) {

  // kick of a loading message
  std::atomic_bool b; b = false;
  auto t = loading_message("Shutting down", b);

  // run all the commands
  auto [did_load, message] = node.shutdown_cluster();

  // finish the loading message
  b = true; t.join();

  // did we fail
  if(!did_load) {
    std::cout << bbts::red << "Failed to shutdown : \"" << message << "\"\n" << bbts::reset;
  }
  else {
    std::cout << bbts::green << message << bbts::reset;
  }
}


// the prompt
void prompt(bbts::node_t &node) {

  std::cout << "\n";
  std::cout << "\t\t    \\   /  \\   /  \\   /  \\   /  \\   /  \\   /  \\   /  \\   /  \n";
  std::cout << "\t\t-----///----///----///----///----///----///----///----///-----\n";
  std::cout << "\t\t    /   \\  /   \\  /   \\  /   \\  /   \\  /   \\  /   \\  /   \\  \n";
  std::cout << "\n";
  std::cout << "\t\t\tWelcome to " << bbts::green << "BarbaTOS" << bbts::reset << ", the tensor operating system\n";
  std::cout << "\t\t\t\tVersion : 0.1 - Lupus Rex\n";
  std::cout << "\t\t\t\tEmail : dj16@rice.edu\n";
  std::cout << '\n';

  auto rootMenu = std::make_unique<Menu>("cli");

  // setup the info command
  rootMenu->Insert("info",[&](std::ostream &out, const std::string &what) {

    if(what == "cluster") {
      node.print_cluster_info(out);
    }
    else if(what == "storage") {
      node.print_storage_info();
    }
    
  },"Returns information about the cluster. Usage : info [cluster, storage, tensor] \n ");

  rootMenu->Insert("info",[&](std::ostream &out, const std::string &what, int32_t id) {

    if(what == "tensor") {
      node.print_tensor_info(static_cast<bbts::tid_t>(id));
    }

  },"Returns information about the cluster. Usage : info [cluster, storage, tensor] [tid] \n ");

  rootMenu->Insert("load",[&](std::ostream &out, const std::string &file) {

    load_binary_command(node, file);

  },"Load commands form a binary file. Usage : load <file>\n");


  rootMenu->Insert("run",[&](std::ostream &out) {

    run_commands(node);

  },"Run scheduled commands.\n");

  rootMenu->Insert("verbose",[&](std::ostream &out, bool val) {

    verbose(node, val);

  },"Enables or disables debug messages. verbose [true|false]\n");

  rootMenu->Insert("print",[&](std::ostream &out, const std::string &file) {

    print(node, file);

  },"Prints command stored in a file. Usage : print <file>\n");

  rootMenu->Insert("clear",[&](std::ostream &out) {

    clear(node);

  },"Clears the tensor operating system.\n");

  // init the command line interface
  Cli cli(std::move(rootMenu));

  // global exit action
  cli.ExitAction([&](auto &out) { shutdown(node); });

  // start the cli session
  CliFileSession input(cli);
  input.Start();
}

int main(int argc, char **argv) {

  // make the configuration
  auto config = std::make_shared<bbts::node_config_t>(bbts::node_config_t{.argc=argc, .argv = argv, .num_threads = 8});

  // create the node
  bbts::node_t node(config);

  // init the node
  node.init();

  // sync everything
  node.sync();

  // kick off the prompt
  std::thread t;
  if (node.get_rank() == 0) {
    t = std::thread([&]() { prompt(node); });
  }

  // the node
  node.run();

  // wait for the prompt to finish
  if (node.get_rank() == 0) { t.join();}

  return 0;
}
