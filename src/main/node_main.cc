#include <iostream>
#include "../tensor/tensor.h"
#include "../tensor/tensor_factory.h"
#include "../server/node.h"
#include "../commands/parsed_command.h"
#include "../utils/terminal_color.h"

#include "../../third_party/cli/include/cli/cli.h"
#include "../../third_party/cli/include/cli/clifilesession.h"

using namespace cli;


void load_binary_command() {

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
  rootMenu->Insert("info",
                   [&](std::ostream &out) { node.print_cluster_info(out); },
                   "Returns information about the cluster\n");

  rootMenu->Insert("load",[&](std::ostream &out, const std::string &file) {

    out << file << '\n';

  },"Load commands form a binary file. Usage : load <file>\n");

  // init the command line interface
  Cli cli(std::move(rootMenu));

  // global exit action
  cli.ExitAction([](auto &out) { out << "Goodbye...\n"; });

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
  if (node.get_rank() == 0) {
    std::thread t = std::thread([&]() { prompt(node); });
    t.detach();
  }

  // the node
  node.run();

  return 0;
}
