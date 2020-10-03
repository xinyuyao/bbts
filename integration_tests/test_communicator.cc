#include "../src/communication/communicator.h"
#include "../src/server/node.h"

// starts our program
int main(int argc, char *argv[]) {

  // make the configuration
  auto config = std::make_shared<bbts::node_config_t>();

  // set the argv and argc
  config->argv = argv;
  config->argc = argc;

  // make the communicator
  bbts::communicator c(config);

  return 0;
}