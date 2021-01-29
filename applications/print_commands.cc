#include <iostream>
#include <iostream>
#include "../src/commands/parsed_command.h"

int main(int argc, char **argv) {

  if(argc != 2) {
    std::cout << "Incorrect usage\n";
    std::cout << "Usage ./print_commands <file>.bbts\n";
  }

  // load them and print them
  bbts::parsed_command_list_t cmd_list;
  cmd_list.deserialize(std::string(argv[1]));
  cmd_list.print();

  return 0;
}