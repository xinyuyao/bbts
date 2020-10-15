#include <gtest/gtest.h>
#include "../src/commands/reservation_station.h"
#include "../src/commands/commands.h"

namespace bbts {

TEST(TestReservationStation, FewLocalCommands) {

  // create the storage
  storage_ptr_t storage = std::make_shared<storage_t>();

  // create two input tensors
  storage->create_tensor(0, 100);
  storage->create_tensor(1, 100);

  // create the reservation station
  reservation_station_t rs(0, storage);

  // register the tensor
  rs.register_tensor(0);
  rs.register_tensor(1);

  // make a command that applies something to tensor 0
  rs.queue_command(std::make_unique<command_t>(command_t{._id = 0, 
                                                         ._type = command_t::op_type_t::APPLY, 
                                                         ._input_tensors = {command_t::tid_node_id_t{.tid = 0, .node = 0}},
                                                         ._output_tensors = {command_t::tid_node_id_t{.tid = 2, .node = 0}},
                                                         ._fun_id = 0}));

  // make a command that deletes tensor with tid 0
  rs.queue_command(std::make_unique<command_t>(command_t{._id = 1, 
                                                         ._type = command_t::op_type_t::DELETE, 
                                                         ._input_tensors = {command_t::tid_node_id_t{.tid = 0, .node = 0}},
                                                         ._output_tensors = {},
                                                         ._fun_id = -1}));


  // make a command that runs a reduce
  rs.queue_command(std::make_unique<command_t>(command_t{._id = 2, 
                                                         ._type = command_t::op_type_t::REDUCE, 
                                                         ._input_tensors = {command_t::tid_node_id_t{.tid = 1, .node = 0},
                                                                            command_t::tid_node_id_t{.tid = 2, .node = 0}},
                                                         ._output_tensors = {command_t::tid_node_id_t{.tid = 3, .node = 0}},
                                                         ._fun_id = 0}));

  // make a command that deletes all the tensors
  rs.queue_command(std::make_unique<command_t>(command_t{._id = 3, 
                                                         ._type = command_t::op_type_t::DELETE, 
                                                         ._input_tensors = {command_t::tid_node_id_t{.tid = 1, .node = 0},
                                                                            command_t::tid_node_id_t{.tid = 2, .node = 0}},
                                                         ._output_tensors = {},
                                                         ._fun_id = -1}));

}

}