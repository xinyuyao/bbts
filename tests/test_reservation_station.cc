#include <gtest/gtest.h>
#include "../src/commands/reservation_station.h"
#include "../src/commands/commands.h"

namespace bbts {

TEST(TestReservationStation, FewLocalCommands1) {

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
  EXPECT_TRUE(rs.queue_command(std::make_unique<command_t>(command_t{._id = 0, 
                                                         ._type = command_t::op_type_t::APPLY, 
                                                         ._input_tensors = {command_t::tid_node_id_t{.tid = 0, .node = 0}},
                                                         ._output_tensors = {command_t::tid_node_id_t{.tid = 2, .node = 0}},
                                                         ._fun_id = {0, 0}})));

  // make a command that deletes tensor with tid 0
  EXPECT_TRUE(rs.queue_command(std::make_unique<command_t>(command_t{._id = 1, 
                                                                     ._type = command_t::op_type_t::DELETE, 
                                                                     ._input_tensors = {command_t::tid_node_id_t{.tid = 0, .node = 0}},
                                                                     ._output_tensors = {},
                                                                     ._fun_id = {0, 0}})));


  // make a command that runs a reduce
  EXPECT_TRUE(rs.queue_command(std::make_unique<command_t>(command_t{._id = 2, 
                                                                     ._type = command_t::op_type_t::REDUCE, 
                                                                     ._input_tensors = {command_t::tid_node_id_t{.tid = 1, .node = 0},
                                                                                        command_t::tid_node_id_t{.tid = 2, .node = 0}},
                                                                     ._output_tensors = {command_t::tid_node_id_t{.tid = 3, .node = 0}},
                                                                     ._fun_id = {0, 0}})));

  // make a command that deletes all the tensors except for the tid = 3 tensor
  EXPECT_TRUE(rs.queue_command(std::make_unique<command_t>(command_t{._id = 3, 
                                                                     ._type = command_t::op_type_t::DELETE, 
                                                                     ._input_tensors = {command_t::tid_node_id_t{.tid = 1, .node = 0},
                                                                                        command_t::tid_node_id_t{.tid = 2, .node = 0}},
                                                                     ._output_tensors = {},
                                                                     ._fun_id = {0, 0}})));


  // get the first command to execute
  auto c1 = rs.get_next_command();

  // retire the command as we pretend we have executed it
  storage->create_tensor(2, 100);
  EXPECT_TRUE(rs.retire_command(std::move(c1)));

  // get the next command 
  auto c2 = rs.get_next_command();
  storage->create_tensor(3, 100);
  EXPECT_TRUE(rs.retire_command(std::move(c2)));

  // make sure there is only one tensors
  EXPECT_EQ(storage->get_num_tensors(), 1);
}


TEST(TestReservationStation, FewLocalCommands2) {

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
  EXPECT_TRUE(rs.queue_command(std::make_unique<command_t>(command_t{._id = 0, 
                                                                     ._type = command_t::op_type_t::APPLY, 
                                                                     ._input_tensors = {command_t::tid_node_id_t{.tid = 0, .node = 0}},
                                                                     ._output_tensors = {command_t::tid_node_id_t{.tid = 2, .node = 0}},
                                                                     ._fun_id = {0, 0}})));

  // make a command that runs a reduce
  EXPECT_TRUE(rs.queue_command(std::make_unique<command_t>(command_t{._id = 2, 
                                                                     ._type = command_t::op_type_t::REDUCE, 
                                                                     ._input_tensors = {command_t::tid_node_id_t{.tid = 1, .node = 0},
                                                                                        command_t::tid_node_id_t{.tid = 2, .node = 0}},
                                                                     ._output_tensors = {command_t::tid_node_id_t{.tid = 3, .node = 0}},
                                                                     ._fun_id = {0, 0}})));




  // get the first command to execute
  auto c1 = rs.get_next_command();

  // retire the command as we pretend we have executed it
  storage->create_tensor(2, 100);
  EXPECT_TRUE(rs.retire_command(std::move(c1)));

  // get the next command 
  auto c2 = rs.get_next_command();
  storage->create_tensor(3, 100);
  EXPECT_TRUE(rs.retire_command(std::move(c2)));

  // make a command that deletes all the tensors except for the tid = 3 tensor
  EXPECT_TRUE(rs.queue_command(std::make_unique<command_t>(command_t{._id = 3, 
                                                                     ._type = command_t::op_type_t::DELETE, 
                                                                     ._input_tensors = {command_t::tid_node_id_t{.tid = 0, .node = 0},
                                                                                        command_t::tid_node_id_t{.tid = 1, .node = 0},
                                                                                        command_t::tid_node_id_t{.tid = 2, .node = 0},
                                                                                        command_t::tid_node_id_t{.tid = 3, .node = 0}},
                                                                     ._output_tensors = {},
                                                                     ._fun_id = {0, 0}})));
  // make sure there is only one tensors
  EXPECT_EQ(storage->get_num_tensors(), 0);
}

TEST(TestReservationStation, LotOfLocalCommands) {

}

}