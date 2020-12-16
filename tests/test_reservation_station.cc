#include <gtest/gtest.h>
#include <thread>
#include "../src/commands/reservation_station.h"

namespace bbts {

// the reservation station needs a deleter thread
std::thread create_deleter_thread(reservation_station_ptr_t &_rs, storage_ptr_t &_sto) {

  // create the thread
  return std::thread([_rs, _sto]() {

    // while we have something remove
    tid_t id;
    while((id = _rs->get_to_remove()) != -1) {
      _sto->remove_by_tid(id);
    }

  });
}

TEST(TestReservationStation, FewLocalCommands1) {

  // tensors = { (0, 0), (1, 0) }
  // APPLY (.input = {(0, 0)}, .output = {(2, 0)})
  // DELETE (.input = {(0, 0)})
  // REDUCE (.input = {(1, 0), (2, 0)}, .output = {(3, 0)})
  // DELETE (.input = {(1, 0), (2, 0)})

  // create the storage
  storage_ptr_t storage = std::make_shared<storage_t>();

  // create two input tensors
  storage->create_tensor(0, 100);
  storage->create_tensor(1, 100);

  // create the reservation station
  auto rs = std::make_shared<reservation_station_t>(0, 1);

  // create the deleter thread
  auto deleter = create_deleter_thread(rs, storage);

  // register the tensor
  rs->register_tensor(0);
  rs->register_tensor(1);

  // make a command that applies something to tensor 0
  EXPECT_TRUE(rs->queue_command(command_t::create_unique(0,
                                       command_t::op_type_t::APPLY,
                                       {0, 0},
                                       {command_t::tid_node_id_t{.tid = 0, .node = 0}},
                                       {command_t::tid_node_id_t{.tid = 2, .node = 0}})));

  // make a command that deletes tensor with tid 0
  EXPECT_TRUE(rs->queue_command(command_t::create_unique(1,
                                                                command_t::op_type_t::DELETE,
                                                                {0, 0},
                                                                {command_t::tid_node_id_t{.tid = 0, .node = 0}},
                                                                {})));

  // make a command that runs a reduce
  EXPECT_TRUE(rs->queue_command(command_t::create_unique(2,
                                                                command_t::op_type_t::REDUCE,
                                                                {0, 0},
                                                                {command_t::tid_node_id_t{.tid = 1, .node = 0},
                                                                 command_t::tid_node_id_t{.tid = 2, .node = 0}},
                                                                {command_t::tid_node_id_t{.tid = 3, .node = 0}})));

  // make a command that deletes all the tensors except for the tid = 3 tensor
  EXPECT_TRUE(rs->queue_command(command_t::create_unique(3,
                                                                command_t::op_type_t::DELETE,
                                                                {0, 0},
                                                                {command_t::tid_node_id_t{.tid = 1, .node = 0},
                                                                 command_t::tid_node_id_t{.tid = 2, .node = 0}},
                                                                {})));

  // get the first command to execute
  auto c1 = rs->get_next_command();

  // retire the command as we pretend we have executed it
  storage->create_tensor(2, 100);
  EXPECT_TRUE(rs->retire_command(std::move(c1)));

  // get the next command
  auto c2 = rs->get_next_command();
  storage->create_tensor(3, 100);
  EXPECT_TRUE(rs->retire_command(std::move(c2)));

  // shutdown the reservation station
  rs->shutdown();

  // wait for stuff to finish
  deleter.join();

  // make sure there is only one tensors
  EXPECT_EQ(storage->get_num_tensors(), 1);
}


TEST(TestReservationStation, FewLocalCommands2) {

  // tensors = { (0, 0), (1, 0) }
  // APPLY (.input = {(0, 0)}, .output = {(2, 0)})
  // REDUCE (.input = {(1, 0), (2, 0)}, .output = {(3, 0)})
  // DELETE (.input = {(0, 0), (1, 0), (2, 0), (3, 0)})

  // create the storage
  storage_ptr_t storage = std::make_shared<storage_t>();

  // create two input tensors
  storage->create_tensor(0, 100);
  storage->create_tensor(1, 100);

  // create the reservation station
  auto rs = std::make_shared<reservation_station_t>(0, 1);

  // create the deleter thread
  auto deleter = create_deleter_thread(rs, storage);

  // register the tensor
  rs->register_tensor(0);
  rs->register_tensor(1);

  // make a command that applies something to tensor 0
  EXPECT_TRUE(rs->queue_command(command_t::create_unique(0,
                                                                command_t::op_type_t::APPLY,
                                                                {0, 0},
                                                                {command_t::tid_node_id_t{.tid = 0, .node = 0}},
                                                                {command_t::tid_node_id_t{.tid = 2, .node = 0}})));

  // make a command that runs a reduce
  EXPECT_TRUE(rs->queue_command(command_t::create_unique(1,
                                                                command_t::op_type_t::REDUCE,
                                                                {0, 0},
                                                                {command_t::tid_node_id_t{.tid = 1, .node = 0},
                                                                 command_t::tid_node_id_t{.tid = 2, .node = 0}},
                                                                {command_t::tid_node_id_t{.tid = 3, .node = 0}})));


  // get the first command to execute
  auto c1 = rs->get_next_command();

  // retire the command as we pretend we have executed it
  storage->create_tensor(2, 100);
  EXPECT_TRUE(rs->retire_command(std::move(c1)));

  // get the next command 
  auto c2 = rs->get_next_command();
  storage->create_tensor(3, 100);
  EXPECT_TRUE(rs->retire_command(std::move(c2)));

  // make a command that deletes all the tensors except for the tid = 3 tensor
  EXPECT_TRUE(rs->queue_command(command_t::create_unique(2,
                                                        command_t::op_type_t::DELETE,
                                                        {0, 0},
                                                        {command_t::tid_node_id_t{.tid = 0, .node = 0},
                                                         command_t::tid_node_id_t{.tid = 1, .node = 0},
                                                         command_t::tid_node_id_t{.tid = 2, .node = 0},
                                                         command_t::tid_node_id_t{.tid = 3, .node = 0}},
                                                        {})));

  // shutdown the reservation station
  rs->shutdown();

  // wait for stuff to finish
  deleter.join();

  // make sure there is only one tensors
  EXPECT_EQ(storage->get_num_tensors(), 0);
}

TEST(TestReservationStation, TwoNodesBMM) {

  std::vector<reservation_station_ptr_t> rss;

  //       Tensors for A
  // | rowID | colID | tid | node |
  // |   0   |   0   |  0  |  1   |
  // |   0   |   1   |  1  |  0   |
  // |   1   |   0   |  2  |  0   |
  // |   1   |   1   |  3  |  1   |

  rss[1]->register_tensor(0);
  rss[0]->register_tensor(1);
  rss[0]->register_tensor(2);
  rss[1]->register_tensor(3);


  //       Tensors for B
  // | rowID | colID | tid | node |
  // |   0   |   0   |  4  |  1   |
  // |   0   |   1   |  5  |  1   |
  // |   1   |   0   |  6  |  0   |
  // |   1   |   1   |  7  |  0   |

  rss[1]->register_tensor(4);
  rss[1]->register_tensor(5);
  rss[0]->register_tensor(6);
  rss[0]->register_tensor(7);


  std::vector<command_ptr_t> _cmds;

  /// 1.1 shuffle A.rowID

  // MOVE (.input = {(0, 1)}, .output = {(0, 0)})
  _cmds.emplace_back(command_t::create_unique(0,
                                              command_t::op_type_t::MOVE,
                                              {-1, -1},
                                              {command_t::tid_node_id_t{.tid = 0, .node = 1}},
                                              {command_t::tid_node_id_t{.tid = 0, .node = 0}}));

  // MOVE (.input = {(0, 2)}, .output = {(1, 2)})
  _cmds.emplace_back(command_t::create_unique(1,
                                              command_t::op_type_t::MOVE,
                                              {-1, -1},
                                              {command_t::tid_node_id_t{.tid = 2, .node = 0}},
                                              {command_t::tid_node_id_t{.tid = 2, .node = 1}}));

  /// 1.2 broadcast B

  // MOVE (.input = {(4, 1)}, .output = {(4, 0)})
  _cmds.emplace_back(command_t::create_unique(2,
                                              command_t::op_type_t::MOVE,
                                              {-1, -1},
                                              {command_t::tid_node_id_t{.tid = 4, .node = 1}},
                                              {command_t::tid_node_id_t{.tid = 4, .node = 0}}));


  // MOVE (.input = {(5, 1)}, .output = {(5, 0)})
  _cmds.emplace_back(command_t::create_unique(3,
                                              command_t::op_type_t::MOVE,
                                              {-1, -1},
                                              {command_t::tid_node_id_t{.tid = 5, .node = 1}},
                                              {command_t::tid_node_id_t{.tid = 5, .node = 0}}));

  // MOVE (.input = {(6, 0)}, .output = {(6, 1)})
  _cmds.emplace_back(command_t::create_unique(4,
                                              command_t::op_type_t::MOVE,
                                              {-1, -1},
                                              {command_t::tid_node_id_t{.tid = 6, .node = 0}},
                                              {command_t::tid_node_id_t{.tid = 6, .node = 1}}));

  // MOVE (.input = {(7, 0)}, .output = {(7, 1)})
  _cmds.emplace_back(command_t::create_unique(5,
                                              command_t::op_type_t::MOVE,
                                              {-1, -1},
                                              {command_t::tid_node_id_t{.tid = 7, .node = 0}},
                                              {command_t::tid_node_id_t{.tid = 7, .node = 1}}));

  /// 2.1 Do the multiply

  // (0, 0) x (0, 0) - APPLY (.input = {(0, 0), (4, 0)}, .output = {(8, 0)})
  _cmds.emplace_back(command_t::create_unique(6,
                                              command_t::op_type_t::APPLY,
                                              {0, 0},
                                              {command_t::tid_node_id_t{.tid = 0, .node = 0},
                                                      command_t::tid_node_id_t{.tid = 4, .node = 0}},
                                              {command_t::tid_node_id_t{.tid = 8, .node = 0}}));


  // (0, 1) x (1, 0) - APPLY (.input = {(1, 0), (6, 0)}, .output = {(9, 0)})
  _cmds.emplace_back(command_t::create_unique(7,
                                              command_t::op_type_t::APPLY,
                                              {0, 0},
                                              {command_t::tid_node_id_t{.tid = 1, .node = 0},
                                                      command_t::tid_node_id_t{.tid = 6, .node = 0}},
                                              {command_t::tid_node_id_t{.tid = 9, .node = 0}}));

  // (1, 0) x (0, 0) - APPLY (.input = {(2, 1), (4, 1)}, .output = {(10, 1)})
  _cmds.emplace_back(command_t::create_unique(8,
                                              command_t::op_type_t::APPLY,
                                              {0, 0},
                                              {command_t::tid_node_id_t{.tid = 2, .node = 1},
                                                      command_t::tid_node_id_t{.tid = 4, .node = 1}},
                                              {command_t::tid_node_id_t{.tid = 10, .node = 1}}));

  // (1, 1) x (1, 0) - APPLY (.input = {(3, 1), (6, 1)}, .output = {(11, 1)})
  _cmds.emplace_back(command_t::create_unique(9,
                                              command_t::op_type_t::APPLY,
                                              {0, 0},
                                              {command_t::tid_node_id_t{.tid = 3, .node = 1},
                                                      command_t::tid_node_id_t{.tid = 6, .node = 1}},
                                              {command_t::tid_node_id_t{.tid = 11, .node = 1}}));

  // (0, 0) x (0, 1) - APPLY (.input = {(0, 0), (5, 0)}, .output = {(12, 0)})
  _cmds.emplace_back(command_t::create_unique(10,
                                              command_t::op_type_t::APPLY,
                                              {0, 0},
                                              {command_t::tid_node_id_t{.tid = 0, .node = 0},
                                                      command_t::tid_node_id_t{.tid = 5, .node = 0}},
                                              {command_t::tid_node_id_t{.tid = 12, .node = 0}}));

  // (0, 1) x (1, 1) - APPLY (.input = {(1, 0), (7, 0)}, .output = {(13, 0)})
  _cmds.emplace_back(command_t::create_unique(11,
                                              command_t::op_type_t::APPLY,
                                              {0, 0},
                                              {command_t::tid_node_id_t{.tid = 1, .node = 0},
                                                      command_t::tid_node_id_t{.tid = 7, .node = 0}},
                                              {command_t::tid_node_id_t{.tid = 13, .node = 0}}));

  // (1, 0) x (0, 1) - APPLY (.input = {(2, 1), (5, 1)}, .output = {(14, 1)})
  _cmds.emplace_back(command_t::create_unique(12,
                                              command_t::op_type_t::APPLY,
                                              {0, 0},
                                              {command_t::tid_node_id_t{.tid = 2, .node = 1},
                                                      command_t::tid_node_id_t{.tid = 5, .node = 1}},
                                              {command_t::tid_node_id_t{.tid = 14, .node = 1}}));

  // (1, 1) x (1, 1) - APPLY (.input = {(3, 1), (7, 1)}, .output = {(15, 1)})
  _cmds.emplace_back(command_t::create_unique(13,
                                              command_t::op_type_t::APPLY,
                                              {0, 0},
                                              {command_t::tid_node_id_t{.tid = 3, .node = 1},
                                                      command_t::tid_node_id_t{.tid = 7, .node = 1}},
                                              {command_t::tid_node_id_t{.tid = 15, .node = 1}}));

  /// 2.2 Do the reduce


  // (0, 0) x (0, 0) + (0, 1) x (1, 0) - REDUCE (.input = {(8, 0), (9, 0)}, .output = {(16, 0)})
  _cmds.emplace_back(command_t::create_unique(14,
                                              command_t::op_type_t::REDUCE,
                                              {0, 0},
                                              {command_t::tid_node_id_t{.tid = 8, .node = 0},
                                                      command_t::tid_node_id_t{.tid = 9, .node = 0}},
                                              {command_t::tid_node_id_t{.tid = 16, .node = 0}}));

  // (1, 0) x (0, 0) + (1, 1) x (1, 0) - REDUCE (.input = {(10, 1), (11, 1)}, .output = {(17, 1)})
  _cmds.emplace_back(command_t::create_unique(15,
                                              command_t::op_type_t::REDUCE,
                                              {0, 0},
                                              {command_t::tid_node_id_t{.tid = 10, .node = 1},
                                                      command_t::tid_node_id_t{.tid = 11, .node = 1}},
                                              {command_t::tid_node_id_t{.tid = 17, .node = 1}}));

  // (0, 0) x (0, 0) + (0, 1) x (1, 0) - REDUCE (.input = {(12, 0), (13, 0)}, .output = {(18, 0)})
  _cmds.emplace_back(command_t::create_unique(16,
                                              command_t::op_type_t::REDUCE,
                                              {0, 0},
                                              {command_t::tid_node_id_t{.tid = 12, .node = 0},
                                                      command_t::tid_node_id_t{.tid = 13, .node = 0}},
                                              {command_t::tid_node_id_t{.tid = 18, .node = 0}}));

  // (0, 0) x (0, 0) + (0, 1) x (1, 0) - REDUCE (.input = {(14, 1), (15, 1)}, .output = {(19, 1)})
  _cmds.emplace_back(command_t::create_unique(17,
                                              command_t::op_type_t::REDUCE,
                                              {0, 0},
                                              {command_t::tid_node_id_t{.tid = 14, .node = 1},
                                                      command_t::tid_node_id_t{.tid = 15, .node = 1}},
                                              {command_t::tid_node_id_t{.tid = 19, .node = 1}}));


  /// Remove the intermediate results
}

}