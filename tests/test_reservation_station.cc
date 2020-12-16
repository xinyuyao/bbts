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
  _cmds.emplace_back(command_t::create_unique(0,
                                              command_t::op_type_t::MOVE,
                                              {-1, -1},
                                              {command_t::tid_node_id_t{.tid = 2, .node = 0}},
                                              {command_t::tid_node_id_t{.tid = 2, .node = 1}}));

  /// 1.2 broadcast B



  // (0, 0) x (0, 0) or as tids {0}
  // (0, 1) x (1, 0)


  // (0, 0) x (0, 0)
  // (0, 1) x (1, 0)

  // (0, 0) x (0, 0)
  // (0, 1) x (1, 0)

  // (0, 0) x (0, 0)
  // (0, 1) x (1, 0)


}

}