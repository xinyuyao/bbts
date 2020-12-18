#include <gtest/gtest.h>
#include <thread>
#include <atomic>
#include "../src/commands/reservation_station.h"
#include "../src/utils/concurent_queue.h"

namespace bbts {

// the reservation station needs a deleter thread
std::thread create_deleter_thread(reservation_station_ptr_t &_rs, std::unordered_set<tid_t> &_sto) {

  // create the thread
  return std::thread([_rs, &_sto]() {

    // while we have something remove
    tid_t id;
    while((id = _rs->get_to_remove()) != -1) {
      _sto.erase(id);
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
  std::unordered_set<tid_t> storage;

  // create two input tensors
  storage.insert(0);
  storage.insert(1);

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
  storage.insert(2);
  EXPECT_TRUE(rs->retire_command(std::move(c1)));

  // get the next command
  auto c2 = rs->get_next_command();
  storage.insert(3);
  EXPECT_TRUE(rs->retire_command(std::move(c2)));

  // shutdown the reservation station
  rs->shutdown();

  // wait for stuff to finish
  deleter.join();

  // make sure there is only one tensors
  EXPECT_EQ(storage.size(), 1);
}


TEST(TestReservationStation, FewLocalCommands2) {

  // tensors = { (0, 0), (1, 0) }
  // APPLY (.input = {(0, 0)}, .output = {(2, 0)})
  // REDUCE (.input = {(1, 0), (2, 0)}, .output = {(3, 0)})
  // DELETE (.input = {(0, 0), (1, 0), (2, 0), (3, 0)})

  // create the storage
  std::unordered_set<tid_t> storage;

  // create two input tensors
  storage.insert(0);
  storage.insert(1);

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
  storage.insert(2);
  EXPECT_TRUE(rs->retire_command(std::move(c1)));

  // get the next command 
  auto c2 = rs->get_next_command();
  storage.insert(3);
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
  EXPECT_EQ(storage.size(), 0);
}


TEST(TestReservationStation, TwoNodesBMM) {

  // create the storage
  std::vector<std::unordered_set<tid_t>> sto(2);

  // create the two reservation stations
  std::vector<reservation_station_ptr_t> rss;
  rss.push_back(std::make_shared<reservation_station_t>(0, 2));
  rss.push_back(std::make_shared<reservation_station_t>(1, 2));

  //       Tensors for A
  // | rowID | colID | tid | node |
  // |   0   |   0   |  0  |  1   |
  // |   0   |   1   |  1  |  0   |
  // |   1   |   0   |  2  |  0   |
  // |   1   |   1   |  3  |  1   |

  // (0, 0)
  rss[1]->register_tensor(0);
  sto[1].insert(0);

  // (0, 1)
  rss[0]->register_tensor(1);
  sto[0].insert(1);

  // (1, 0)
  rss[0]->register_tensor(2);
  sto[0].insert(2);

  // (1, 1)
  rss[1]->register_tensor(3);
  sto[1].insert(3);

  //       Tensors for B
  // | rowID | colID | tid | node |
  // |   0   |   0   |  4  |  1   |
  // |   0   |   1   |  5  |  1   |
  // |   1   |   0   |  6  |  0   |
  // |   1   |   1   |  7  |  0   |

  // (0, 0)
  rss[1]->register_tensor(4);
  sto[1].insert(4);

  // (1, 0)
  rss[1]->register_tensor(5);
  sto[1].insert(5);

  // (0, 1)
  rss[0]->register_tensor(6);
  sto[0].insert(6);

  // (1, 1)
  rss[0]->register_tensor(7);
  sto[0].insert(7);

  // we put the commands we want to schedule here
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

  // remove them from node 0
  _cmds.emplace_back(command_t::create_unique(18,
                                            command_t::op_type_t::DELETE,
                                            {0, 0},
                                            {command_t::tid_node_id_t{.tid = 8, .node = 0},
                                                    command_t::tid_node_id_t{.tid = 9, .node = 0},
                                                    command_t::tid_node_id_t{.tid = 12, .node = 0},
                                                    command_t::tid_node_id_t{.tid = 13, .node = 0},
                                                    command_t::tid_node_id_t{.tid = 0, .node = 0},
                                                    command_t::tid_node_id_t{.tid = 4, .node = 0},
                                                    command_t::tid_node_id_t{.tid = 5, .node = 0}},
                                                   {}));

  // remove them from node 1
  _cmds.emplace_back(command_t::create_unique(19,
                                              command_t::op_type_t::DELETE,
                                              {0, 0},
                                              {command_t::tid_node_id_t{.tid = 10, .node = 1},
                                                      command_t::tid_node_id_t{.tid = 11, .node = 1},
                                                      command_t::tid_node_id_t{.tid = 14, .node = 1},
                                                      command_t::tid_node_id_t{.tid = 15, .node = 1},
                                                      command_t::tid_node_id_t{.tid = 2, .node = 1},
                                                      command_t::tid_node_id_t{.tid = 6, .node = 1},
                                                      command_t::tid_node_id_t{.tid = 7, .node = 1}},
                                                     {}));

  // schedule them all at once
  for(auto &cmd : _cmds) {

    // if it uses node 0
    if(cmd->uses_node(0)) {
      EXPECT_TRUE(rss[0]->queue_command(cmd->clone()));
    }

    // check if node 1 uses the command
    if(cmd->uses_node(1)) {
      EXPECT_TRUE(rss[1]->queue_command(cmd->clone()));
    }
  }

  // create the queues for commands
  using _remote_cmd_t = std::tuple<command_ptr_t, std::atomic<bool>*>;
  std::vector<bbts::concurent_queue<_remote_cmd_t>> remote_cmds(2);

  // these threads will process the remote move operations
  std::vector<std::thread> _remote_executor_threads;
  for(node_id_t node = 0; node < 2; ++node) {

    // create the thread to pull
    std::thread t = std::thread([&remote_cmds, &_remote_executor_threads, node, &rss, &sto]() {

      // process the remote commands
      while (true) {

        // wait for the command
        _remote_cmd_t _rc;
        remote_cmds[node].wait_dequeue(_rc);

        // check if we are done...
        if(std::get<0>(_rc) == nullptr) {
          break;
        }

        // simulate the execution
        usleep(rand() % 100);

        // we only have remote moves here
        EXPECT_EQ(std::get<0>(_rc)->type, command_t::MOVE);
        if(std::get<0>(_rc)->type == command_t::MOVE) {

          // get the output tensor
          auto [out_tid, out_node] = std::get<0>(_rc)->get_output(0);

          // make sure it is the same node
          EXPECT_EQ(node, out_node);
          std::cout << "MOVE\n" << std::flush;

          // mark that the tensor is
          sto[node].insert(out_tid);

          // register the tensor with the reservation station
          rss[node]->register_tensor(out_tid);
        }

        // retire the command
        rss[node]->retire_command(std::move(std::get<0>(_rc)));

        // mark that we are done
        *std::get<1>(_rc) = true;
      }


    });

    // store the thread
    _remote_executor_threads.push_back(std::move(t));
  }

  // simulator threads
  std::vector<std::thread> _executor_threads;
  for(node_id_t node = 0; node < 2; ++node) {

    // create the thread to pull
    std::thread t = std::thread([&rss, &remote_cmds, node]() {

      while (true) {

        // get the command
        auto cmd = rss[node]->get_next_command();

        // log the command
        std::cout << "command " << cmd->id << " on node " << node << '\n' << std::flush;

        if(cmd == nullptr) {
          break;
        }

        // if we have a move
        if(cmd->type == command_t::MOVE) {

          // get the target node
          auto target = cmd->get_output(0);

          // we use this to busy wait
          std::atomic<bool> done{};
          done = false;

          // push the command
          _remote_cmd_t c = { cmd->clone(), &done };
          remote_cmds[target.node].enqueue(c);

          // add some waiting to simulate running the command
          usleep(rand() % 100);

          // wait for the remote command to be done
          while(!done) {}
        }
        else if(cmd->type == command_t::APPLY) {

        }
        else if(cmd->type == command_t::DELETE) {

        }
        else if(cmd->type == command_t::REDUCE) {

        }

      }

    });

    _executor_threads.push_back(std::move(t));
  }

  // create the deleters
  std::vector<std::thread> deleters;
  deleters.push_back(std::move(create_deleter_thread(rss[0], sto[0])));
  deleters.push_back(std::move(create_deleter_thread(rss[1], sto[1])));

  // shutdown the rss
  //rss[0]->shutdown();
  //rss[1]->shutdown();

  // wait for the deleters
  deleters[0].join();
  deleters[1].join();
}

}