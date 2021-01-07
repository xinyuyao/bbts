#include <gtest/gtest.h>
#include <thread>
#include <atomic>
#include "../src/commands/reservation_station.h"
#include "../src/utils/concurent_queue.h"

namespace bbts {

using _remote_cmd_t = std::tuple<command_ptr_t, std::atomic<bool>*>;

// the reservation station needs a deleter thread
std::thread create_deleter_thread(reservation_station_ptr_t &_rs, std::unordered_set<tid_t> &_sto, int32_t _num_to_del) {

  // create the thread

  return std::thread([_rs, &_sto, _num_to_del]() {

    // while we have something remove
    tid_t id;
    auto to_deleted = _num_to_del;
    while(true) {

      // check if done
      if(to_deleted == 0) {
        break;
      }

      // get the next tensor to remove
      id = _rs->get_to_remove();
      if(id == -1) {
        break;
      }

      // do some random sleeping
      usleep(rand() % 1000 + 100);

      // deleted
      _sto.erase(id);
      std::cout << "Remove tensor : " <<  id << '\n' << std::flush;
      to_deleted--;
    }
  });
}

std::thread create_remote_processing_thread(int32_t node,
                                            std::vector<reservation_station_ptr_t> &rss,
                                            std::vector<std::unordered_set<tid_t>> &sto,
                                            std::vector<bbts::concurent_queue<_remote_cmd_t>> &remote_cmds) {

  // create the thread to pull
  std::thread t = std::thread([&remote_cmds, node, &rss, &sto]() {

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
      if(std::get<0>(_rc)->type == command_t::MOVE) {

        // get the output tensor
        auto [out_tid, out_node] = std::get<0>(_rc)->get_output(0);

        // make sure it is the same node
        EXPECT_EQ(node, out_node);
        std::cout << "MOVE from " << out_node << "\n" << std::flush;

        // mark that the tensor is
        sto[node].insert(out_tid);
      }
      else if(std::get<0>(_rc)->type == command_t::REDUCE) {

        std::cout << "REDUCE from \n" << std::flush;
      }

      // retire the command
      rss[node]->retire_command(std::move(std::get<0>(_rc)));

      // mark that we are done
      *std::get<1>(_rc) = true;
    }

  });

  return std::move(t);
}

std::thread remote_tensor_notification_sender(int32_t my_node,
                                              int32_t out_node,
                                              std::vector<reservation_station_ptr_t> &rss,
                                              std::vector<bbts::concurent_queue<std::pair<std::vector<tid_t>, int32_t>>> &remote_notifications) {
  // create the thread
  std::thread t = std::thread([my_node, out_node, &rss, &remote_notifications]() {

    while (true) {

      // get tensors to notify the other node
      bool is_done;
      auto tensors = rss[my_node]->tensors_to_notify_node(out_node, is_done);

      // if it is node break out
      if (is_done) {
        break;
      }

      // add the remote commands
      std::pair<std::vector<tid_t>, int32_t> to_store = {tensors, my_node};
      remote_notifications[out_node].enqueue(to_store);
    }

  });

  return std::move(t);
}

std::thread tensor_notifier(int32_t my_node,
                            std::vector<reservation_station_ptr_t> &rss,
                            std::vector<bbts::concurent_queue<std::pair<std::vector<tid_t>, int32_t>>> &remote_notifications) {
  // create the thread
  std::thread t = std::thread([my_node, &rss, &remote_notifications]() {

    while (true) {

      // wait for the command
      std::pair<std::vector<tid_t>, int32_t> tensors;
      remote_notifications[my_node].wait_dequeue(tensors);

      // check if we are done...
      if (std::get<1>(tensors) == -1) {
        break;
      }

      // notify that the tensors became available
      rss[my_node]->notify_available_tensors(tensors.second, tensors.first);
    }
  });

  return std::move(t);
}

std::thread create_command_processing_thread(std::vector<reservation_station_ptr_t> &rss,
                                             std::vector<std::unordered_set<tid_t>> &sto,
                                             std::vector<bbts::concurent_queue<_remote_cmd_t>> &remote_cmds,
                                             int32_t node) {

  // create the thread to pull
  std::thread t = std::thread([&rss, &remote_cmds, &sto, node]() {

    while (true) {

      // get the command
      auto cmd = rss[node]->get_next_command();
      if(cmd == nullptr) {
        break;
      }

      // if we have a move
      if(cmd->type == command_t::MOVE) {

        // move the
        std::cout << "MOVE " << cmd->id << " on my_node : " << node << " Executed...\n" << std::flush;

        // get the target node
        auto target = cmd->get_output(0);

        // we use this to busy wait
        std::atomic<bool> done{};
        done = false;

        // push the command
        _remote_cmd_t c = {cmd->clone(), &done };
        remote_cmds[target.node].enqueue(c);

        // add some waiting to simulate running the command
        usleep(rand() % 100);

        // retire the command
        rss[node]->retire_command(std::move(cmd));

        // wait for the remote command to be done
        while(!done) {}
      }
      else if(cmd->type == command_t::APPLY) {

        std::cout << "APPLY " << cmd->id << " on my_node : " << node << " Executed...\n" << std::flush;

        if(cmd->get_root_node() == node) {

          // store the outputs so we can add it to the storage
          auto outputs = cmd->get_outputs();

          // retire the command
          rss[node]->retire_command(std::move(cmd));

          // update all the outputs in the storage
          for(auto &o : outputs) {
            sto[node].insert(o.tid);
          }
        }
        else {

          // all applies are local
          throw std::runtime_error("How did this happen!");
        }
      }
      else if(cmd->type == command_t::DELETE) {

        // this should never happen
        throw std::runtime_error("We should never get a delete to execute, delete is implicit...");
      }
      else if(cmd->type == command_t::REDUCE) {

        // check if the reduce is remote or local
        if(cmd->is_local_reduce(node)) {

          std::cout << "LOCAL_REDUCE " << cmd->id << " on node " << node << '\n' << std::flush;

          // store the outputs so we can add it to the storage
          auto outputs = cmd->get_outputs();

          // retire the command
          rss[node]->retire_command(std::move(cmd));

          // update all the outputs in the storage
          for(auto &o : outputs) {
            sto[node].insert(o.tid);
          }
        }
        else {

          std::cout << "REMOTE_REDUCE " << cmd->id << " on node " << node << '\n' << std::flush;

          // store the outputs so we can add it to the storage
          auto outputs = cmd->get_outputs();

          // update all the outputs in the storage
          for(auto &o : outputs) {
            sto[node].insert(o.tid);
          }

          // figure out the nodes we need to retire the command to
          std::unordered_set<node_id_t> nodes_to_retire;
          for(auto &in : cmd->get_inputs()) {
            nodes_to_retire.insert(in.node);
          }
          for(auto &out : cmd->get_outputs()) {
            nodes_to_retire.insert(out.node);
          }

          // we use this to busy wait
          std::vector<std::atomic<bool>> done(nodes_to_retire.size());
          for(auto &d : done) { d = false; }

          // retire the command for each node
          int32_t i = 0;
          for(auto n : nodes_to_retire) {

            if(n == node) {
              done[i++] = true;
              continue;
            }

            // push the command
            _remote_cmd_t c = {cmd->clone(), &done[i++] };
            remote_cmds[n].enqueue(c);
          }

          // add some waiting to simulate running the command
          usleep(rand() % 100);

          // retire the command
          rss[node]->retire_command(std::move(cmd));

          // wait till all done
          for(auto &d : done) { while(!d) {} }
        }
      }
    }

  });

  return std::move(t);
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
  auto deleter = create_deleter_thread(rs, storage, 2);

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
  auto deleter = create_deleter_thread(rs, storage, 2);

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
                                            {command_t::tid_node_id_t{.tid = 1, .node = 0},
                                                    command_t::tid_node_id_t{.tid = 2, .node = 0},
                                                    command_t::tid_node_id_t{.tid = 6, .node = 0},
                                                    command_t::tid_node_id_t{.tid = 7, .node = 0},
                                                    command_t::tid_node_id_t{.tid = 0, .node = 0},
                                                    command_t::tid_node_id_t{.tid = 4, .node = 0},
                                                    command_t::tid_node_id_t{.tid = 5, .node = 0},
                                                    command_t::tid_node_id_t{.tid = 8, .node = 0},
                                                    command_t::tid_node_id_t{.tid = 9, .node = 0},
                                                    command_t::tid_node_id_t{.tid = 12, .node = 0},
                                                    command_t::tid_node_id_t{.tid = 13, .node = 0}},
                                                   {}));

  // remove them from node 1
  _cmds.emplace_back(command_t::create_unique(19,
                                              command_t::op_type_t::DELETE,
                                              {0, 0},
                                              {command_t::tid_node_id_t{.tid = 0, .node = 1},
                                                      command_t::tid_node_id_t{.tid = 3, .node = 1},
                                                      command_t::tid_node_id_t{.tid = 4, .node = 1},
                                                      command_t::tid_node_id_t{.tid = 5, .node = 1},
                                                      command_t::tid_node_id_t{.tid = 2, .node = 1},
                                                      command_t::tid_node_id_t{.tid = 6, .node = 1},
                                                      command_t::tid_node_id_t{.tid = 7, .node = 1},
                                                      command_t::tid_node_id_t{.tid = 10, .node = 1},
                                                      command_t::tid_node_id_t{.tid = 11, .node = 1},
                                                      command_t::tid_node_id_t{.tid = 14, .node = 1},
                                                      command_t::tid_node_id_t{.tid = 15, .node = 1}},
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
  std::vector<bbts::concurent_queue<_remote_cmd_t>> remote_cmds(2);

  // these threads will process the remote move operations
  std::vector<std::thread> _remote_executor_threads;
  _remote_executor_threads.reserve(2);
  for(node_id_t node = 0; node < 2; ++node) {

    // store the thread
    _remote_executor_threads.push_back(std::move(create_remote_processing_thread(node, rss, sto, remote_cmds)));
  }

  // simulator threads
  std::vector<std::thread> _executor_threads;
  _executor_threads.reserve(2);
  for(node_id_t node = 0; node < 2; ++node) {

    _executor_threads.push_back(std::move(create_command_processing_thread(rss, sto, remote_cmds, node)));
  }

  // create the deleters
  std::vector<std::thread> deleters;
  deleters.push_back(std::move(create_deleter_thread(rss[0], sto[0], 11)));
  deleters.push_back(std::move(create_deleter_thread(rss[1], sto[1], 11)));

  // wait for the deleters
  deleters[0].join();
  deleters[1].join();

  // shutdown the rss
  rss[0]->shutdown();
  rss[1]->shutdown();

  _remote_cmd_t c = {nullptr, nullptr};
  remote_cmds[0].enqueue(c);
  remote_cmds[1].enqueue(c);

  // wait for remote executors to finish
  for(auto &t : _remote_executor_threads) {
    t.join();
  }

  // wait for the executors
  for(auto &t : _executor_threads) {
    t.join();
  }

  // make sure there are exactly two for the reduce...
  EXPECT_EQ(sto[0].size(), 2);
  EXPECT_EQ(sto[1].size(), 2);
}

TEST(TestReservationStation, TwoNodesCMM) {

  // create the storage
  std::vector<std::unordered_set<tid_t>> sto(2);

  std::vector<bbts::concurent_queue<std::pair<std::vector<tid_t>, int32_t>>> remote_notifications(2);

  // create the two reservation stations
  std::vector<reservation_station_ptr_t> rss;
  rss.push_back(std::make_shared<reservation_station_t>(0, 2));
  rss.push_back(std::make_shared<reservation_station_t>(1, 2));

  //       Tensors for A
  // | rowID | colID | tid | node |
  // |   0   |   0   |  0  |  1   | MOVE TO NODE 0
  // |   0   |   1   |  1  |  0   | MOVE TO NODE 1
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
  // |   0   |   0   |  4  |  1   | MOVE TO 0
  // |   0   |   1   |  5  |  0   |
  // |   1   |   0   |  6  |  1   |
  // |   1   |   1   |  7  |  0   | MOVE TO 1

  // (0, 0)
  rss[1]->register_tensor(4);
  sto[1].insert(4);

  // (1, 0)
  rss[0]->register_tensor(5);
  sto[0].insert(5);

  // (0, 1)
  rss[1]->register_tensor(6);
  sto[1].insert(6);

  // (1, 1)
  rss[0]->register_tensor(7);
  sto[0].insert(7);

  // we put the commands we want to schedule here
  std::vector<command_ptr_t> _cmds;


  /// 1.1 shuffle A.colID

  // MOVE (.input = {( 0,  1)}, .output = {( 0,  0)})
  _cmds.emplace_back(command_t::create_unique(0,
                                              command_t::op_type_t::MOVE,
                                              {-1, -1},
                                              {command_t::tid_node_id_t{.tid = 0, .node = 1}},
                                              {command_t::tid_node_id_t{.tid = 0, .node = 0}}));

  // MOVE (.input = {(1,  0)}, .output = {(1,  1)})
  _cmds.emplace_back(command_t::create_unique(1,
                                              command_t::op_type_t::MOVE,
                                              {-1, -1},
                                              {command_t::tid_node_id_t{.tid = 1, .node = 0}},
                                              {command_t::tid_node_id_t{.tid = 1, .node = 1}}));

  /// 1.2 shuffle B.rowID

  // MOVE (.input = {(4, 1)}, .output = {(4, 0)})
  _cmds.emplace_back(command_t::create_unique(2,
                                              command_t::op_type_t::MOVE,
                                              {-1, -1},
                                              {command_t::tid_node_id_t{.tid = 4, .node = 1}},
                                              {command_t::tid_node_id_t{.tid = 4, .node = 0}}));

  // MOVE (.input = {(7, 0)}, .output = {(7, 1)})
  _cmds.emplace_back(command_t::create_unique(3,
                                              command_t::op_type_t::MOVE,
                                              {-1, -1},
                                              {command_t::tid_node_id_t{.tid = 7, .node = 0}},
                                              {command_t::tid_node_id_t{.tid = 7, .node = 1}}));


  /// 2.1 Do the multiply

  // (0, 0) x (0, 0) - APPLY (.input = {(0, 0), (4, 0)}, .output = {(8, 0)})
  _cmds.emplace_back(command_t::create_unique(4,
                                              command_t::op_type_t::APPLY,
                                              {0, 0},
                                              {command_t::tid_node_id_t{.tid = 0, .node = 0},
                                                      command_t::tid_node_id_t{.tid = 4, .node = 0}},
                                              {command_t::tid_node_id_t{.tid = 8, .node = 0}}));


  // (0, 1) x (1, 0) - APPLY (.input = {(1, 1), (6, 1)}, .output = {(9, 1)})
  _cmds.emplace_back(command_t::create_unique(5,
                                              command_t::op_type_t::APPLY,
                                              {0, 0},
                                              {command_t::tid_node_id_t{.tid = 1, .node = 1},
                                                      command_t::tid_node_id_t{.tid = 6, .node = 1}},
                                              {command_t::tid_node_id_t{.tid = 9, .node = 1}}));

  // (1, 0) x (0, 0) - APPLY (.input = {(2, 0), (4, 0)}, .output = {(10, 0)})
  _cmds.emplace_back(command_t::create_unique(6,
                                              command_t::op_type_t::APPLY,
                                              {0, 0},
                                              {command_t::tid_node_id_t{.tid = 2, .node = 0},
                                                      command_t::tid_node_id_t{.tid = 4, .node = 0}},
                                              {command_t::tid_node_id_t{.tid = 10, .node = 0}}));

  // (1, 1) x (1, 0) - APPLY (.input = {(3, 1), (6, 1)}, .output = {(11, 1)})
  _cmds.emplace_back(command_t::create_unique(7,
                                              command_t::op_type_t::APPLY,
                                              {0, 0},
                                              {command_t::tid_node_id_t{.tid = 3, .node = 1},
                                                      command_t::tid_node_id_t{.tid = 6, .node = 1}},
                                              {command_t::tid_node_id_t{.tid = 11, .node = 1}}));

  // (0, 0) x (0, 1) - APPLY (.input = {(0, 0), (5, 0)}, .output = {(12, 0)})
  _cmds.emplace_back(command_t::create_unique(8,
                                              command_t::op_type_t::APPLY,
                                              {0, 0},
                                              {command_t::tid_node_id_t{.tid = 0, .node = 0},
                                                      command_t::tid_node_id_t{.tid = 5, .node = 0}},
                                              {command_t::tid_node_id_t{.tid = 12, .node = 0}}));

  // (0, 1) x (1, 1) - APPLY (.input = {(1, 0), (7, 0)}, .output = {(13, 0)})
  _cmds.emplace_back(command_t::create_unique(9,
                                              command_t::op_type_t::APPLY,
                                              {0, 0},
                                              {command_t::tid_node_id_t{.tid = 1, .node = 1},
                                                      command_t::tid_node_id_t{.tid = 7, .node = 1}},
                                              {command_t::tid_node_id_t{.tid = 13, .node = 1}}));

  // (1, 0) x (0, 1) - APPLY (.input = {(2, 1), (5, 1)}, .output = {(14, 1)})
  _cmds.emplace_back(command_t::create_unique(10,
                                              command_t::op_type_t::APPLY,
                                              {0, 0},
                                              {command_t::tid_node_id_t{.tid = 2, .node = 0},
                                                      command_t::tid_node_id_t{.tid = 5, .node = 0}},
                                              {command_t::tid_node_id_t{.tid = 14, .node = 0}}));

  // (1, 1) x (1, 1) - APPLY (.input = {(3, 1), (7, 1)}, .output = {(15, 1)})
  _cmds.emplace_back(command_t::create_unique(11,
                                              command_t::op_type_t::APPLY,
                                              {0, 0},
                                              {command_t::tid_node_id_t{.tid = 3, .node = 1},
                                                      command_t::tid_node_id_t{.tid = 7, .node = 1}},
                                              {command_t::tid_node_id_t{.tid = 15, .node = 1}}));

  /// 2.2 Do the reduce

  // (0, 0) x (0, 0) + (0, 1) x (1, 0) - REDUCE (.input = {(8, 0), (9, 1)}, .output = {(16, 0)})
  _cmds.emplace_back(command_t::create_unique(12,
                                              command_t::op_type_t::REDUCE,
                                              {0, 0},
                                              {command_t::tid_node_id_t{.tid = 8, .node = 0},
                                                      command_t::tid_node_id_t{.tid = 9, .node = 1}},
                                              {command_t::tid_node_id_t{.tid = 16, .node = 0}}));

  // (1, 0) x (0, 0) + (1, 1) x (1, 0) - REDUCE (.input = {(10, 0), (11, 1)}, .output = {(17, 1)})
  _cmds.emplace_back(command_t::create_unique(13,
                                              command_t::op_type_t::REDUCE,
                                              {0, 0},
                                              {command_t::tid_node_id_t{.tid = 10, .node = 0},
                                                      command_t::tid_node_id_t{.tid = 11, .node = 1}},
                                              {command_t::tid_node_id_t{.tid = 17, .node = 1}}));

  // (0, 0) x (0, 0) + (0, 1) x (1, 0) - REDUCE (.input = {(12, 0), (13, 0)}, .output = {(18, 0)})
  _cmds.emplace_back(command_t::create_unique(14,
                                              command_t::op_type_t::REDUCE,
                                              {0, 0},
                                              {command_t::tid_node_id_t{.tid = 12, .node = 0},
                                                      command_t::tid_node_id_t{.tid = 13, .node = 1}},
                                              {command_t::tid_node_id_t{.tid = 18, .node = 0}}));

  // (0, 0) x (0, 0) + (0, 1) x (1, 0) - REDUCE (.input = {(14, 1), (15, 1)}, .output = {(19, 1)})
  _cmds.emplace_back(command_t::create_unique(15,
                                              command_t::op_type_t::REDUCE,
                                              {0, 0},
                                              {command_t::tid_node_id_t{.tid = 14, .node = 0},
                                                      command_t::tid_node_id_t{.tid = 15, .node = 1}},
                                              {command_t::tid_node_id_t{.tid = 19, .node = 1}}));

  /// 3.0 Remove the intermediate results

  // remove them from node 0
  _cmds.emplace_back(command_t::create_unique(16,
                                              command_t::op_type_t::DELETE,
                                              {0, 0},
                                              { command_t::tid_node_id_t{.tid = 1, .node = 0},
                                                       command_t::tid_node_id_t{.tid = 2, .node = 0},
                                                       command_t::tid_node_id_t{.tid = 5, .node = 0},
                                                       command_t::tid_node_id_t{.tid = 7, .node = 0},
                                                       command_t::tid_node_id_t{.tid = 0, .node = 0},
                                                       command_t::tid_node_id_t{.tid = 4, .node = 0},
                                                       command_t::tid_node_id_t{.tid = 8, .node = 0},
                                                       command_t::tid_node_id_t{.tid = 10, .node = 0},
                                                       command_t::tid_node_id_t{.tid = 12, .node = 0},
                                                       command_t::tid_node_id_t{.tid = 14, .node = 0},
                                                       command_t::tid_node_id_t{.tid = 16, .node = 0},
                                                       command_t::tid_node_id_t{.tid = 18, .node = 0}}, {}));

  // remove them from node 0
  _cmds.emplace_back(command_t::create_unique(17,
                                              command_t::op_type_t::DELETE,
                                              {0, 0},
                                              {  command_t::tid_node_id_t{.tid = 0, .node = 1},
                                                        command_t::tid_node_id_t{.tid = 3, .node = 1},
                                                        command_t::tid_node_id_t{.tid = 4, .node = 1},
                                                        command_t::tid_node_id_t{.tid = 6, .node = 1},
                                                        command_t::tid_node_id_t{.tid = 1, .node = 1},
                                                        command_t::tid_node_id_t{.tid = 7, .node = 1},
                                                        command_t::tid_node_id_t{.tid = 9, .node = 1},
                                                        command_t::tid_node_id_t{.tid = 11, .node = 1},
                                                        command_t::tid_node_id_t{.tid = 13, .node = 1},
                                                        command_t::tid_node_id_t{.tid = 15, .node = 1},
                                                        command_t::tid_node_id_t{.tid = 17, .node = 1},
                                                        command_t::tid_node_id_t{.tid = 19, .node = 1}}, {}));

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
  std::vector<bbts::concurent_queue<_remote_cmd_t>> remote_cmds(2);

  // these threads will process the remote move operations
  std::vector<std::thread> _remote_executor_threads;
  _remote_executor_threads.reserve(2);
  for(node_id_t node = 0; node < 2; ++node) {

    // store the thread
    _remote_executor_threads.push_back(std::move(create_remote_processing_thread(node, rss, sto, remote_cmds)));
  }

  // simulator threads
  std::vector<std::thread> _executor_threads;
  _executor_threads.reserve(2);
  for(node_id_t node = 0; node < 2; ++node) {

    _executor_threads.push_back(std::move(create_command_processing_thread(rss, sto, remote_cmds, node)));
  }

  //
  std::vector<std::thread> _notifier_threads;
  _notifier_threads.reserve(2);
  for(node_id_t node = 0; node < 2; ++node) {
    for(node_id_t out_node = 0; out_node < 2; ++out_node) {
      _notifier_threads.push_back(std::move(remote_tensor_notification_sender(node,
                                                                              out_node,
                                                                              rss,
                                                                              remote_notifications)));
    }
  }

  //
  std::vector<std::thread> _notification_handler_threads;
  _notification_handler_threads.reserve(2);
  for(node_id_t node = 0; node < 2; ++node) {
    _notification_handler_threads.push_back(std::move(tensor_notifier(node, rss, remote_notifications)));
  }

  // create the deleters
  std::vector<std::thread> deleters;
  deleters.push_back(std::move(create_deleter_thread(rss[0], sto[0], 12)));
  deleters.push_back(std::move(create_deleter_thread(rss[1], sto[1], 12)));

  // wait for the deleters
  deleters[0].join();
  deleters[1].join();

  // shutdown the rss
  rss[0]->shutdown();
  rss[1]->shutdown();

  _remote_cmd_t c = {nullptr, nullptr};
  remote_cmds[0].enqueue(c);
  remote_cmds[1].enqueue(c);

  std::pair<std::vector<tid_t>, int32_t> d = {{}, -1};
  remote_notifications[0].enqueue(d);
  remote_notifications[1].enqueue(d);

  // wait for remote executors to finish
  for(auto &t : _remote_executor_threads) {
    t.join();
  }

  // wait for the executors
  for(auto &t : _executor_threads) {
    t.join();
  }

  // wait for the executors
  for(auto &t : _notifier_threads) {
    t.join();
  }

  // wait for the executors
  for(auto &t : _notification_handler_threads) {
    t.join();
  }

  // make sure there are exactly two for the reduce...
  EXPECT_EQ(sto[0].size(), 0);
  EXPECT_EQ(sto[1].size(), 0);
}

}