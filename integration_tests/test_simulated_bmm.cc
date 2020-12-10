#include <map>
#include <thread>
#include "../src/operations/move_op.h"
#include "../src/commands/reservation_station.h"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "EndlessLoop"
using namespace bbts;

using index_t = std::map<std::tuple<int, int>, std::tuple<int, bool>>;
using multi_index_t = std::map<std::tuple<int, int>, std::vector<int>>;

index_t create_matrix_tensors(char matrix, reservation_station_ptr_t &rs, bbts::tensor_factory_ptr_t &tf, bbts::storage_ptr_t &ts,
                              int n, int split, int my_rank, int num_nodes, int &cur_tid) {

  // the index
  index_t index;

  // block size
  int block_size = n / split;

  // grab the format impl_id of the dense tensor
  auto fmt_id = tf->get_tensor_ftm("dense");

  // create all the rows an columns we need
  auto hash_fn = std::hash<int>();
  for (int row_id = 0; row_id < split; ++row_id) {
    for (int col_id = 0; col_id < split; ++col_id) {

      // check if this block is on this node
      auto hash = hash_fn(row_id * split + col_id) % num_nodes;
      if (hash == my_rank) {

        // ok it is on this node make a tensor
        // make the meta
        dense_tensor_meta_t dm{fmt_id, block_size, block_size};

        // get the size of the tensor we need to crate
        auto tensor_size = tf->get_tensor_size(dm);

        // crate the tensor
        auto t = ts->create_tensor(cur_tid, tensor_size);

        // init the tensor
        auto &dt = tf->init_tensor(t, dm).as<dense_tensor_t>();

        // set the index
        index[{row_id, col_id}] = {cur_tid, true};
        rs->register_tensor(cur_tid);

        std::cout << "CREATE(matrix=" << matrix << ", tensor=(" << row_id << ", " << col_id << "), tid=" << cur_tid << " , node=" << my_rank
                  << ")\n";
      } else {

        // store the index
        index[{row_id, col_id}] = {cur_tid, false};
      }

      // go to the next one
      cur_tid++;
    }
  }

  // return the index
  return std::move(index);
}

std::vector<command_ptr_t> create_broadcast(index_t &idx, int my_rank, int num_nodes, int &cur_cmd) {

  // go through the tuples
  std::vector<command_ptr_t> commands;
  for (auto &t : idx) {

    // unfold the tuple
    auto[tid, present] = t.second;

    // if the block is not present on this node continue
    if (!present) {
      continue;
    }

    // make the command
    auto cmd = bbts::command_t::create_unique(1, num_nodes);
    cmd->type = bbts::command_t::MOVE;
    cmd->id = cur_cmd++;
    cmd->get_input(0) = {.tid = tid, .node = my_rank};

    // go through all the nodes
    for (int32_t node = 0; node < num_nodes; ++node) {

      // skip this node
      if (node == my_rank) { continue; }

      // set the output tensor
      cmd->get_output(node) = {.tid = tid, .node = my_rank};
    }

    // store the command
    commands.emplace_back(std::move(cmd));
  }

  return commands;
}

std::vector<command_ptr_t> create_shuffle(index_t &idx, int my_rank, int num_nodes, int &cur_cmd) {

  // go through the tuples
  std::vector<command_ptr_t> commands;
  for (auto &t : idx) {

    // unfold the tuple
    auto[tid, present] = t.second;

    // where we need to move
    int32_t to_node = std::get<0>(t.first) % num_nodes;

    // if it stays on this node or is not present we are cool
    if (to_node == my_rank || !present) {
      continue;
    }

    // make the command
    auto cmd = bbts::command_t::create_unique(1, 1);
    cmd->type = bbts::command_t::MOVE;
    cmd->id = cur_cmd++;
    cmd->get_input(0) = {.tid = tid, .node = my_rank};
    cmd->get_output(0) = {.tid = tid, .node = to_node};

    // store the command
    commands.emplace_back(std::move(cmd));

    std::cout << "MOVE(tensor=(" << get<0>(t.first) << ", " << get<1>(t.first) << "), tid=" << tid << ", to_node="
              << to_node << ", my_node=" << my_rank << ")\n";
  }

  return std::move(commands);
}

std::vector<command_ptr_t> create_join(udf_manager_ptr &udm, index_t &lhs, index_t &rhs, multi_index_t &out_idx,
                                       int my_rank, int num_nodes, int split, int &cur_cmd, int &cur_tid) {

  // return me that matcher for matrix addition
  auto matcher = udm->get_matcher_for("matrix_mult");

  // get the ud object
  auto ud = matcher->findMatch({"dense", "dense"}, {"dense"}, false, 0);

  // generate all the commands
  std::vector<command_ptr_t> commands;
  for (int a_row_id = 0; a_row_id < split; ++a_row_id) {

    // check this should be joined here
    if (a_row_id % num_nodes != my_rank) {
      continue;
    }

    // form all the ones we need to join
    for (int b_col_id = 0; b_col_id < split; ++b_col_id) {

      // create all the join groups that need to be reduced together
      auto &tensor_to_reduce = out_idx[{a_row_id, b_col_id}];
      for (int ab_row_col_id = 0; ab_row_col_id < split; ++ab_row_col_id) {

        // make the command
        auto cmd = bbts::command_t::create_unique(2, 1);
        cmd->type = bbts::command_t::APPLY;
        cmd->id = cur_cmd++;
        cmd->fun_id = ud->impl_id;

        // get the tids for the left and right
        auto[l, l_present] = lhs[{a_row_id, ab_row_col_id}];
        auto[r, r_present] = rhs[{ab_row_col_id, b_col_id}];

        // log this
        std::cout << "JOIN((" << a_row_id << "," << ab_row_col_id << ")[" << l << "]," << "(" << ab_row_col_id << ","
                  << b_col_id << ")[" << r << "])\n";

        // set the left and right input
        cmd->get_input(0) = {.tid = l, .node = my_rank};
        cmd->get_input(1) = {.tid = r, .node = my_rank};

        //set the output
        cmd->get_output(0) = {.tid = cur_tid, .node = my_rank};

        // store the command
        commands.emplace_back(move(cmd));
        tensor_to_reduce.push_back(cur_tid++);
      }
    }
  }

  // move the commands
  return std::move(commands);
}

std::vector<command_ptr_t> create_agg(udf_manager_ptr &udm, multi_index_t &to_agg_idx, index_t &out,
                                      int my_rank, int &cur_cmd, int &cur_tid) {

  // return me that matcher for matrix addition
  auto matcher = udm->get_matcher_for("matrix_add");

  // get the ud object
  auto ud = matcher->findMatch({"dense", "dense"}, {"dense"}, false, 0);

  // generate all the commands
  std::vector<command_ptr_t> commands;
  for (auto &to_reduce : to_agg_idx) {

    // make the command
    auto cmd = bbts::command_t::create_unique(to_reduce.second.size(), 1);
    cmd->type = bbts::command_t::REDUCE;
    cmd->id = cur_cmd++;
    cmd->fun_id = ud->impl_id;

    // set the input tensors we want to reduce
    for(int i = 0; i < to_reduce.second.size(); ++i) {
      cmd->get_input(i) = {.tid = to_reduce.second[i], .node = my_rank};
    }

    cmd->get_output(0) = {.tid = cur_tid++, .node = my_rank};
    commands.emplace_back(move(cmd));
  }

  return std::move(commands);
}

void schedule_all(bbts::reservation_station_t &rs, std::vector<command_ptr_t> &cmds) {

  // schedule the commands
  std::cout << "Scheduling " << cmds.size() << "\n";
  for (auto &c : cmds) {
    rs.queue_command(std::move(c));
  }
}

int main(int argc, char **argv) {

  // the number of threads per node
  const int32_t num_threads = 1;

  // make the configuration
  auto config = std::make_shared<bbts::node_config_t>(bbts::node_config_t{.argc=argc, .argv = argv});

  // create the storage
  storage_ptr_t ts = std::make_shared<storage_t>();

  // create the tensor factory
  auto tf = std::make_shared<bbts::tensor_factory_t>();

  // crate the udf manager
  auto udm = std::make_shared<udf_manager>(tf);

  // init the communicator with the configuration
  bbts::communicator_t comm(config);
  auto my_rank = comm.get_rank();
  auto num_nodes = comm.get_num_nodes();

  // create the reservation station
  auto rs = std::make_shared<bbts::reservation_station_t>(my_rank, ts);

  // create two tensors split into num_nodes x num_nodes, we split them by some hash
  int tid_offset = 0;
  std::cout << "Creating tensor A....\n";
  auto a_idx = create_matrix_tensors('A' , rs, tf, ts, 1000, num_nodes, my_rank, num_nodes, tid_offset);
  std::cout << "Creating tensor B....\n";
  auto b_idx = create_matrix_tensors('B', rs, tf, ts, 1000, num_nodes, my_rank, num_nodes, tid_offset);

  // create the shuffle commands
  int32_t cmd_offest = 0;
  std::cout << "Create the shuffle commands...\n";
  auto shuffle_cmds = create_shuffle(a_idx, my_rank, num_nodes, cmd_offest);

  // create the broadcast commands
  std::cout << "Creating broadcast commands...\n";
  auto bcast_cmds = create_broadcast(b_idx, my_rank, num_nodes, cmd_offest);

  // create an join commands
  std::cout << "Creating join commands...\n";
  multi_index_t join;
  auto join_cmds = create_join(udm, a_idx, b_idx, join, my_rank, num_nodes, num_nodes, cmd_offest, tid_offset);

  // create an aggregation commands
  std::cout << "Creating aggregation commands...\n";
  index_t final;
  auto agg_cmds = create_agg(udm, join, final, my_rank, cmd_offest, tid_offset);

  // schedule the commands
  schedule_all(*rs, bcast_cmds);
  schedule_all(*rs, shuffle_cmds);
  schedule_all(*rs, join_cmds);
  schedule_all(*rs, agg_cmds);

  // kick of a bunch of threads that are going to grab commands
  std::vector<std::thread> commandExecutors;
  commandExecutors.reserve(num_threads);
  for (int32_t t = 0; t < num_threads; ++t) {

    // each thread is grabbing a command
    commandExecutors.emplace_back([&rs, &ts, &udm, &tf, &comm]() {

      std::vector<tensor_meta_t> _out_meta_tmp;

      for (;;) {

        // grab the next command
        auto cmd = rs->get_next_command();

        // are we doing an apply (applies are local so we are cool)
        if (cmd->type == bbts::command_t::APPLY) {

          // get the ud function we want to run
          auto call_me = udm->get_fn_impl(cmd->fun_id);

          // make the meta for the input
          std::vector<tensor_meta_t *> inputs_meta;
          for(int idx = 0; idx < cmd->get_num_inputs(); ++idx) {
            auto &in = cmd->get_input(idx);
            inputs_meta.push_back(&ts->get_by_tid(in.tid)->_meta);
          }
          bbts::ud_impl_t::meta_params_t input_meta(move(inputs_meta));

          // get the meta for the outputs
          _out_meta_tmp.resize(cmd->get_num_outputs());
          std::vector<tensor_meta_t *> outputs_meta;
          outputs_meta.reserve(_out_meta_tmp.size());

          // fill them up
          for (auto &om : _out_meta_tmp) { outputs_meta.push_back(&om); }
          bbts::ud_impl_t::meta_params_t out_meta(std::move(outputs_meta));

          // get the meta
          call_me->get_out_meta(input_meta, out_meta);

          // form all the inputs
          std::vector<tensor_t *> inputs;
          for(int idx = 0; idx < cmd->get_num_inputs(); ++idx) {
            auto &in = cmd->get_input(idx);
            inputs.push_back(ts->get_by_tid(in.tid));
          }
          ud_impl_t::tensor_params_t inputParams = {std::move(inputs)};

          // form the outputs
          std::vector<tensor_t *> outputs;
          for (int32_t i = 0; i < cmd->get_num_outputs(); ++i) {

            // get the size of tensor
            auto num_bytes = tf->get_tensor_size(out_meta.get_by_idx(i));

            // create the output tensor
            outputs.push_back(ts->create_tensor(cmd->get_output(i).tid, num_bytes));
          }
          ud_impl_t::tensor_params_t outputParams = {std::move(outputs)};

          // apply the function
          call_me->fn(inputParams, outputParams);

          // retire command
          std::cout << "Executed Apply for Function (" << cmd->fun_id.ud_id << ", " << cmd->fun_id.impl_id << ")\n";
          rs->retire_command(std::move(cmd));
        }
        // check if it is a point-to-point MOVE
        else if(cmd->type == bbts::command_t::MOVE && cmd->get_num_outputs() == 1) {

          std::cout << "Sending...\n";

          // send the command
          auto to_node = cmd->get_output(0).node;
          comm.op_request(cmd, to_node);

          // get the tensor
          auto tid = cmd->get_input(0).tid;
          auto t = ts->get_by_tid(tid);

          // initiate the MOVE
          auto move = move_op_t(comm, tid, t, true, *tf, *ts, to_node);
          move.apply();
        }
      }

    });
  }

  while(true) {

    // get the command
    auto cmd = comm.listen_for_op_request();

    std::cout << "Got command\n";

    // get the node from which we get the tensor from
    auto from_node = cmd->get_input(0).node;
    auto tid = cmd->get_input(0).tid;

    // create the move
    auto move = move_op_t(comm, tid, nullptr, false, *tf, *ts, from_node);
    auto m = move.apply();
    std::cout << "Finished command\n";
  }

  // wait for the threads to finish
  for (auto &t : commandExecutors) {
    t.join();
  }

  return 0;
}
#pragma clang diagnostic pop