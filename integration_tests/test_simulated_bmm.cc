#include <map>
#include "../src/operations/move_op.h"

using namespace bbts;

using index_t = std::map<std::tuple<int, int>, int>;
using multi_index_t = std::map<std::tuple<int, int>, std::vector<int>>;

index_t create_matrix_tensors(bbts::tensor_factory_ptr_t &tf, bbts::storage_ptr_t &ts,
                              int n, int split, int my_rank, int num_nodes, int &cur_tid) {

  // the index
  index_t index;

  // block size
  int block_size = n / split;

  // grab the format id of the dense tensor
  auto fmt_id = tf->get_tensor_ftm("dense");

  // create all the rows an columns we need
  auto hash_fn = std::hash<int>();
  for(int row_id = 0; row_id < split; ++row_id) {
    for(int col_id = 0; col_id < split; ++col_id) {

      // check if this block is on this node
      auto hash = hash_fn(row_id * split + col_id) % num_nodes;
      if(hash == my_rank) {

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
        index[{row_id, col_id}] = cur_tid;
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
  for(auto &t : idx) {

    // make the command
    auto cmd = std::make_unique<bbts::command_t>();
    cmd->_type = bbts::command_t::MOVE;
    cmd->_id = cur_cmd++;
    cmd->_input_tensors.push_back({.tid = t.second, .node = my_rank});

    // go through all the nodes
    for(int32_t node = 0; node < num_nodes; ++node) {

      // skip this node
      if(node == my_rank) { continue; }

      // set the output tensor
      cmd->_output_tensors.push_back({.tid = t.second, .node = my_rank});
    }

    // store the command
    commands.emplace_back(std::move(cmd));
  }

  return commands;
}

std::vector<command_ptr_t> create_shuffle(index_t &idx, int my_rank, int num_nodes, int &cur_cmd) {

  // go through the tuples
  std::vector<command_ptr_t> commands;
  for(auto &t : idx) {

    // where we need to move
    int32_t toNode = std::get<0>(t.first) % num_nodes;

    // make the command
    auto cmd = std::make_unique<bbts::command_t>();
    cmd->_type = bbts::command_t::MOVE;
    cmd->_id = cur_cmd++;
    cmd->_input_tensors.push_back({.tid = t.second, .node = my_rank});
    cmd->_output_tensors.push_back({.tid = t.second, .node = toNode});
  }

  return std::move(commands);
}

std::vector<command_ptr_t> create_join(udf_manager_ptr &udm, index_t &lhs, index_t &rhs, multi_index_t &out_idx,
                                       int my_rank, int split, int &cur_cmd, int &cur_tid) {

  // return me that matcher for matrix addition
  auto matcher = udm->get_matcher_for("matrix_mult");

  // get the ud object
  auto ud = matcher->findMatch({"dense", "dense"}, {"dense"}, false, 0);

  // generate all the commands
  std::vector<command_ptr_t> commands;
  for(int a_row_id = 0; a_row_id < split; ++a_row_id) {
    for (int b_col_id = 0; b_col_id < split; ++b_col_id) {

      // create all the join groups that need to be reduced together
      auto &tensor_to_reduce = out_idx[{a_row_id, b_col_id}];
      for (int ab_row_col_id = 0; ab_row_col_id < split; ++ab_row_col_id) {

        // make the command
        auto cmd = std::make_unique<bbts::command_t>();
        cmd->_type = bbts::command_t::APPLY;
        cmd->_id = cur_cmd++;
        cmd->_fun_id = ud->id;

        // get the tids for the left and right
        auto l = lhs[{a_row_id, ab_row_col_id}];
        auto r = rhs[{ab_row_col_id, b_col_id}];

        // set the left and right input
        cmd->_input_tensors.push_back({.tid = l, .node = my_rank});
        cmd->_input_tensors.push_back({.tid = r, .node = my_rank});

        //set the output
        cmd->_output_tensors.push_back({.tid = cur_tid, .node = my_rank});

        // store the command
        commands.emplace_back(move(cmd));
        tensor_to_reduce.push_back(cur_tid++);
      }
    }
  }

  // move the commands
  return std::move(commands);
}

int main(int argc, char **argv) {

  // make the configuration
  auto config = std::make_shared<bbts::node_config_t>(bbts::node_config_t{.argc=argc, .argv = argv});

  // create the storage
  storage_ptr_t ts = std::make_shared<storage_t>();

  // create the tensor factory
  bbts::tensor_factory_ptr_t tf = std::make_shared<bbts::tensor_factory_t>();

  // crate the udf manager
  auto udm = std::make_shared<udf_manager>(tf);

  // init the communicator with the configuration
  bbts::communicator_t comm(config);

  // check the number of nodes
  if(comm.get_num_nodes() % 2 != 0) {
    std::cerr << "Must use an even number of nodes.\n";
    return -1;
  }

  // create two tensors split into num_nodes x num_nodes, we split them by some hash
  std::cout << "Creating tensors....\n";
  int tid_offset = 0;
  auto a_idx = create_matrix_tensors(tf, ts, 1000, comm.get_num_nodes(), comm.get_rank(), comm.get_num_nodes(), tid_offset);
  auto b_idx = create_matrix_tensors(tf, ts, 1000, comm.get_num_nodes(), comm.get_rank(), comm.get_num_nodes(), tid_offset);

  // create the broadcast commands
  std::cout << "Creating broadcast commands...\n";
  int32_t cmd_offest = 0;
  auto bcast_cmds = create_broadcast(a_idx, comm.get_rank(), comm.get_num_nodes(), cmd_offest);

  // create the shuffle commands
  std::cout << "Create the shuffle commands...\n";
  auto shuffle_cmds = create_shuffle(b_idx, comm.get_rank(), comm.get_num_nodes(), cmd_offest);

  // create an join commands
  std::cout << "Creating join commands...\n";
  multi_index_t join;
  create_join(udm, a_idx, b_idx, join, comm.get_rank(), comm.get_num_nodes(), cmd_offest, tid_offset);

  // create an aggregation commands
  std::cout << "Creating aggregation commands...\n";


  return 0;
}