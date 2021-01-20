#include <map>
#include <thread>
#include <atomic>
#include "../src/operations/move_op.h"
#include "../src/commands/reservation_station.h"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "EndlessLoop"
using namespace bbts;

using index_t = std::map<std::tuple<int32_t, int32_t>, std::tuple<node_id_t, tid_t>>;
using to_agg_index_t = std::map<std::tuple<int32_t, int32_t>, std::vector<std::tuple<tid_t, node_id_t>>>;

// creates the matrix tensors on this node
index_t create_matrix_tensors(char matrix,
                              reservation_station_ptr_t &rs,
                              bbts::tensor_factory_ptr_t &tf,
                              bbts::storage_ptr_t &ts,
                              int n,
                              int split,
                              int my_rank,
                              int num_nodes,
                              int &cur_tid) {

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
      auto target_node = hash_fn(row_id * split + col_id) % num_nodes;
      if (target_node == my_rank) {

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
        index[{row_id, col_id}] = {target_node, cur_tid};
        rs->register_tensor(cur_tid);

        std::cout << "CREATE(matrix=" << matrix << ", tensor=(" << row_id << ", " << col_id << "), tid=" << cur_tid
                  << " , node=" << my_rank
                  << ")\n";
      } else {

        // store the index
        index[{row_id, col_id}] = {target_node, cur_tid};
      }

      // go to the next one
      cur_tid++;
    }
  }

  // return the index
  return std::move(index);
}

template<class fun>
void create_shuffle(size_t num_nodes,
                    size_t split,
                    command_id_t &cur_cmd,
                    fun fn,
                    index_t &mat_locs,
                    std::vector<command_ptr_t> &_cmds,
                    std::vector<std::vector<int32_t>> &to_del) {


  // do the shuffle
  for (int32_t rowID = 0; rowID < split; ++rowID) {
    for (int32_t colID = 0; colID < split; ++colID) {

      // get the tid and the node of this block
      auto &[node, tid] = mat_locs[{rowID, colID}];

      // no need to move here
      auto target_node = (node_id_t) fn(rowID, colID, num_nodes);
      if (node == target_node) {
        continue;
      }

      // move it
      _cmds.emplace_back(command_t::create_unique(cur_cmd++,
                                                  command_t::op_type_t::MOVE,
                                                  {-1, -1},
                                                  {command_t::tid_node_id_t{.tid = tid, .node = node}},
                                                  {command_t::tid_node_id_t{.tid = tid, .node = target_node}}));


      // mark that we need to delete it later
      to_del[target_node].push_back(tid);
    }
  }
}

to_agg_index_t create_multiply(const udf_manager_ptr &udm,
                               command_id_t &cur_cmd,
                               size_t split,
                               size_t num_nodes,
                               index_t a_mat,
                               index_t b_mat,
                               int32_t &tid_offset,
                               std::vector<command_ptr_t> &_cmds,
                               std::vector<std::vector<int32_t>> &to_del) {

  // create all the multiply commands
  std::map<std::tuple<int32_t, int32_t>, std::vector<std::tuple<tid_t, node_id_t>>> multiplies;

  // return me that matcher for matrix addition
  auto matcher = udm->get_matcher_for("matrix_mult");

  // get the ud object
  auto ud = matcher->findMatch({"dense", "dense"}, {"dense"}, false, 0);

  // generate the applies to do the multiplication
  for (int32_t i = 0; i < split; ++i) {
    for (int32_t j = 0; j < split; ++j) {
      for (int32_t k = 0; k < split; ++k) {

        // get the tid and the node
        auto[a_node, a_tid] = a_mat[{i, k}];
        auto[b_node, b_tid] = b_mat[{k, j}];

        // get the target node
        auto target_node = (node_id_t) (k % num_nodes);

        // add the command
        _cmds.emplace_back(command_t::create_unique(cur_cmd++,
                                                    command_t::op_type_t::APPLY,
                                                    ud->impl_id,
                                                    {command_t::tid_node_id_t{.tid = a_tid, .node = target_node},
                                                     command_t::tid_node_id_t{.tid = b_tid, .node = target_node}},
                                                    {command_t::tid_node_id_t{.tid = tid_offset, .node = target_node}}));

        // mark that we need to delete this tensor later
        to_del[target_node].push_back(tid_offset);

        // do the multiplies
        multiplies[{i, j}].push_back({tid_offset, target_node});

        // go to next tid
        tid_offset++;
      }
    }
  }

  return std::move(multiplies);
}

void generate_aggregation(const udf_manager_ptr &udm,
                          size_t split,
                          size_t num_nodes,
                          int32_t &tid_offset,
                          command_id_t &cur_cmd,
                          to_agg_index_t &multiplies,
                          std::vector<command_ptr_t> &_cmds) {

  // return me that matcher for matrix addition
  auto matcher = udm->get_matcher_for("matrix_add");

  // get the ud object
  auto ud = matcher->findMatch({"dense", "dense"}, {"dense"}, false, 0);

  // create the aggregate
  for (int32_t rowID = 0; rowID < split; ++rowID) {
    for (int32_t colID = 0; colID < split; ++colID) {

      // all the multiplied tensors
      auto &muls = multiplies[{rowID, colID}];

      // get the target node
      auto target_node = (node_id_t) ((rowID + colID * split) % num_nodes);

      // figure out the inputs
      std::vector<bbts::command_t::tid_node_id_t> inputs;
      for (auto &mul : muls) {
        auto &[tid, node] = mul;
        inputs.push_back({.tid = tid, .node = node});
      }

      // create the reduce command
      _cmds.emplace_back(command_t::create_unique(cur_cmd++,
                                                  command_t::op_type_t::REDUCE,
                                                  ud->impl_id,
                                                  inputs,
                                                  {command_t::tid_node_id_t{.tid = tid_offset, .node = target_node}}));

      tid_offset++;
    }
  }
}

void create_delete(size_t num_nodes,
                   std::vector<std::vector<int32_t>> &to_del, std::vector<command_ptr_t> &_cmds,
                   command_id_t &cur_cmd) {

  // prepare the removes
  for (int32_t node = 0; node < num_nodes; ++node) {


    // store the number we need to delete
    std::vector<bbts::command_t::tid_node_id_t> _inputs;
    _inputs.reserve(to_del[node].size());
    for (auto t : to_del[node]) {
      _inputs.push_back(command_t::tid_node_id_t{.tid = t, .node = node});
    }

    // remove them from node
    _cmds.emplace_back(command_t::create_unique(cur_cmd++,
                                                command_t::op_type_t::DELETE,
                                                {0, 0},
                                                _inputs,
                                                {}));
  }
}

std::vector<command_ptr_t> generate_commands(size_t num_nodes,
                                             size_t split,
                                             index_t a_mat,
                                             index_t b_mat,
                                             int32_t &tid_offset,
                                             const udf_manager_ptr &udm) {

  // we put the commands we want to schedule here
  std::vector<command_ptr_t> _cmds;
  command_id_t cur_cmd = 0;

  // all the tensors that we need to delete
  std::vector<std::vector<int32_t>> to_del(num_nodes);

  // create the shuffle
  create_shuffle(num_nodes,
                 split,
                 cur_cmd,
                 [](int32_t rowID, int32_t colID, size_t num_nodes) { return colID % num_nodes; },
                 a_mat,
                 _cmds,
                 to_del);

  // create the shuffle
  create_shuffle(num_nodes,
                 split,
                 cur_cmd,
                 [](int32_t rowID, int32_t colID, size_t num_nodes) { return rowID % num_nodes; },
                 b_mat,
                 _cmds,
                 to_del);


  // create the multiply commands
  auto multiplies = create_multiply(udm, cur_cmd, split, num_nodes, a_mat, b_mat, tid_offset, _cmds, to_del);

  // generate the aggregation
  generate_aggregation(udm, split, num_nodes, tid_offset, cur_cmd, multiplies, _cmds);

  // create the delete
  create_delete(num_nodes, to_del, _cmds, cur_cmd);

  // move the commands
  return std::move(_cmds);
}

// the reservation station needs a deleter thread
std::thread create_deleter_thread(reservation_station_ptr_t &_rs, bbts::storage_ptr_t &_sto) {

  // create the thread
  return std::thread([_rs, _sto]() {

    // while we have something remove
    tid_t id;
    while (true) {

      // get the next tensor to remove
      id = _rs->get_to_remove();
      if (id == -1) {
        break;
      }

      // deleted
      _sto->remove_by_tid(id);
      std::cout << "Remove tensor : " << id << '\n' << std::flush;
    }
  });
}

std::thread create_command_processing_thread(const communicator_ptr_t& comm,
                                             reservation_station_ptr_t &rs,
                                             udf_manager_ptr &udm,
                                             bbts::storage_ptr_t &ts,
                                             bbts::tensor_factory_ptr_t &tf,
                                             int32_t node) {

  // create the thread to pull
  std::thread t = std::thread([comm, rs, ts, node, udm, tf]() {

    while (true) {

      // get the command
      auto cmd = rs->get_next_command();
      if (cmd == nullptr) {
        break;
      }

      // if we have a move
      if (cmd->type == command_t::MOVE) {

        // move the
        std::cout << "MOVE " << cmd->id << " on my_node : " << node << " Executed...\n" << std::flush;

        // forward the command to the right nodes
        if(!comm->op_request(cmd)) {
          throw std::runtime_error("Failed to forward the command.");
        }

        // it is a point to point move
        if(cmd->is_move()) {

          // get the tensor we want to sent
          auto t = ts->get_by_tid(cmd->get_input(0).tid);

          // create the move operation
          move_op_t op(*comm, cmd->id, t, cmd->get_input(0).tid, true, *tf, *ts, cmd->get_output(0).node);

          // do the apply
          op.apply();
        }
        // it is a broadcast
        else {

          std::cout << "BROADCAST\n";
        }

        // retire the command so it knows that we have processed the tensors
        rs->retire_command(std::move(cmd));

      } else if (cmd->type == command_t::APPLY) {

        std::cout << "APPLY " << cmd->id << " on my_node : " << node << " Executed...\n" << std::flush;

        // return me that matcher for matrix addition
        auto ud = udm->get_fn_impl(cmd->fun_id);

        /// 1. Figure out the meta data for the output

        // make the input meta parameters
        ud_impl_t::meta_params_t input_meta_params(cmd->get_num_inputs());
        for (size_t idx = 0; idx < cmd->get_num_inputs(); ++idx) {

          // store it
          auto t = ts->get_by_tid(cmd->get_input(idx).tid);
          assert(t != nullptr);
          input_meta_params.set(idx, t->_meta);
        }

        // figure out the output parameters
        std::vector<tensor_meta_t> parameters(cmd->get_num_outputs());
        ud_impl_t::meta_params_t output_meta_params(parameters);

        // get the output meta
        ud->get_out_meta(input_meta_params, output_meta_params);

        /// 2. Prepare the output parameters

        // make the input parameters
        ud_impl_t::tensor_params_t input_params(cmd->get_num_inputs());
        for (size_t idx = 0; idx < cmd->get_num_inputs(); ++idx) {

          // store it
          auto t = ts->get_by_tid(cmd->get_input(idx).tid);
          input_params.set(idx, *t);
        }

        // setup the output parameters
        ud_impl_t::tensor_params_t output_params(cmd->get_num_outputs());
        for (size_t idx = 0; idx < cmd->get_num_outputs(); ++idx) {

          // the size of the tensor
          auto ts_size = tf->get_tensor_size(output_meta_params.get_by_idx(idx));

          // store it
          auto t = ts->create_tensor(cmd->get_output(idx).tid, ts_size);
          output_params.set(idx, *t);
        }

        /// 3. Run the actual UD Function

        // apply the ud function
        ud->fn(input_params, output_params);

        // retire the command so it knows that we have processed the tensors
        rs->retire_command(std::move(cmd));

      } else if (cmd->type == command_t::DELETE) {

        // this should never happen
        throw std::runtime_error("We should never get a delete to execute, delete is implicit...");

      } else if (cmd->type == command_t::REDUCE) {

        // check if the reduce is remote or local
        if (cmd->is_local_reduce(node)) {

          std::cout << "LOCAL_REDUCE " << cmd->id << " on node " << node << '\n' << std::flush;

        } else {

          std::cout << "REMOTE_REDUCE " << cmd->id << " on node " << node << '\n' << std::flush;
        }
      }
    }
  });

  return std::move(t);
}

std::thread expect_remote_command(bbts::storage_ptr_t &ts,
                                  bbts::tensor_factory_ptr_t &tf,
                                  reservation_station_ptr_t &rs,
                                  const bbts::communicator_ptr_t &comm) {

  // create the thread
  std::thread t = std::thread([comm, ts, tf, rs]() {

    // while we
    while (true) {

      // get the remote command
      auto cmd = comm->expect_op_request();

      // if this is a shutdown just finish immediately
      if(cmd->type == bbts::command_t::MOVE) {

        // check if this is the move
        if(cmd->is_move()) {

          // create the move operation
          move_op_t op(*comm, cmd->id, nullptr, cmd->get_input(0).tid, false, *tf, *ts, cmd->get_input(0).node);

          // do the apply
          op.apply();

          // retire the command
          rs->retire_command(std::move(cmd));
        }
        // check if this is a broadcast
        else {

        }

      }
      else if(cmd->type == bbts::command_t::REDUCE) {

      }
      else if(cmd->type == bbts::command_t::SHUTDOWN) {
        break;
      }
      else {

        // throw the runtime error
        throw std::runtime_error("This is bad can not process a command of type " + std::to_string(cmd->type));
      }
    }
  });

  return std::move(t);
}

int main(int argc, char **argv) {

  // the number of threads per node
  const int32_t num_threads = 1;
  const size_t split = 2;

  // make the configuration
  auto config = std::make_shared<bbts::node_config_t>(bbts::node_config_t{.argc=argc, .argv = argv});

  // create the storage
  storage_ptr_t ts = std::make_shared<storage_t>();

  // create the tensor factory
  auto tf = std::make_shared<bbts::tensor_factory_t>();

  // crate the udf manager
  auto udm = std::make_shared<udf_manager>(tf);

  // init the communicator with the configuration
  bbts::communicator_ptr_t comm = std::make_shared<bbts::communicator_t>(config);
  auto my_rank = comm->get_rank();
  auto num_nodes = comm->get_num_nodes();

  // create the reservation station
  auto rs = std::make_shared<bbts::reservation_station_t>(my_rank, num_nodes);

  // create two tensors split into num_nodes x num_nodes, we split them by some hash
  int32_t tid_offset = 0;
  std::cout << "Creating tensor A....\n";
  auto a_idx = create_matrix_tensors('A', rs, tf, ts, 1000, split, my_rank, num_nodes, tid_offset);
  std::cout << "Creating tensor B....\n";
  auto b_idx = create_matrix_tensors('B', rs, tf, ts, 1000, split, my_rank, num_nodes, tid_offset);

  // generate all the commands
  auto cmds = generate_commands(num_nodes, split, a_idx, b_idx, tid_offset, udm);

  // schedule them all at once
  for (auto &_cmd : cmds) {

    // if it uses the node
    if (_cmd->uses_node(my_rank)) {
      rs->queue_command(_cmd->clone());
    }
  }

  // now that we have scheduled all wait
  comm->barrier();

  // kick off the deleter thread
  auto deleter = create_deleter_thread(rs, ts);

  // executors
  std::vector<std::thread> _notification_handler_threads;
  _notification_handler_threads.reserve(num_nodes);
  for (node_id_t node = 0; node < num_nodes; ++node) {
    _notification_handler_threads.push_back(std::move(create_command_processing_thread(comm,
                                                                                       rs,
                                                                                       udm,
                                                                                       ts,
                                                                                       tf,
                                                                                       comm->get_num_nodes())));
  }

  // this kicks off and handles remove commands (MOVE and REDUCE)
  auto command_expect = expect_remote_command(ts, tf, rs, comm);

  // wait for the deleter to finish
  deleter.join();

  // wait for expect co
  command_expect.join();

  return 0;
}