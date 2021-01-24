#include "../src/server/node.h"
#include <unistd.h>
#include <map>

using namespace bbts;

using index_t = std::map<std::tuple<int, int>, std::tuple<int, bool>>;
using multi_index_t = std::map<std::tuple<int, int>, std::vector<int>>;

class test_node : public bbts::node_t {
 public:

  explicit test_node(const node_config_ptr_t &config) : node_t(config) {}

  // create the matrices
  void create_matrices() {

    auto num_nodes = _comm->get_num_nodes();
    auto my_rank = _comm->get_rank();

    // create two tensors split into num_nodes x num_nodes, we split them by some hash
    int tid_offset = 0;

    // create the tensor A
    //std::cout << "Creating tensor A....\n";
    a_idx =
        create_matrix_tensors('A', _res_station, _factory, _storage, 1000, num_nodes, my_rank, num_nodes, tid_offset);

    // create the tensor B
    //std::cout << "Creating tensor B....\n";
    b_idx =
        create_matrix_tensors('B', _res_station, _factory, _storage, 1000, num_nodes, my_rank, num_nodes, tid_offset);
  }

  void schedule() {

    // we only schedule on node 0
    if (_comm->get_rank() != 0) {
      return;
    }

    int32_t curCmd = 0;
    for (auto node = 0; node < _comm->get_num_nodes(); ++node) {
      auto cmds = create_shuffle(a_idx, node, _comm->get_num_nodes(), curCmd);
      schedule_all(cmds);
    }
  }

 private:

  // crates a matrix distributed around the cluster
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

          //std::cout << "CREATE(matrix=" << matrix << ", tensor=(" << row_id << ", " << col_id << "), tid=" << cur_tid << " , node=" << my_rank << ")\n";
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

  // create the shuffle commands
  std::vector<command_ptr_t> create_shuffle(index_t &idx, int my_rank, int num_nodes, int &cur_cmd) {

    // go through the tuples
    std::vector<command_ptr_t> commands;
    for (auto &t : idx) {

      // unfold the tuple
      auto[tid, present] = t.second;

      // where we need to move
      int32_t to_node = std::get<0>(t.first) % num_nodes;

      // if it stays on this node or is not present we are cool
      if (to_node == my_rank) {
        continue;
      }

      // make the command
      auto cmd = bbts::command_t::create_move(cur_cmd++, {.tid = tid, .node = my_rank}, {.tid = tid, .node = to_node});

      // store the command
      commands.emplace_back(std::move(cmd));

      //std::cout << "MOVE(tensor=(" << get<0>(t.first) << ", " << get<1>(t.first) << "), tid=" << tid << ", to_node=" << to_node << ", my_node=" << my_rank << ")\n";
    }

    return std::move(commands);
  }

  // schedule all commands
  void schedule_all(std::vector<command_ptr_t> &cmds) {

    // schedule the commands
    //std::cout << "Scheduling " << cmds.size() << "\n";
    for (auto &c : cmds) {
      _scheduler->schedule(std::move(c));
    }
  }

  // tells us what tensors blocks are present initially
  index_t a_idx;
  index_t b_idx;

};

int main(int argc, char **argv) {

  // make the configuration
  auto config = std::make_shared<bbts::node_config_t>(bbts::node_config_t{.argc=argc, .argv = argv});

  // make the node
  test_node node(config);

  // init the node
  node.init();

  // create the matrices
  node.create_matrices();

  // schedule the commands
  node.schedule();

  // run it
  node.run();

  return 0;
}