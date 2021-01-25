#include "../src/operations/broadcast_op.h"

int main(int argc, char **argv) {

  // we set the root node to three as we want the results to got there
  const bbts::node_id_t root_node = 2;

  // make the configuration
  auto config = std::make_shared<bbts::node_config_t>(bbts::node_config_t{.argc = argc, .argv = argv});

  // create the tensor factory
  bbts::tensor_factory_ptr_t factory = std::make_shared<bbts::tensor_factory_t>();

  // init the communicator with the configuration
  bbts::communicator_t comm(config);

  // check the number of nodes
  if (root_node >= comm.get_num_nodes()) {
    std::cerr << "Can not run add more nodes or change the root node\n";
    return -1;
  }

  // create the storage
  bbts::storage_t storage;

  // get the impl_id
  auto id = factory->get_tensor_ftm("dense");

  // make the meta
  bbts::dense_tensor_meta_t dm{id, 100, 200};
  auto &m = dm.as<bbts::tensor_meta_t>();

  // get how much we need to allocate
  auto size = factory->get_tensor_size(m);
  std::unique_ptr<char[]> a_mem(new char[size]);

  // init the tensor
  auto &a = factory->init_tensor((bbts::tensor_t *) a_mem.get(), m).as<bbts::dense_tensor_t>();

  // get a reference to the metadata
  auto &am = a.meta().m();

  // if this is the root node init the matrix we want to broadcast
  if (comm.get_rank() == root_node) {

    // init the matrix
    for (int i = 0; i < am.num_rows * am.num_cols; ++i) {
      a.data()[i] = 1 + root_node + i;
    }
  }

  // pick all the nodes with an even rank, for testing purpouses 
  std::vector<bbts::node_id_t> nodes;
  for (bbts::node_id_t i = 0; i < comm.get_num_nodes(); i += 2) {
    nodes.push_back(i);
  }
  std::swap(nodes[0], *std::find(nodes.begin(), nodes.end(), root_node));

  // if the rank is even this takes part
  bool success = (am.num_rows * am.num_cols) != 0;
  if (comm.get_rank() % 2 == 0) {

    // make a broadcast
    bbts::broadcast_op_t bcst(comm,
                              *factory,
                              storage,
                              bbts::command_t::node_list_t{._data = nodes.data(), ._num_elements = nodes.size()},
                              888,
                              &a,
                              12);

    // execute the broadcast
    auto &bcs = bcst.apply()->as<bbts::dense_tensor_t>();

    // check if the values are fine
    for (int i = 0; i < am.num_rows * am.num_cols; ++i) {
      int32_t val = 1 + root_node + i;
      if (bcs.data()[i] != val) {
        success = false;
        std::cout << "not ok " << val << " " << bcs.data()[i] << '\n';
      }
    }
  }

  // wait for all
  comm.barrier();

  if (!success) {
    std::cout << "Failed here\n";
  } else {
    std::cout << "Everything ok\n";
  }

  return 0;
}