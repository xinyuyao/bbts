#include "../src/operations/move_op.h"

using namespace bbts;

int main(int argc, char **argv) {

  // make the configuration
  auto config = std::make_shared<bbts::node_config_t>(bbts::node_config_t{.argc=argc, .argv = argv});

  // create the tensor factory
  bbts::tensor_factory_ptr_t factory = std::make_shared<bbts::tensor_factory_t>();

  // init the communicator with the configuration
  bbts::communicator_t comm(config);

  // check the number of nodes
  if(comm.get_num_nodes() % 2 != 0) {
    std::cerr << "Must use an even number of nodes.\n";
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

  // we are only involving the even ranks in teh computation
  bool success = true;
  if(comm.get_rank() % 2 == 0) {

    std::unique_ptr<char[]> a_mem(new char[size]);

    // init the tensor
    auto &a = factory->init_tensor((bbts::tensor_t*) a_mem.get(), m).as<bbts::dense_tensor_t>();

    // get a reference to the metadata
    auto &am = a.meta().m();

    // init the matrix
    for(int i = 0; i < am.num_rows * am.num_cols; ++i) {
      a.data()[i] = 1 + comm.get_rank() + i;
    }

    // add some stats about the output
    bbts::tensor_stats_t _stats;
    _stats.add_tensor(comm.get_rank(), false);

    // create the move
    auto move = move_op_t(comm, comm.get_rank(), &a, size, _stats, comm.get_rank(), true, storage, comm.get_rank() + 1);
    auto mm = move.apply();
  }
  else {

    // add some stats about the output
    bbts::tensor_stats_t _stats;
    _stats.add_tensor(comm.get_rank() - 1, false);

    // create the move
    auto move = move_op_t(comm, comm.get_rank() - 1, nullptr, size, _stats, comm.get_rank() - 1, false, storage, comm.get_rank() - 1);
    auto m = move.apply();

    // get the dense tensor
    auto &a = m->as<bbts::dense_tensor_t>();
    
    // get a reference to the metadata
    auto &am = a.meta().m();

    // check the values
    for(int i = 0; i < am.num_rows * am.num_cols; ++i) {

      // the value
      int32_t val = comm.get_rank() + i;
      if(a.data()[i] != val) {
        success = false;
        std::cout << "not ok " << val << " " << a.data()[i] << '\n';
      }
    }
  }

  // wait for all
  comm.barrier();

  // if works
  if(success) {
    std::cerr << "all good\n";
  }

  return !success;
}