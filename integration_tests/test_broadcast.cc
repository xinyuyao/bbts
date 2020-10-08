#include <mpich/mpi.h>

#include <iostream>

#include "../src/communication/communicator.h"
#include "../src/storage/storage.h"
#include "../src/tensor/builtin_formats.h"
#include "../src/ud_functions/udf_manager.h"

using namespace bbts;

class broadcast_op {
public:

  // constructs the broadcast operation
  broadcast_op(bbts::mpi_communicator_t &_comm,
               bbts::tensor_factory_t &_factory, bbts::storage_t &_storage,
               const std::vector<bbts::node_id_t> &_nodes, int32_t _root,
               int32_t _tag, bbts::tensor_t *_in)
      : _comm(_comm),
        _factory(_factory),
        _storage(_storage),
        _nodes(_nodes),
        _root(_root),
        _tag(_tag),
        _in(_in) {}

  // the mpi communicator we are going to use to perform the communication
  bbts::mpi_communicator_t &_comm;

  // we use the tensor factory to initialize the tensors and calculate the required size
  bbts::tensor_factory_t &_factory;

  // the storage we use this to allocate the output and the intermediate tensors
  bbts::storage_t &_storage;

  // the nodes
  const std::vector<bbts::node_id_t> &_nodes;

  // the root node of the reduce
  int32_t _root;

  // the tag we use to identify this reduce
  int32_t _tag;

  // the input tensor of this node
  bbts::tensor_t *_in;

  // calculates the highest bit in an integer
  static inline int opal_hibit(int value, int start) {
    unsigned int mask;

    /* Only look at the part that the caller wanted looking at */
    mask = value & ((1 << start) - 1);

    if ((0 == mask)) {
      [[unlikely]]
      return -1;
    }

    start = (8 * sizeof(int) - 1) - __builtin_clz(mask);

    return start;
  }

  // cubedim The smallest cube dimension containing that value
  static inline int opal_cube_dim(int value) {
    int dim, size;

    if ((1 >= value)) {
      [[unlikely]]
      return 0;
    }
    size = 8 * sizeof(int);
    dim = size - __builtin_clz(value - 1);

    return dim;
  }

  // runs the broadcast
  bbts::tensor_t *apply() {

    int size = _comm.get_num_nodes();
    int rank = _comm.get_rank();
    int vrank = (rank + size - _root) % size;

    int dim = opal_cube_dim(size);
    int hibit = opal_hibit(vrank, dim);
    --dim;

    // the root node has the vrank 0, if this is not the root node 
    // we need to recieve the broadcasted tensor
    if (vrank > 0) {

      // figure out the node we need to recieve the data from
      assert(hibit >= 0);
      int peer = ((vrank & ~(1 << hibit)) + _root) % size;

      // try to get the request
      auto req = _comm.expect_request_sync(peer, _tag);

      // check if there is an error
      if (!req.success) {
        std::cout << "Error 6\n";
      }

      // allocate a buffer for the tensor
      _in = _storage.create_tensor(req.num_bytes);

      // recieve the request and check if there is an error
      if (!_comm.recieve_request_sync(_in, req)) {
        std::cout << "Error 5\n";
      }
    }

    // allocate the requests
    std::vector<communicator::async_request_t> requests;
    requests.reserve(size);

    // send the tensor to the right nodes
    for (int i = hibit + 1, mask = 1 << i; i <= dim; ++i, mask <<= 1) {
      int peer = vrank | mask;
      if (peer < size) {

        // figure out where we need to send it
        peer = (peer + _root) % size;

        // return the size of the tensor
        auto output_size = _factory.get_tensor_size(_in->_meta);

        // send the tensor async
        requests.emplace_back(_comm.send_async(_in, output_size, peer, _tag));

        // we failed here return null
        if(!requests.back().request) {
          return nullptr;
        }
      }
    }

    // wait on all requests
    bool success = true;
    if (!requests.empty()) {
      for(auto &r : requests) {
        
        // wait for request to finish
        if(MPI_Wait(&r.request, MPI_STATUSES_IGNORE) != MPI_SUCCESS) {
          success = false;
        }
      }
    }

    // if we failed return null
    if(!success) {
      return nullptr;
    }

    // return the tensor
    return _in;
  }
};

int main(int argc, char **argv) {

  // we set the root node to three as we want the results to got there
  const bbts::node_id_t root_node = 0;

  // make the configuration
  auto config = std::make_shared<bbts::node_config_t>(bbts::node_config_t{.argc = argc, .argv = argv});

  // create the tensor factory
  bbts::tensor_factory_ptr_t factory = std::make_shared<bbts::tensor_factory_t>();

  // init the communicator with the configuration
  bbts::mpi_communicator_t comm(config);

  // check the number of nodes
  if (root_node >= comm.get_num_nodes()) {
    std::cerr << "Can not run add more nodes or change the root node\n";
    return -1;
  }

  // create the storage
  bbts::storage_t storage;

  // get the id
  auto id = factory->get_tensor_ftm("dense");

  // crate the udf manager
  bbts::udf_manager manager(factory);

  // return me that matcher for matrix addition
  auto matcher = manager.get_matcher_for("matrix_add");

  // get the ud object
  auto ud = matcher->findMatch({"dense", "dense"}, {"dense"}, false, 0);

  // make the meta
  bbts::dense_tensor_meta_t dm{id, 100, 200};
  auto &m = dm.as<bbts::tensor_meta_t>();

  // get how much we need to allocate
  auto size = factory->get_tensor_size(m);
  std::unique_ptr<char[]> a_mem(new char[size]);

  // init the tensor
  auto &a = factory->init_tensor((bbts::tensor_t *)a_mem.get(), m).as<bbts::dense_tensor_t>();

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
  for(bbts::node_id_t i = 0; i < comm.get_num_nodes(); i += 2) {
    nodes.push_back(i);
  }

  // make a broadcast
  broadcast_op bcst(comm, *factory, storage, nodes, 0, 888, &a);

  // execute the broadcast
  auto &bcs = bcst.apply()->as<dense_tensor_t>();

  // check if the values are fine
  bool success = (am.num_rows * am.num_cols) != 0;
  for(int i = 0; i < am.num_rows * am.num_cols; ++i) {
      int32_t val = 1 + root_node + i;
      if(bcs.data()[i] != val) {
        success = false;
        std::cout << "not ok " << val << " " << bcs.data()[i] << '\n';
      }
  }

  // wait for all
  comm.barrier();

  if(!success) {
    std::cout << "Failed here\n";
  }
  else  {
    std::cout << "Everything ok\n";
  }

  return 0;
}