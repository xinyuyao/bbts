#include "../src/communication/communicator.h"
#include "../src/tensor/builtin_formats.h"
#include "../src/storage/storage.h"
#include "../src/ud_functions/udf_manager.h"
#include <mpich/mpi.h>
#include <iostream>
#include <algorithm>

class move_op {
public:

  // the mpi communicator we are going to use to perform the communication
  bbts::communicator_t &_comm;

  // the tag we use to identify this reduce
  int32_t _tag; 

  // the tensor of this node, either the input or the output
  bbts::tensor_t *_tensor; 

  // is this node the sender
  bool _is_sender;

  bbts::tensor_factory_t &_factory;
  
  bbts::storage_t &_storage;


  // the node we are send or recieving from...
  bbts::node_id_t _node;

  // constructs the reduce operation
  move_op(bbts::communicator_t &_comm, int32_t _tag, bbts::tensor_t *_tensor,  
          bool _is_sender, bbts::tensor_factory_t &_factory, 
          bbts::storage_t &_storage, bbts::node_id_t _node) : _comm(_comm),
                                                              _tag(_tag),
                                                              _tensor(_tensor),
                                                              _is_sender(_is_sender),
                                                              _factory(_factory),
                                                              _storage(_storage),
                                                              _node(_node) {}

  // apply this operation
  bbts::tensor_t *apply() {
    
    // is this the sender, if so we initiate a send request
    if(_is_sender) {

      // get the number of bytes we need to send
      auto num_bytes = _factory.get_tensor_size(_tensor->_meta);

      // do the sending
      if(!_comm.send_sync(_tensor, num_bytes, _node, _tag)) {
        std::cout << "Error 1\n";
      }

    } else {

      // try to get the request
      auto req = _comm.expect_request_sync(_node, _tag);

      // check if there is an error
      if (!req.success) {
        std::cout << "Error 2\n";
      }

      // allocate a buffer for the tensor
      _tensor = _storage.create_tensor(req.num_bytes);

      // recieve the request and check if there is an error
      if (!_comm.recieve_request_sync(_tensor, req)) {
        std::cout << "Error 3\n";
      }
    }

    // return the tensor
    return _tensor;
  }

};

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

  // we are only involving the even ranks in teh computation
  bool success = true;
  if(comm.get_rank() % 2 == 0) {

    // get the id
    auto id = factory->get_tensor_ftm("dense");

    // make the meta
    bbts::dense_tensor_meta_t dm{id, 100, 200};
    auto &m = dm.as<bbts::tensor_meta_t>();

    // get how much we need to allocate
    auto size = factory->get_tensor_size(m);
    std::unique_ptr<char[]> a_mem(new char[size]);

    // initi the tensor
    auto &a = factory->init_tensor((bbts::tensor_t*) a_mem.get(), m).as<bbts::dense_tensor_t>();

    // get a reference to the metadata
    auto &am = a.meta().m();

    // init the matrix
    for(int i = 0; i < am.num_rows * am.num_cols; ++i) {
      a.data()[i] = 1 + comm.get_rank() + i;
    }

    // craete the move
    auto move = move_op(comm, comm.get_rank(), &a, true, *factory, storage, comm.get_rank() + 1);
    auto mm = move.apply();
  }
  else {

    // create the move
    auto move = move_op(comm, comm.get_rank() - 1, nullptr, false, *factory, storage, comm.get_rank() - 1);
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