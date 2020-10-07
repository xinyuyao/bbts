#include "../src/communication/communicator.h"
#include "../src/tensor/builtin_formats.h"
#include "../src/storage/storage.h"
#include "../src/ud_functions/udf_manager.h"
#include <mpich/mpi.h>
#include <stdio.h>
#include <iostream>


// This code is based on the implementation from MPICH-1.
// Here's the algorithm.  Relative to the root, look at the bit pattern in
// my rank.  Starting from the right (lsb), if the bit is 1, send to
// the node with that bit zero and exit; if the bit is 0, receive from the
// node with that bit set and combine (as long as that node is within the
// group)

// Note that by receiving with source selection, we guarantee that we get
// the same bits with the same input.  If we allowed the parent to receive
// the children in any order, then timing differences could cause different
// results (roundoff error, over/underflows in some cases, etc).

// Because of the way these are ordered, if root is 0, then this is correct
// for both commutative and non-commutitive operations.  If root is not
// 0, then for non-commutitive, we use a root of zero and then send
// the result to the root.  To see this, note that the ordering is
// mask = 1: (ab)(cd)(ef)(gh)            (odds send to evens)
// mask = 2: ((ab)(cd))((ef)(gh))        (3,6 send to 0,4)
// mask = 4: (((ab)(cd))((ef)(gh)))      (4 sends to 0)

// Comments on buffering.
// If the datatype is not contiguous, we still need to pass contiguous
// data to the user routine.
// In this case, we should make a copy of the data in some format,
// and send/operate on that.

// In general, we can't use MPI_PACK, because the alignment of that
// is rather vague, and the data may not be re-usable.  What we actually
// need is a "squeeze" operation that removes the skips.
class reduce {
public:

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
  bbts::tensor_t &_in; 

  // the reduce operation
  const bbts::ud_impl_t &_reduce_op;

  // constructs the reduce operation
  reduce(bbts::mpi_communicator_t &_comm, bbts::tensor_factory_t &_factory, 
         bbts::storage_t &_storage, const std::vector<bbts::node_id_t> &_nodes,
         int32_t _root, int32_t _tag, bbts::tensor_t &_in, const bbts::ud_impl_t &_reduce_op) : _comm(_comm), 
                                                                                                _factory(_factory),
                                                                                                _storage(_storage),
                                                                                                _nodes(_nodes),
                                                                                                _root(_root),
                                                                                                _tag(_tag),
                                                                                                _in(_in),
                                                                                                _reduce_op(_reduce_op) {}

  bbts::tensor_t *apply() {

    int32_t mask = 0x1;
    int32_t lroot = 0;

    // relative rank
    int32_t relrank = (_comm.get_rank() - lroot + _comm.get_num_nodes()) % _comm.get_num_nodes();

    // get the lhs address
    bbts::tensor_t *lhs = &_in;

    // make empty input parameter
    bbts::tensor_meta_t out_meta;
    bbts::ud_impl_t::tensor_params_t input_tensors({nullptr, nullptr});
    bbts::ud_impl_t::tensor_params_t output_tensor({nullptr});
    bbts::ud_impl_t::meta_params_t input_meta({nullptr, nullptr});
    bbts::ud_impl_t::meta_params_t output_meta({&out_meta});

    // get the id of the output
    auto id = _factory.get_tensor_ftm(_reduce_op.outputTypes.front());

    // do stuff
    int32_t source;
    while (mask < _comm.get_num_nodes()) {

      // receive 
      if ((mask & relrank) == 0) {
        
        source = (relrank | mask);
        if (source < _comm.get_num_nodes()) {

          // wait till we get a message from the right node
          source = (source + lroot) % _comm.get_num_nodes();

          // try to get the request
          auto req = _comm.expect_request_sync(source, _tag);

          // check if there is an error
          if (!req.success) {
            std::cout << "Error 6\n";
          }

          // allocate a buffer for the tensor
          auto rhs = _storage.create_tensor(req.num_bytes);

          // recieve the request and check if there is an error
          if (!_comm.recieve_request_sync(rhs, req)) {
            std::cout << "Error 5\n";
          }

          // how much do we need to allocated
          input_meta.set<0>(lhs->_meta);
          input_meta.set<1>(rhs->_meta);

          // get the meta data
          _reduce_op.get_out_meta(input_meta, output_meta);

          // set the format as get_out_meta is not responsble for doing that
          out_meta.fmt_id = id;

          // return the size of the tensor
          auto output_size = _factory.get_tensor_size(output_meta.get<0>());

          // allocate and init the output
          auto out = _storage.create_tensor(output_size);
          _factory.init_tensor(out, out_meta);

          // set the input tensors to the function
          input_tensors.set<0>(*lhs);
          input_tensors.set<1>(*rhs);

          // set the output tensor to the function
          output_tensor.set<0>(*out);

          // run the function
          _reduce_op.fn(input_tensors, output_tensor);

          // manage the memory
          if(lhs != &_in) {
              _storage.remove_by_tensor(*lhs);
          }
          _storage.remove_by_tensor(*rhs);
          
          // set the lhs
          lhs = out;
        }

      } else {

        // I've received all that I'm going to.  Send my result to my parent
        source = ((relrank & (~mask)) + lroot) % _comm.get_num_nodes();

        // return the size of the tensor
        auto num_bytes = _factory.get_tensor_size(lhs->_meta);

        // do the send and log the error if there was any
        if (!_comm.send_sync(lhs, num_bytes, source, _tag)) {        
            std::cout << "Error 4\n";
        }

        break;
      }
      mask <<= 1;
    }

    // the result is at the node with rank 0, we need to move it
    if (_root != 0) {

      // the node with rank 0 sends the node with the root rank recieves
      if (_comm.get_rank() == 0) {

        // send it to the root
        size_t num_bytes = _factory.get_tensor_size(lhs->_meta);
        if (!_comm.send_sync(lhs, num_bytes, _root, _tag)) {        
            std::cout << "Error 3 \n";
        }

      } else if (_comm.get_rank() == _root) {
        
          // wait for the message
          auto req = _comm.expect_request_sync(0, _tag);

          // check if there is an error
          if (!req.success) {
            std::cout << "Error 2 \n";
          }

          // manage the memory
          if(lhs != &_in) {
              std::cout << "Got here\n" << std::flush;
              _storage.remove_by_tensor(*lhs);
          }
          lhs = _storage.create_tensor(req.num_bytes);

          // check if there is an error
          if (!_comm.recieve_request_sync(lhs, req)) {
            std::cout << "Error 1 \n";
          }
      }
    }

    // free the lhs
    if(_comm.get_rank() != _root) {
      if(lhs != &_in) {
        _storage.remove_by_tensor(*lhs);
      }
      lhs = nullptr;
    }

    // return the reduced tensor
    return lhs;
  }
  
};

int main(int argc, char **argv) {

  // make the configuration
  auto config = std::make_shared<bbts::node_config_t>(bbts::node_config_t{.argc=argc, .argv = argv});

  // create the tensor factory
  bbts::tensor_factory_ptr_t factory = std::make_shared<bbts::tensor_factory_t>();

  // init the communicator with the configuration
  bbts::mpi_communicator_t comm(config);

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

  // initi the tensor
  auto &a = factory->init_tensor((bbts::tensor_t*) a_mem.get(), m).as<bbts::dense_tensor_t>();

  // get a reference to the metadata
  auto &am = a.meta().m();

  // init the matrix
  for(int i = 0; i < am.num_rows * am.num_cols; ++i) {
    a.data()[i] = 1 + comm.get_rank() + i;
  }

  // pick all the nodes with an even rank, for testing purpouses 
  std::vector<bbts::node_id_t> nodes;
  for(bbts::node_id_t i = 0; i < comm.get_num_nodes(); ++i) {
    nodes.push_back(i);
  }

  // craete the reduce
  auto reduce_op = reduce(comm, *factory, storage, nodes, 0, 111, a, *ud);
  auto b = reduce_op.apply();

  bool success = false;
  if(b != nullptr) {
  
    auto &bb = b->as<bbts::dense_tensor_t>();
    for(int i = 0; i < am.num_rows * am.num_cols; ++i) {
        
        int32_t val = 0;
        for(int rank = 0; rank < comm.get_num_nodes(); ++rank) {
          success = true;
          val += 1 + rank + i;
        }
        if(bb.data()[i] != val) {
          success = false;
          std::cout << "not ok " << val << " " << bb.data()[i] << '\n';
        }
    }
  }

  // wait for all
  comm.barrier();

  if(success) {
    std::cout << "all good\n";
    return 0;
  }

  return 1;
}