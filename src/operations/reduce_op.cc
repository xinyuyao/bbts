#include "reduce_op.h"

namespace bbts {

// constructs the reduce operation
reduce::reduce(bbts::mpi_communicator_t &_comm, bbts::tensor_factory_t &_factory, 
               bbts::storage_t &_storage, const std::vector<bbts::node_id_t> &_nodes,
               int32_t _root, int32_t _tag, bbts::tensor_t &_in, const bbts::ud_impl_t &_reduce_op) : _comm(_comm), 
                                                                                                      _factory(_factory),
                                                                                                      _storage(_storage),
                                                                                                      _nodes(_nodes),
                                                                                                      _root(_root),
                                                                                                      _tag(_tag),
                                                                                                      _in(_in),
                                                                                                      _reduce_op(_reduce_op) {}

// get the number of nodes
int32_t reduce::get_num_nodes() const {
  return _nodes.size();
}

// get local rank
int32_t reduce::get_local_rank() const {
  return std::distance(_nodes.begin(), std::find(_nodes.begin(), _nodes.end(), _comm.get_rank()));
}

// get global rank
int32_t reduce::get_global_rank(int32_t local_rank) const {
  return _nodes[local_rank];
}

bbts::tensor_t *reduce::apply() {

  int32_t mask = 0x1;
  int32_t lroot = 0;

  // relative rank
  int32_t relrank = (get_local_rank() - lroot + get_num_nodes()) % get_num_nodes();

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
  while (mask < get_num_nodes()) {

    // receive 
    if ((mask & relrank) == 0) {
      
      source = (relrank | mask);
      if (source < get_num_nodes()) {

        // wait till we get a message from the right node
        source = (source + lroot) % get_num_nodes();

        // try to get the request
        auto req = _comm.expect_request_sync(get_global_rank(source), _tag);

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
      source = ((relrank & (~mask)) + lroot) % get_num_nodes();

      // return the size of the tensor
      auto num_bytes = _factory.get_tensor_size(lhs->_meta);

      // do the send and log the error if there was any
      if (!_comm.send_sync(lhs, num_bytes, get_global_rank(source), _tag)) {        
          std::cout << "Error 4\n";
      }

      break;
    }
    mask <<= 1;
  }

  // the result is at the node with rank 0, we need to move it
  if (_root != 0) {

    // the node with rank 0 sends the node with the root rank recieves
    if (get_local_rank() == 0) {

      // send it to the root
      size_t num_bytes = _factory.get_tensor_size(lhs->_meta);
      if (!_comm.send_sync(lhs, num_bytes, get_global_rank(_root), _tag)) {        
          std::cout << "Error 3 \n";
      }

    } else if (get_local_rank() == _root) {
      
        // wait for the message
        auto req = _comm.expect_request_sync(get_global_rank(0), _tag);

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
  if(get_local_rank() != _root) {
    if(lhs != &_in) {
      _storage.remove_by_tensor(*lhs);
    }
    lhs = nullptr;
  }

  // return the reduced tensor
  return lhs;
}

}