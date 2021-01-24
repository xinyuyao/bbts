#include "reduce_op.h"

namespace bbts {

reduce_op_t::reduce_op_t(bbts::communicator_t &_comm, bbts::tensor_factory_t &_factory, bbts::storage_t &_storage,
                         const bbts::command_t::node_list_t &_nodes, int32_t _tag,
                         bbts::tensor_t &_in, bbts::tid_t _out_tid, const bbts::ud_impl_t &_reduce_op) : _comm(_comm),
                                                                                                         _factory(_factory),
                                                                                                         _storage(_storage),
                                                                                                         _nodes(_nodes),
                                                                                                         _tag(_tag),
                                                                                                         _out_tid(_out_tid),
                                                                                                         _in(_in),
                                                                                                         _reduce_op(_reduce_op) {}

int32_t reduce_op_t::get_num_nodes() const {
  return _nodes.size();
}

int32_t reduce_op_t::get_local_rank() const {

  auto it = _nodes.find(_comm.get_rank());
  return it.distance_from(_nodes.begin());
}

int32_t reduce_op_t::get_global_rank(int32_t local_rank) const {
  return _nodes[local_rank];
}

bbts::tensor_t *reduce_op_t::apply() {

  int32_t mask = 0x1;
  int32_t lroot = 0;

  // relative rank
  int32_t vrank = (get_local_rank() - lroot + get_num_nodes()) % get_num_nodes();

  // get the lhs address
  bbts::tensor_t *lhs = &_in;

  // make empty input parameter
  bbts::tensor_meta_t out_meta{};
  bbts::ud_impl_t::tensor_params_t input_tensors({nullptr, nullptr});
  bbts::ud_impl_t::tensor_params_t output_tensor({nullptr});
  bbts::ud_impl_t::meta_params_t input_meta({nullptr, nullptr});
  bbts::ud_impl_t::meta_params_t output_meta({&out_meta});

  // get the impl_id of the output
  auto id = _factory.get_tensor_ftm(_reduce_op.outputTypes.front());

  // do stuff
  int32_t source;
  while (mask < get_num_nodes()) {

    // receive 
    if ((mask & vrank) == 0) {
      
      source = (vrank | mask);
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
        if (!_comm.receive_request_sync(rhs, req)) {
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
      source = ((vrank & (~mask)) + lroot) % get_num_nodes();

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

  // free the lhs
  if(get_local_rank() != 0) {
    if(lhs != &_in) {
      _storage.remove_by_tensor(*lhs);
    }
    lhs = nullptr;
  }
  else {

    // assign a tid to the result of the aggregation
    _storage.assign_tid(*lhs, _out_tid);
  }

  // return the reduced tensor
  return lhs;
}

}