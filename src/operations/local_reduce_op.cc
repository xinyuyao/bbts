#include "local_reduce_op.h"

bbts::local_reduce_op_t::local_reduce_op_t(bbts::tensor_factory_t &_factory,
                                           bbts::storage_t &_storage,
                                           bbts::tensor_stats_t &_stats,
                                           const std::vector<bbts::tensor_t *> &_inputs,
                                           const ud_impl_t::tensor_params_t &_params,
                                           bbts::tid_t _out_tid,
                                           const bbts::ud_impl_t &_reduce_op) : _factory(_factory),
                                                                                _storage(_storage),
                                                                                _stats(_stats),
                                                                                _inputs(_inputs),
                                                                                _params(_params),
                                                                                _out_tid(_out_tid),
                                                                                _reduce_op(_reduce_op),
                                                                                _input_tensors({nullptr, nullptr}),
                                                                                _output_tensor({nullptr}),
                                                                                _input_meta({nullptr, nullptr}),
                                                                                _output_meta({&_out_meta})  {}


bbts::tensor_t* bbts::local_reduce_op_t::apply() {

  // is the output on the gpu (is this a gpu ud function)
  bool is_gpu = _stats.is_gpu(_out_tid);

  // get the first left side
  bbts::tensor_t *lhs = _inputs.front();
  for(size_t idx = 1; idx < _inputs.size(); ++idx) {

    /// 1.1 get the right side tensor

    // get the other side
    bbts::tensor_t *rhs = _inputs[idx];

    // how much do we need to allocated
    _input_meta.set<0>(lhs->_meta);
    _input_meta.set<1>(rhs->_meta);

    // get the meta data
    _reduce_op.get_out_meta(_params, _input_meta, _output_meta);

    /// 1.2 allocate the output tensor

    // return the size of the tensor
    auto output_size = _factory.get_tensor_size(_output_meta.get<0>());

    // allocate and init the output
    auto out = _storage.create_tensor(output_size, is_gpu);
    _factory.init_tensor(out, _out_meta);

    /// 1.3 run the ud function

    // set the input tensors to the function
    _input_tensors.set<0>(*lhs);
    _input_tensors.set<1>(*rhs);

    // set the output tensor to the function
    _output_tensor.set<0>(*out);

    // run the function
    _reduce_op.call_ud(_params, _input_tensors, _output_tensor);

    /// 1.4 deallocate the previous output tensor and swap

    // remove additionally every allocated tensor
    if(idx != 1) {
      _storage.remove_by_tensor(*lhs);
    }

    // set the output as lhs
    lhs = out;
  }

  // assign a tid to the result of the aggregation
  _storage.assign_tid(*lhs, _out_tid);

  // return the tensor
  return lhs;
}