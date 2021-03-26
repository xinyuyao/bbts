#include "ud_function.h"

void bbts::ud_impl_t::call_ud(const bbts::ud_impl_t::tensor_params_t &_params,
                              const tensor_args_t &_in,
                              tensor_args_t &_out) const {

    // jut call the function
    fn(_params, _in, _out);
}