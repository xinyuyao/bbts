#pragma once

#include <cassert>
#include "../../src/ud_functions/ud_function.h"

namespace bbts {

struct ffnn_scale_mul : public ud_impl_t {

  // initializes the function
  ffnn_scale_mul();

  // returns an estimate of the complexity
  size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                             const meta_args_t &_in) override;

  // return the meta of the output
  void get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                    const meta_args_t &_in, meta_args_t &_out) const override;

  // does the work
  static void scale_mul(const bbts::ud_impl_t::tensor_params_t &params,
                  const tensor_args_t &_in, tensor_args_t &_out);

};

}