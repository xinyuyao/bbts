#pragma once

#include <assert.h>
#include "ud_function.h"
#include "../tensor/builtin_formats.h"

namespace bbts {

struct dense_matrix_add_t : public ud_impl_t {

  // initializes the function
  dense_matrix_add_t();

  // returns an estimate of the complexity
  size_t get_complexity_hint(const meta_params_t &_in) override;

  // return the meta of the output
  void get_out_meta(const meta_params_t &_in, meta_params_t &_out) override;

  // does the work
  static void add(const tensor_params_t &_in, tensor_params_t &_out);

};

}