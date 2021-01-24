#pragma once

#include <assert.h>
#include "ud_function.h"
#include "../tensor/builtin_formats.h"

namespace bbts {

/// 1. Matrix addition
ud_func_ptr_t get_matrix_add_udf();

struct dense_matrix_add_t : public ud_impl_t {

  // initializes the function
  dense_matrix_add_t();

  // returns an estimate of the complexity
  size_t get_complexity_hint(const meta_args_t &_in) override;

  // return the meta of the output
  void get_out_meta(const meta_args_t &_in, meta_args_t &_out) const override;

  // does the work
  static void add(const tensor_args_t &_in, tensor_args_t &_out);

};

/// 2. Matrix multiply
ud_func_ptr_t get_matrix_mult_udf();

struct dense_matrix_mult_t : public ud_impl_t {

  // initializes the function
  dense_matrix_mult_t();

  // returns an estimate of the complexity
  size_t get_complexity_hint(const meta_args_t &_in) override;

  // return the meta of the output
  void get_out_meta(const meta_args_t &_in, meta_args_t &_out) const override;

  // does the work
  static void mult(const tensor_args_t &_in, tensor_args_t &_out);

};

}