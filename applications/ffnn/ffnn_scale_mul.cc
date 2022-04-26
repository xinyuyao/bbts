#include "ffnn_types.h"
#include "ffnn_scale_mul.h"
#include <cassert>

bbts::ffnn_scale_mul::ffnn_scale_mul() {

  // set the names
  impl_name = "ffnn_scale_mul_cpu";
  ud_name = "ffnn_scale_mul";

  // set the input and output types
  inputTypes = {"ffnn_dense"};
  outputTypes = {"ffnn_dense"};

  // both inputs zero and one can be used as the inplace output
  inputInplace = {0, 1};

  // this is a CPU dense add
  is_gpu = false;

  // set the function that actually performs the add
  fn = &ffnn_scale_mul::scale_mul;
}

size_t bbts::ffnn_scale_mul::get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                                                     const bbts::ud_impl_t::meta_args_t &_in) {

  // O(n * m)
  const auto &m_a = _in.get<0>().as<ffnn_dense_meta_t>().m();
  return 5.91992e-10 * m_a.num_rows * m_a.num_cols;
}

void bbts::ffnn_scale_mul::get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                                            const bbts::ud_impl_t::meta_args_t &_in,
                                            bbts::ud_impl_t::meta_args_t &_out) const {

  // get the input argeters
  const auto &m_a = _in.get<0>().as<ffnn_dense_meta_t>().m();

  // get the output argeters
  auto &m_out = _out.get<0>().as<ffnn_dense_meta_t>().m();

  // set the new values
  m_out = {.num_rows = m_a.num_rows, 
           .num_cols = m_a.num_cols, 
           .row_idx = m_a.row_idx,
           .col_idx = m_a.col_idx,
           .has_bias = m_a.has_bias, 
           .num_aggregated = 1};
}

void bbts::ffnn_scale_mul::scale_mul(const bbts::ud_impl_t::tensor_params_t &params,
                                  const bbts::ud_impl_t::tensor_args_t &_in,
                                  bbts::ud_impl_t::tensor_args_t &_out) {

  // get the constants
  auto ca = params.get_float_or_default<0>(1.0f);

  // get the tensors as dense tensors
  auto &a = _in.get<0>().as<ffnn_dense_t>();
  auto &out = _out.get<0>().as<ffnn_dense_t>();

  // get the meta for the tensors
  auto &m_a = a.meta().m();
  auto &m_out = out.meta().m();

  // make sure the matrix size matches, this is only
  // present during the debug build
//   assert(m_a.num_rows == m_b.num_rows);
//   assert(m_a.num_cols == m_b.num_cols);
//   assert(m_a.has_bias == m_b.has_bias);

  // add a and b
  for (auto row = 0; row < m_a.num_rows; ++row) {
    for (auto col = 0; col < m_a.num_cols; ++col) {
      out.data()[row * m_a.num_cols + col] = ca * a.data()[row * m_a.num_cols + col];
    }
  }

  // set the new meta data
  m_out = {.num_rows = m_a.num_rows, 
           .num_cols = m_a.num_cols, 
           .row_idx = m_a.row_idx,
           .col_idx = m_a.col_idx,
           .has_bias = m_a.has_bias, 
           .num_aggregated = 1};

  // sum their biases if they exists
  if(m_a.has_bias) {
    for (auto col = 0; col < m_a.num_cols; ++col) {
      out.bias()[col] = ca * a.bias()[col];
    }
  }
}
