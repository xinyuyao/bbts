#include "ffnn_activation_mult.h"
#include "ffnn_types.h"
#include <cassert>
#include <mkl/mkl_cblas.h>
#include <mkl/mkl.h>

bbts::ffnn_activation_mult::ffnn_activation_mult() {

  // set the names
  impl_name = "ffnn_act_mult_cpu";
  ud_name = "ffnn_act_mult";

  // set the input and output types
  inputTypes = {"ffnn_dense", "ffnn_dense"};
  outputTypes = {"ffnn_dense"};

  // both inputs zero and one can be used as the inplace output
  inputInplace = {};

  // this is a CPU dense mult
  is_gpu = false;

  // set the function that actually performs the add
  fn = &ffnn_activation_mult::mult;
}

size_t bbts::ffnn_activation_mult::get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                                                      const bbts::ud_impl_t::meta_args_t &_in) {

  // O(n * m * k)
  const auto &m_a = _in.get<0>().as<ffnn_dense_meta_t>().m();
  const auto &m_b = _in.get<1>().as<ffnn_dense_meta_t>().m();
  return m_a.num_rows * m_a.num_cols * m_b.num_cols;
}

void bbts::ffnn_activation_mult::get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                                              const bbts::ud_impl_t::meta_args_t &_in,
                                              bbts::ud_impl_t::meta_args_t &_out) const {

  // get the input argeters
  const auto &m_a = _in.get<0>().as<ffnn_dense_meta_t>().m();
  const auto &m_b = _in.get<1>().as<ffnn_dense_meta_t>().m();

  // get the output argeters
  auto &m_out = _out.get<0>().as<ffnn_dense_meta_t>().m();

  // set the output
  uint32_t I = m_a.num_rows;
  uint32_t J = m_b.num_cols;
  m_out = {.num_rows = I, .num_cols = J, .has_bias = false, .num_aggregated = 1};

  auto num_elements = m_out.num_cols * m_out.num_rows;
  num_elements += m_out.has_bias ? m_out.num_cols : 0;
}

void bbts::ffnn_activation_mult::mult(const bbts::ud_impl_t::tensor_params_t &params,
                                     const bbts::ud_impl_t::tensor_args_t &_in,
                                     bbts::ud_impl_t::tensor_args_t &_out) {

  // get the tensors as dense tensors
  auto &a = _in.get<0>().as<ffnn_dense_t>();
  auto &b = _in.get<1>().as<ffnn_dense_t>();
  auto &out = _out.get<0>().as<ffnn_dense_t>();

  // get the meta for the tensors
  auto &m_a = a.meta().m();
  auto &m_b = b.meta().m();
  auto &m_out = out.meta().m();

  // get the sizes
  uint32_t I = m_a.num_rows;
  uint32_t J = m_b.num_cols;
  uint32_t K = m_a.num_cols;

  // make sure the matrix size matches, this is only present during the debug build
  assert(m_a.num_cols == m_b.num_rows);
  assert(m_b.has_bias);

  // get the ptrs
  float *outData = out.data();
  float *in1Data = a.data();
  float *in2Data = b.data();

  // set the new meta data
  m_out = {.num_rows = I, .num_cols = J, .has_bias = false, .num_aggregated = 1};

  // do the multiply
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, I, J, K, 1.0f, in1Data, m_a.num_cols, in2Data, m_b.num_cols, 0.0f, outData, J);

  // add the bias
  for (auto row = 0; row < I; ++row) {
    for (auto col = 0; col < J; ++col) {
      outData[row * J + col] += b.bias()[col];
    }
  }
}