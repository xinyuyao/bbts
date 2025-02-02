#include "ffnn_mult.h"
#include "ffnn_types.h"
#include <mkl/mkl_cblas.h>
#include <mkl/mkl.h>

bbts::ffnn_mult::ffnn_mult() {

  // set the names
  impl_name = "ffnn_mult_cpu";
  ud_name = "ffnn_mult";

  // set the input and output types
  inputTypes = {"ffnn_dense", "ffnn_dense"};
  outputTypes = {"ffnn_dense"};

  // both inputs zero and one can be used as the inplace output
  inputInplace = {};

  // this is a CPU dense mult
  is_gpu = false;

  // set the function that actually performs the add
  fn = &ffnn_mult::mult;
}

size_t bbts::ffnn_mult::get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                                                      const bbts::ud_impl_t::meta_args_t &_in) {

  // O(n * m * k)
  const auto &m_a = _in.get<0>().as<ffnn_dense_meta_t>().m();
  const auto &m_b = _in.get<1>().as<ffnn_dense_meta_t>().m();
  return m_a.num_rows * m_a.num_cols * m_b.num_cols;
}

void bbts::ffnn_mult::get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                                             const bbts::ud_impl_t::meta_args_t &_in,
                                             bbts::ud_impl_t::meta_args_t &_out) const {

  // get the input argeters
  const auto &m_a = _in.get<0>().as<ffnn_dense_meta_t>().m();
  const auto &m_b = _in.get<1>().as<ffnn_dense_meta_t>().m();

  // get the output argeters
  auto &m_out = _out.get<0>().as<ffnn_dense_meta_t>().m();

  // get the sizes
  uint32_t I = !params.get_bool_or_default<0>(false) ? m_a.num_rows : m_a.num_cols;
  uint32_t J = !params.get_bool_or_default<1>(false) ? m_b.num_cols : m_b.num_rows;

  // set the output
  m_out = {I, J, false};
}

void bbts::ffnn_mult::mult(const bbts::ud_impl_t::tensor_params_t &params,
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
  uint32_t I = !params.get_bool_or_default<0>(false) ? m_a.num_rows : m_a.num_cols;
  uint32_t J = !params.get_bool_or_default<1>(false) ? m_b.num_cols : m_b.num_rows;

  // get the inner dimensions
  uint32_t K1 = !params.get_bool_or_default<0>(false) ? m_a.num_cols : m_a.num_rows;
  uint32_t K2 = !params.get_bool_or_default<1>(false) ? m_a.num_rows : m_a.num_cols;

  // make sure the matrix size matches, this is only present during the debug build
  assert(K1 == K2);

  // get the ptrs
  float *outData = out.data();
  float *in1Data = a.data();
  float *in2Data = b.data();

  // figure out if we need to transpose
  CBLAS_TRANSPOSE l_trans = params.get_bool_or_default<0>(false) ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE r_trans = params.get_bool_or_default<1>(false) ? CblasTrans : CblasNoTrans;

  // do the multiply
  cblas_sgemm(CblasRowMajor, l_trans, r_trans, I, J, K1, 1.0f, in1Data, m_a.num_cols, in2Data, m_b.num_cols, 0.0f, outData, J);

  // set the new meta data
  m_out = {I, J, false};
}