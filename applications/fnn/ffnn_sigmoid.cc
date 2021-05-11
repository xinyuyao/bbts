#include "ffnn_sigmoid.h"
#include "ffnn_types.h"
#include <cmath>

bbts::ffnn_sigmoid::ffnn_sigmoid() {

  // set the names
  impl_name = "ffnn_sigmoid_cpu";
  ud_name = "ffnn_sigmoid";

  // set the input and output types
  inputTypes = {"ffnn_dense"};
  outputTypes = {"ffnn_dense"};

  // both inputs zero and one can be used as the inplace output
  inputInplace = {0};

  // this is a CPU dense add
  is_gpu = false;

  // set the function that actually performs the add
  fn = &ffnn_sigmoid::sigmoid;
}

size_t bbts::ffnn_sigmoid::get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                                                     const bbts::ud_impl_t::meta_args_t &_in) {

  // O(n * m)
  const auto &m_a = _in.get<0>().as<ffnn_dense_meta_t>().m();
  return m_a.num_rows * m_a.num_cols;
}

void bbts::ffnn_sigmoid::get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                                            const bbts::ud_impl_t::meta_args_t &_in,
                                            bbts::ud_impl_t::meta_args_t &_out) const {

  // get the input argeters
  const auto &m_a = _in.get<0>().as<ffnn_dense_meta_t>().m();

  // get the output argeters
  auto &m_out = _out.get<0>().as<ffnn_dense_meta_t>().m();

  // set the new values
  m_out = { m_a.num_rows, m_a.num_cols, false};
}

void bbts::ffnn_sigmoid::sigmoid(const bbts::ud_impl_t::tensor_params_t &params,
                                         const bbts::ud_impl_t::tensor_args_t &_in,
                                         bbts::ud_impl_t::tensor_args_t &_out) {

  // get the tensors as dense tensors
  auto &a = _in.get<0>().as<ffnn_dense_t>();
  auto &out = _out.get<0>().as<ffnn_dense_t>();

  // get the meta for the tensors
  auto &m_a = a.meta().m();
  auto &m_out = out.meta().m();

  // 
  float *out_data =  out.data();
  float *a_data   =  a.data();

  // 
  for (auto row = 0; row < m_a.num_rows; ++row) {
    for (auto col = 0; col < m_a.num_cols; ++col) {
      auto v = a_data[row * m_a.num_cols + col];
      out_data[row * m_a.num_cols + col] = 1 / (1.0f + std::exp(v));
    }
  }

  // set the new meta data
  m_out = {m_a.num_rows, m_a.num_cols, false};
}
