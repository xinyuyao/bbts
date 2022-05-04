#include "ffnn_sigmoid.h"
#include <cmath>
#include <cstdint>

bbts::ffnn_sigmoid::ffnn_sigmoid() {

  // set the names
  impl_name = "ffnn_sigmoid_cpu";
  ud_name = "ffnn_sigmoid";

  // set the input and output types
  inputTypes = {"ffnn_dense"};
  outputTypes = {"ffnn_dense"};

  // both inputs zero and one can be used as the inplace output
  inputInplace = {0,1};

  // this is a CPU dense add
  is_gpu = false;

  // set the function that actually performs the sidmoid
  fn = &ffnn_sigmoid::apply_sigmoid;
}

// TODO: how to get complexity hint for sidmoid
size_t bbts::ffnn_sigmoid::get_complexity_hint(
    const bbts::ud_impl_t::tensor_params_t &params,
    const bbts::ud_impl_t::meta_args_t &_in) {

  // O(n * m)
  const auto &m_a = _in.get<0>().as<ffnn_dense_meta_t>().m();
  return m_a.num_rows * m_a.num_cols;
}

void bbts::ffnn_sigmoid::get_out_meta(
    const bbts::ud_impl_t::tensor_params_t &params,
    const bbts::ud_impl_t::meta_args_t &_in,
    bbts::ud_impl_t::meta_args_t &_out) const {

  // get the input argeters
  const auto &m_a = _in.get<0>().as<ffnn_dense_meta_t>().m();

  // get the output argeters
  auto &m_out = _out.get<0>().as<ffnn_dense_meta_t>().m();

  // set the new values
  bool has_bias = m_a.has_bias;
  m_out = {.num_rows = m_a.num_rows, 
           .num_cols = m_a.num_cols, 
           .row_idx = m_a.row_idx,
           .col_idx = m_a.col_idx,
           .has_bias = has_bias, 
           .num_aggregated = 1};
}

void bbts::ffnn_sigmoid::apply_sigmoid(const bbts::ud_impl_t::tensor_params_t &params,
                         const bbts::ud_impl_t::tensor_args_t &_in,
                         bbts::ud_impl_t::tensor_args_t &_out) {
  // get the number of
  auto num_to_reduce = params.get_int_or_default<0>(-1);
  auto elementwise_op = (elementwise_fn_type)params.get_int_or_default<1>(
      (int32_t)elementwise_fn_type::NOOP);

  // get the tensors as dense tensors
  auto &a = _in.get<0>().as<ffnn_dense_t>();
  auto &out = _out.get<0>().as<ffnn_dense_t>();

  // get the meta for the tensors
  auto &m_a = a.meta().m();
  auto &m_out = out.meta().m();

  // set the new meta data
  m_out.num_rows = m_a.num_rows;
  m_out.num_cols = m_a.num_cols;
  m_out.has_bias = m_a.has_bias;
  m_out.num_aggregated = m_a.num_aggregated;
  m_out.row_idx = m_a.row_idx;
  m_out.col_idx = m_a.col_idx;
  
  // make sure the matrix size matches, this is only
  // present during the debug build
  // assert(m_a.num_rows == m_b.num_rows);
  // assert(m_a.num_cols == m_b.num_cols);
  // assert(m_a.has_bias == m_b.has_bias);

/************************** switching the operation calculating here to collecting linearge ******************/
  // sigmoid(a)
  auto out_data = out.data();
  for (auto row = 0; row < m_a.num_rows; ++row) {
    for (auto col = 0; col < m_a.num_cols; ++col) {
      // std::cout << row * m_a.num_cols + col << "\n" << std::flush;
      auto tmp_a = a.data()[row * m_a.num_cols + col];
  

      out_data[row * m_a.num_cols + col] = 1 / (1.0f + std::exp(-tmp_a));
    }
  }

  if (m_a.has_bias) {
    for (auto col = 0; col < m_a.num_cols; ++col) {
      float ta = a.bias()[col];
      out.bias()[col] = ta;
    }
  }

  // ok we need to a
  // if (out.meta().m().num_aggregated == num_to_reduce) {
  //   if (elementwise_op == elementwise_fn_type::RELU) {
  //     relu(out);
  //   } else if (elementwise_op == elementwise_fn_type::SIGMOID) {
  //     sigmoid(out);
  //   }
  // }
}

void bbts::ffnn_sigmoid::sigmoid(ffnn_dense_t &_out) {
  auto num_rows = _out.meta().m().num_rows;
  auto num_cols = _out.meta().m().num_cols;
  auto out = _out.data();

  for (auto row = 0; row < num_rows; ++row) {
    for (auto col = 0; col < num_cols; ++col) {
      auto v = out[row * num_cols + col];
      out[row * num_cols + col] = 1 / (1.0f + std::exp(-v));
    }
  }
}

void bbts::ffnn_sigmoid::relu(ffnn_dense_t &_out) {

  auto num_rows = _out.meta().m().num_rows;
  auto num_cols = _out.meta().m().num_cols;
  auto out = _out.data();

  for (auto row = 0; row < num_rows; ++row) {
    for (auto col = 0; col < num_cols; ++col) {
      out[row * num_cols + col] =
          out[row * num_cols + col] > 0 ? out[row * num_cols + col] : 0;
    }
  }
}
