#include "builtin_functions.h"
#include <mkl/mkl_cblas.h>
#include <mkl/mkl.h>

/// 1. Matrix Add
bbts::ud_func_ptr_t bbts::get_matrix_add_udf() {
  return std::make_unique<ud_func_t>(
      ud_func_t {
          .ud_name = "matrix_add",
          .is_ass = true,
          .is_com = true,
          .num_in = 2,
          .num_out = 1,
          .impls = {},
      }
  );
}

bbts::dense_matrix_add_t::dense_matrix_add_t() {

  // set the names
  impl_name = "dense_matrix_add";
  ud_name = "matrix_add";

  // set the input and output types
  inputTypes = {"dense", "dense"};
  outputTypes = {"dense"};

  // both inputs zero and one can be used as the inplace output
  inputInplace = {0, 1};

  // this is a CPU dense add
  is_gpu = false;

  // set the function that actually performs the add
  fn = &dense_matrix_add_t::add;
}

size_t bbts::dense_matrix_add_t::get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                                                     const bbts::ud_impl_t::meta_args_t &_in) {

  // O(n * m)
  const auto &m_a = _in.get<0>().as<dense_tensor_meta_t>().m();
  return m_a.num_rows * m_a.num_cols;
}

void bbts::dense_matrix_add_t::get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                                            const bbts::ud_impl_t::meta_args_t &_in,
                                            bbts::ud_impl_t::meta_args_t &_out) const {

  // get the input argeters
  const auto &m_a = _in.get<0>().as<dense_tensor_meta_t>().m();

  // get the output argeters
  auto &m_out = _out.get<0>().as<dense_tensor_meta_t>().m();

  // set the new values
  m_out = { m_a.num_rows, m_a.num_cols };
}

void bbts::dense_matrix_add_t::add(const bbts::ud_impl_t::tensor_params_t &params,
                                   const bbts::ud_impl_t::tensor_args_t &_in,
                                   bbts::ud_impl_t::tensor_args_t &_out) {

  // get the tensors as dense tensors
  auto &a = _in.get<0>().as<dense_tensor_t>();
  auto &b = _in.get<1>().as<dense_tensor_t>();
  auto &out = _out.get<0>().as<dense_tensor_t>();

  // get the meta for the tensors
  auto &m_a = a.meta().m();
  auto &m_b = b.meta().m();
  auto &m_out = out.meta().m();

  // make sure the matrix size matches, this is only
  // present during the debug build
  assert(m_a.num_rows == m_b.num_rows);
  assert(m_a.num_cols == m_b.num_cols);
  assert(m_a.num_rows == m_out.num_rows);
  assert(m_a.num_cols == m_out.num_cols);

  // add a and b
  for (auto row = 0; row < m_a.num_rows; ++row) {
    for (auto col = 0; col < m_a.num_cols; ++col) {
      out.data()[row * m_a.num_cols + col] = a.data()[row * m_a.num_cols + col] +
          b.data()[row * m_a.num_cols + col];
    }
  }

  // set the new meta data
  m_out = {m_a.num_rows, m_a.num_cols};
}

/// 2. Matrix Multiply
bbts::ud_func_ptr_t bbts::get_matrix_mult_udf() {
  return std::make_unique<ud_func_t>(
      ud_func_t {
          .ud_name = "matrix_mult",
          .is_ass = true,
          .is_com = false,
          .num_in = 2,
          .num_out = 1,
          .impls = {},
      }
  );
}
bbts::dense_matrix_mult_t::dense_matrix_mult_t() {

  // set the names
  impl_name = "dense_matrix_mult";
  ud_name = "matrix_mult";

  // set the input and output types
  inputTypes = {"dense", "dense"};
  outputTypes = {"dense"};

  // both inputs zero and one can be used as the inplace output
  inputInplace = {0, 0};

  // this is a CPU dense add
  is_gpu = false;

  // set the function that actually performs the add
  fn = &dense_matrix_mult_t::mult;
}

size_t bbts::dense_matrix_mult_t::get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                                                      const bbts::ud_impl_t::meta_args_t &_in) {

  // O(n * m * k)
  const auto &m_a = _in.get<0>().as<dense_tensor_meta_t>().m();
  const auto &m_b = _in.get<1>().as<dense_tensor_meta_t>().m();
  return m_a.num_rows * m_a.num_cols * m_b.num_cols;
}

void bbts::dense_matrix_mult_t::get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                                             const bbts::ud_impl_t::meta_args_t &_in,
                                             bbts::ud_impl_t::meta_args_t &_out) const {

  // get the input argeters
  const auto &m_a = _in.get<0>().as<dense_tensor_meta_t>().m();
  const auto &m_b = _in.get<1>().as<dense_tensor_meta_t>().m();

  // get the output argeters
  auto &m_out = _out.get<0>().as<dense_tensor_meta_t>().m();

  // set the output
  m_out = {m_a.num_rows, m_b.num_cols};
}

void bbts::dense_matrix_mult_t::mult(const bbts::ud_impl_t::tensor_params_t &params,
                                     const bbts::ud_impl_t::tensor_args_t &_in,
                                     bbts::ud_impl_t::tensor_args_t &_out) {

  // get the tensors as dense tensors
  auto &a = _in.get<0>().as<dense_tensor_t>();
  auto &b = _in.get<1>().as<dense_tensor_t>();
  auto &out = _out.get<0>().as<dense_tensor_t>();

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

  // get the ptrs
  float *outData = out.data();
  float *in1Data = a.data();
  float *in2Data = b.data();

  // do the multiply
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, I, J, K, 1.0f, in1Data, K, in2Data, J, 0.0f, outData, J);

  // set the new meta data
  m_out = {m_a.num_rows, m_b.num_cols};
}

bbts::ud_func_ptr_t bbts::get_matrix_uniform_udf() {
  return std::make_unique<bbts::ud_func_t>(
      bbts::ud_func_t {
          .ud_name = "uniform",
          .is_ass = false,
          .is_com = false,
          .num_in = 0,
          .num_out = 1,
          .impls = {},
      }
  );
}

bbts::dense_uniform_t::dense_uniform_t() {

  // set the names
  impl_name = "dense_uniform";
  ud_name = "uniform";

  // set the input and output types
  inputTypes = {};
  outputTypes = {"dense"};

  // both inputs zero and one can be used as the inplace output
  inputInplace = {};

  // this is a CPU dense add
  is_gpu = false;

  // set the function that actually performs the add
  fn = &dense_uniform_t::uniform_rand;
}

size_t bbts::dense_uniform_t::get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                                                  const bbts::ud_impl_t::meta_args_t &_in) {

  // make sure that there are enough parameters
  if(params.num_parameters() < 2){
    throw std::runtime_error("Not enough parameters");
  }

  // O(n * m)
  return params.get_uint<0>() * params.get_uint<1>();
}

void bbts::dense_uniform_t::get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                                         const bbts::ud_impl_t::meta_args_t &_in,
                                         bbts::ud_impl_t::meta_args_t &_out) const {

  // get the output argeters
  auto &m_out = _out.get<0>().as<dense_tensor_meta_t>().m();

  // set the new values
  m_out = { params.get_uint<0>(),  params.get_uint<1>() };
}

void bbts::dense_uniform_t::uniform_rand(const bbts::ud_impl_t::tensor_params_t &params,
                                         const bbts::ud_impl_t::tensor_args_t &_in,
                                         bbts::ud_impl_t::tensor_args_t &_out) {


  // make the random stream
  VSLStreamStatePtr stream;
  vslNewStream(&stream, VSL_BRNG_MCG31, time(nullptr));

  // get the dense tensor
  auto &out = _out.get<0>().as<dense_tensor_t>();
  auto &m_out = out.meta().m();

  // the number of rows and columns
  auto numRows = params.get_uint<0>();
  auto numCols = params.get_uint<1>();

  // the left and right boundary
  auto left = params.get_float_or_default<2>(0.0f);
  auto right = params.get_float_or_default<3>(1.0f);

  // set the new meta data
  m_out = {.num_rows = numRows, .num_cols = numCols};

  // create a bunch of random numbers
  vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, (int32_t) (numRows * numCols), out.data(), left, right);

  // delete the stream
  vslDeleteStream(&stream);
}
