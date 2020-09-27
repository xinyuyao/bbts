#pragma once

#include "../tensor/tensor.h"

namespace bbts {

// the identifier of the ud function
// this should be used whenever we have something that maps or identifies the ud function
using ud_id_t = int32_t;

// uniquely identifies the implementation
// within the ud function
using ud_impl_id_t = int32_t;

struct ud_impl_type_t {

  // the implementation name of this function this has to be unique
  // for example this could be mkl_matrix_multiplication or stressen_matrix_multiplication
  std::string impl_name;

  // the name of the ud function, this is the same for all the ud functions that create an equivalent result
  // for example matrix_multiplication, matrix_addition, etc..
  std::string ud_name;

  // does this implementation require the tensors to be on the gpu
  bool is_gpu;

  // this is a function pointer to the function that needs to be applied
  void *fun;
};

struct ud_func_t {

  // the name of the ud function, this is the same for all the ud functions that create an equivalent result
  // for example matrix_multiplication, matrix_addition, etc..
  std::string ud_name;

  // this tells us whether the ud function is associative, or not
  bool is_ass;

  // this tells us whether the ud function commutative, or not
  bool is_com;

  // this tells us how many tensors we need as an input to this function
  int32_t num_in;

  // this tells us how tensors this function is outputting
  int32_t num_out;

  // these are all the implementations of this ud function
  std::vector<ud_impl_type_t> impls;

  // the expression that will tell us the dimension of the output given the input
  virtual std::vector<tensor_meta_t> get_outputs_meta_for(const std::vector<tensor_meta_t> &inputs_meta) = 0;

  // applies the ud function
  virtual void apply(ud_impl_type_t &ud, const std::vector<tensor_t> &input, const std::vector<tensor_t> &output) = 0;
};


}