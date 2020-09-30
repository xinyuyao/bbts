#pragma once

#include "../tensor/tensor.h"

namespace bbts {

// the identifier of the ud function
// this should be used whenever we have something that maps or identifies the ud function
using ud_id_t = int32_t;

// uniquely identifies the implementation
// within the ud function
using ud_impl_id_t = int32_t;

struct ud_impl_t {

  // the ud function parameters for either the input or the output
  template<class T>
  struct ud_impl_params_t {

    // the input parameters
    ud_impl_params_t(std::vector<T*> in_param) : parameters(std::move(in_param)) {}

    // this is for the output parameters
    template<size_t n>
    T &get() { return *parameters[n]; }

    // this is for the input parameters as they are constant
    template<size_t n>
    const T &get() const { return *parameters[n]; }

   private:

    // holds the input parameters
    std::vector<T*> parameters;
  };

  // define the parameters for the meta
  using tensor_params_t = ud_impl_params_t<tensor_t>;
  using meta_params_t = ud_impl_params_t<tensor_meta_t>;

  // the implementation name of this function this has to be unique
  // for example this could be mkl_matrix_multiplication or stressen_matrix_multiplication
  std::string impl_name;

  // the name of the ud function, this is the same for all the ud functions that create an equivalent result
  // for example matrix_multiplication, matrix_addition, etc..
  std::string ud_name;

  // the input types of the tensors
  std::vector<std::string> inputTypes;

  // tells the system what inputs can be inplace
  std::vector<bool> inputInplace;

  // the output types
  std::string outputType;

  // does this implementation require the tensors to be on the gpu
  bool is_gpu;

  // this is a function pointer to the function that needs to be applied
  std::function<void(const tensor_params_t &_in, tensor_params_t &_out)> fn;

  // returns the complexity hint of the ud function
  virtual size_t get_complexity_hint(const meta_params_t &_in) = 0;

  // returns the output meta data
  virtual void get_out_meta(const meta_params_t &_in, meta_params_t &_out) = 0;
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
  std::vector<ud_impl_t> impls;

  // the expression that will tell us the size of the
  virtual std::vector<tensor_meta_t> get_outputs_meta_for(const std::vector<tensor_meta_t> &inputs_meta) = 0;

  // applies the ud function
  virtual void apply(ud_impl_t &ud, const std::vector<tensor_t> &input, const std::vector<tensor_t> &output) = 0;
};


}