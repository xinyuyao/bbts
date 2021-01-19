#pragma once

#include <memory>
#include "../tensor/tensor.h"

namespace bbts {

// the identifier of the ud function
// this should be used whenever we have something that maps or identifies the ud function
using ud_id_t = int32_t;

// uniquely identifies the implementation
using ud_impl_local_t = int32_t;
struct ud_impl_id_t {

  // the ud function identifier
  ud_id_t ud_id;

  // identifies the implementation withing the ud function
  ud_impl_local_t impl_id;
};


struct ud_impl_t {

  // the ud function parameters for either the input or the output
  template<class T>
  struct ud_impl_params_t {

    // the input parameters
    ud_impl_params_t(std::vector<T*> in_param) : parameters(std::move(in_param)) {}

    // copy the input parameters
    ud_impl_params_t(std::vector<T> &in_param) {

      // fill up the parameters
      parameters.reserve(in_param.size());
      for(auto &p : in_param) {
        parameters.push_back(&p);
      }
    }

    // just with empty parameters
    ud_impl_params_t(size_t num_params) : parameters(num_params) {}

    // this is for the output parameters
    template<size_t n>
    T &get() { return *parameters[n]; }

    // this is for the input parameters as they are constant
    template<size_t n>
    const T &get() const { return *parameters[n]; }

    // this gets the parameters at runtime
    const T &get_by_idx(size_t idx) { return *parameters[idx]; }

    // sets the parameter
    template<size_t n>
    void set(T &_in) {
      parameters[n] = &_in;
    }

    // set the parameter on index
    void set(size_t n, T &_in) {
      parameters[n] = &_in;
    }

   private:

    // holds the input parameters
    std::vector<T*> parameters;
  };

  // define the parameters for the meta
  using tensor_params_t = ud_impl_params_t<tensor_t>;
  using meta_params_t = ud_impl_params_t<tensor_meta_t>;

  // each apply is a call to these
  using ud_impl_callable = std::function<void(const tensor_params_t &_in, tensor_params_t &_out)>;

  // the impl_id of the implementation, this is initialized by the udf manager
  ud_impl_id_t impl_id;

  // the implementation name of this function this has to be unique
  // for example this could be mkl_matrix_multiplication or stressen_matrix_multiplication
  std::string impl_name;

  // the name of the ud function, this is the same for all the ud functions that create an equivalent result
  // for example matrix_multiplication, matrix_addition, etc..
  std::string ud_name;

  // the input types of the tensors
  std::vector<std::string> inputTypes;

  // tells the system what inputs can be inplace
  // an input can only be inplace if and only if there is one output
  // and that output can be the same tensor that input without chaning the result
  std::vector<int32_t> inputInplace;

  // the output types
  std::vector<std::string> outputTypes;

  // does this implementation require the tensors to be on the gpu
  bool is_gpu;

  // make the virtual destructor
  virtual ~ud_impl_t() = default;

  // this is a function pointer to the function that needs to be applied
  // we don't use virtual function on purpose applied function can be a free standing function
  ud_impl_callable fn;

  // returns the complexity hint of the ud function
  virtual size_t get_complexity_hint(const meta_params_t &_in) = 0;

  // returns the output meta data
  virtual void get_out_meta(const meta_params_t &_in, meta_params_t &_out) const = 0;
};

// define a nice way to say unique_ptr of ud_impl_t
using ud_impl_ptr_t = std::unique_ptr<ud_impl_t>;

struct ud_func_t {

  // the impl_id of the implementation, this is initialized by the udf manager
  ud_id_t id;

  // the name of the ud function, this is the same for all the ud functions that create an equivalent result
  // for example matrix_multiplication, matrix_addition, etc..
  std::string ud_name;

  // this tells us whether the ud function is associative, or not
  bool is_ass;

  // this tells us whether the ud function commutative, or not
  bool is_com;

  // this tells us how many tensors we need as an input to this function
  size_t num_in;

  // this tells us how tensors this function is outputting
  size_t num_out;

  // these are all the implementations of this ud function
  std::vector<ud_impl_ptr_t> impls;

  // adds the implementation to the ud function
  ud_impl_id_t add_impl(ud_impl_ptr_t _impl) {

    // add the implementation to the list of all implementations
    auto impl_id = static_cast<ud_impl_local_t>(impls.size());
    _impl->impl_id = ud_impl_id_t{.ud_id = id, .impl_id = impl_id};
    impls.emplace_back(std::move(_impl));

    return impls.back()->impl_id;
  }
};

// define a nicer way to say unique_ptr of ud_func_t
using ud_func_ptr_t = std::unique_ptr<ud_func_t>;

}