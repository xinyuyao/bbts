#pragma once

#include <vector>
#include <cstdint>

namespace bbts {

  using tid_t = int32_t;

  // the dense tensor class in our system
  struct tensor_t {

    // the maximum number of indices supported
    static const size_t MAX_DIM_SIZE = 10;

    // the id of this tensor
    tid_t id;

    // the number of dimensions of this tensor
    int32_t num_dims;

    // the indices of the tensor
    int32_t dims[MAX_DIM_SIZE];

    // after this tensor header is all the
    float data[0];

    // returns the number of bytes
    [[nodiscard]] size_t num_bytes() const {

      // figure out the size of the data[0]
      auto data_size = sizeof(float);
      for(auto dim = 0; dim < num_dims; ++dim) { data_size *= dims[dim]; }

      // add the size of the header to it and return
      return data_size + sizeof(tensor_t);
    }

    static tensor_t init(void *here, tid_t id, const std::vector<int32_t> &dims) {

      // init the tensor
      tensor_t &ref = *(new (here) (tensor_t) {.id = id, .num_dims = static_cast<int32_t>(dims.size())});
      for(auto dim = 0; dim < dims.size(); ++dim) { ref.dims[dim] = dims[dim]; }

      // return it as a reference
      return ref;
    }

  };


}