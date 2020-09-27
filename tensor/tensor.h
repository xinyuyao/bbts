#pragma once

#include <vector>
#include <cstdint>

namespace bbts {

  // the identifier of the tensor
  using tid_t = int32_t;

  // holds the meta data of the tensor
  struct tensor_meta_t {

    // the maximum number of indices supported
    static const size_t MAX_DIM_SIZE = 10;

    // the number of dimensions of this tensor
    int32_t num_dims;

    // the indices of the tensor
    int32_t dims[MAX_DIM_SIZE];
  };

  // the dense tensor class in our system
  struct tensor_t {

    // the id of this tensor
    tid_t id;

    // the meta of the tensor
    tensor_meta_t meta;

    // after this tensor header is all the
    float data[0];

    // returns the number of bytes
    [[nodiscard]] size_t num_bytes() const;

    // initializes the tensor to a location
    static tensor_t & init(void *here, tid_t id, const std::vector<int32_t> &dims);

    // initializes the tensor with the given meta data
    static tensor_t & init(void *here, const tensor_meta_t &meta);

  };
}