#pragma once

#include "../../src/tensor/tensor.h"

namespace bbts {

struct ffnn_tensor_meta_t : public tensor_meta_t {

  // the meta stucture
  struct m_t {

    uint32_t num_rows;
    uint32_t num_cols;
    bool     has_bias;
  };

  // returns the meta data struct
  m_t &m() const {

    // we use it as the blob
    return *((m_t*) _blob);
  }

  // init the tensor with the format impl_id
  ffnn_tensor_meta_t(tfid_t _id) : tensor_meta_t{.fmt_id = _id} {}

  // init the tensor meta with row and column numbers
  ffnn_tensor_meta_t(tfid_t _id, bool has_bias, uint32_t num_rows, uint32_t num_cols) : tensor_meta_t{.fmt_id = _id} {
    this->m() = {.num_rows = num_rows, .num_cols = num_cols, .has_bias = has_bias};
  }
};

struct ffnn_dense_t : public tensor_t {

  // return the meta data of the dense tensor
  ffnn_tensor_meta_t &meta() const {
    return *((ffnn_tensor_meta_t*) &_meta);
  }

  // return the 
  float *data() const {
    return (float*) _blob;
  }

  // returns the bias
  float *bias() const {
    return ((float*) _blob) + meta().m().num_rows * meta().m().num_cols;
  }

  // return creation functions
  static tensor_creation_fs_t get_creation_fs();
};

}