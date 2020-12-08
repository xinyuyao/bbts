#pragma once

#include "tensor.h"

namespace bbts {

struct dense_tensor_meta_t : public tensor_meta_t {

  // returns the meta data struct
  auto &m() const {

    struct m {
      int32_t num_rows;
      int32_t num_cols;
    };

    // we use it as the blob
    return *((m*) _blob);
  }

  // init the tensor with the format impl_id
  dense_tensor_meta_t(tfid_t _id) : tensor_meta_t{.fmt_id = _id} {}

  // init the tensor meta with row and column numbers
  dense_tensor_meta_t(tfid_t _id, int32_t num_rows, int32_t num_cols) : tensor_meta_t{.fmt_id = _id} {
    this->m() = {.num_rows = num_rows, .num_cols = num_cols};
  }
};

struct dense_tensor_t : public tensor_t {

  // return the meta data of the dense tensor
  dense_tensor_meta_t &meta() const {
    return *((dense_tensor_meta_t*) &_meta);
  }

  // returns the payload of the tensor
  float *data() {
    return (float*) _blob;
  }

  // return creation functions
  static tensor_creation_fs_t get_creation_fs();
};

}