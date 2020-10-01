#include "builtin_formats.h"

namespace bbts {

tensor_creation_fs_t bbts::dense_tensor_t::get_creation_fs() {

  // return the init function
  auto init = [](void *here, const tensor_meta_t &_meta) -> tensor_t & {
    auto &t = *(dense_tensor_t *) here;
    auto &m = *(dense_tensor_meta_t * ) & _meta;
    t.meta() = m;
    return t;
  };

  // return the size
  auto size = [](const tensor_meta_t &_meta) {
    auto &m = *(dense_tensor_meta_t *) &_meta;
    return sizeof(tensor_meta_t) + m.m().num_cols * m.m().num_rows * sizeof(float);
  };

  // return the tensor creation functions
  return tensor_creation_fs_t{.get_size = size, .init_tensor = init};
}

}