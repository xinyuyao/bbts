#include "tensor.h"

size_t bbts::tensor_t::num_bytes() const {

  // figure out the size of the data[0]
  auto data_size = sizeof(float);
  for(auto dim = 0; dim < meta.num_dims; ++dim) { data_size *= meta.dims[dim]; }

  // add the size of the header to it and return
  return data_size + sizeof(tensor_t);
}

bbts::tensor_t & bbts::tensor_t::init(void *here, bbts::tid_t id, const std::vector<int32_t> &dims) {

  // init the tensor
  tensor_t &ref = *(new (here) (tensor_t) {.id = id});

  // init the dimensions
  ref.meta = {.num_dims = static_cast<int32_t>(dims.size())};
  for(auto dim = 0; dim < dims.size(); ++dim) { ref.meta.dims[dim] = dims[dim]; }

  // return it as a reference
  return ref;
}

bbts::tensor_t & bbts::tensor_t::init(void *here, const bbts::tensor_meta_t &meta) {
  return *(new (here) (tensor_t) {.meta = meta});
}
