#include "../src/tensor/tensor.h"
#include "../src/tensor/tensor_factory.h"
#include "../src/ud_functions/ud_function.h"
#include "../src/ud_functions/udf_manager.h"

#include <cstring> // memset

using namespace bbts;

enum scalar_type: char {
  st_float,
  st_int,
  st_bool
};

struct blob_meta_t : public tensor_meta_t {

  // returns the meta data struct
  auto &m() const {

    struct m {
      scalar_type which_type;
      uint32_t num;
    };

    // we use it as the blob
    return *((m*) _blob);
  }

  blob_meta_t(tfid_t _id) : tensor_meta_t{.fmt_id = _id} {}

  // init the tensor meta
  blob_meta_t(tfid_t _id, uint32_t num) : tensor_meta_t{.fmt_id = _id} {
    this->m() = {.num = num};
  }

  size_t get_blob_size() {
    auto &meta = this->m();
    
    size_t elem_size;
    switch(meta.which_type) {
      case st_float: elem_size = sizeof(float); break;
      case st_int:   elem_size = sizeof(int);   break;
      case st_bool:  elem_size = sizeof(bool);  break;
    }

    return elem_size * meta.num;
  }
};

struct blob_t : public tensor_t {

  // return the meta data of the dense tensor
  blob_meta_t &meta() const {
    return *((blob_meta_t*) &_meta);
  }

  // returns the payload of the tensor
  void *data() {
    return (void*) _blob;
  }

  // return creation functions
  static tensor_creation_fs_t get_creation_fs() {
    // return the init function
    auto init = [](void *here, const tensor_meta_t &_meta) -> tensor_t & {
      auto &t = *(blob_t *) here;
      auto &m = *(blob_meta_t * ) & _meta;
      t.meta() = m;
      return t;
    };
  
    // return the size
    auto size = [](const tensor_meta_t &_meta) {
      auto &m = *(blob_meta_t *) &_meta;
      return sizeof(tensor_meta_t) + m.get_blob_size();
    };
  
    // return the tensor creation functions
    return tensor_creation_fs_t{.get_size = size, .init_tensor = init};
  }
};

struct init_zero_t : public ud_impl_t {
  init_zero_t() {
    impl_name = "init_zero";
    ud_name = "init_zero";
    inputTypes = {"blob"};
    outputTypes = {"blob"};
    inputInplace = {0};
    is_gpu = false;
    fn = &init_zero_t::fn_;
  }

  // returns an estimate of the complexity
  size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                             const meta_args_t &_in) override {
    return 0;
  }

  // return the meta of the output
  void get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                    const meta_args_t &_in, meta_args_t &_out) const override {
    _out = _in;
  }

  // does the work
  static void fn_(const bbts::ud_impl_t::tensor_params_t &params,
                  const tensor_args_t &_in, tensor_args_t &_out) {
    blob_t& out = _out.get<0>().as<blob_t>();
    memset(out.data(), 0, out.meta().get_blob_size() / sizeof(char));
  }
};


ud_func_ptr_t get_init_zero() {
  return std::make_unique<ud_func_t>(
      ud_func_t {
          .ud_name = "init_zero",
          .is_ass = false,
          .is_com = false,
          .num_in = 1,
          .num_out = 1,
          .impls = {}
      }
  );
}

extern "C" {

  void register_tensors(tensor_factory_ptr_t tensor_factory) {
    tensor_factory->register_fmt("blob", blob_t::get_creation_fs());
  }
  
  void register_udfs(udf_manager_ptr udf_manager) {
    udf_manager->register_udf(std::make_unique<ud_func_t>(
          ud_func_t {
            .ud_name = "init_zero",
            .is_ass = false,
            .is_com = false,
            .num_in = 1,
            .num_out = 1,
            .impls = {}
          }));
    udf_manager->register_udf_impl(std::make_unique<init_zero_t>());
  }

}
