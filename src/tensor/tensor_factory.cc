#include "tensor_factory.h"
#include "builtin_formats.h"

bbts::tfid_t bbts::tensor_factory_t::register_fmt(const std::string &_fmt_name,
                                                  const bbts::tensor_creation_fs_t &_fmt_funs) {

  // check if the format exits
  auto it = _reg_fmts.find(_fmt_name);
  if(it == _reg_fmts.end()) {
    _reg_fmts[_fmt_name] = _fmt_fs.size();
    _fmt_fs.emplace_back(_fmt_funs);
  }

  // we can not register twice the same format
  return -1;
}

bbts::tensor_t &bbts::tensor_factory_t::init_tensor(void *here, const bbts::tensor_meta_t &_meta) {

  // find the function to initialize the tensor
  if(_meta.fmt_id < _fmt_fs.size()) {
    return _fmt_fs[_meta.fmt_id].init_tensor(here, _meta);
  }

  // check if the format exits
  throw std::runtime_error("Requested get_tensor_size for a format not registered with the system.");
}

size_t bbts::tensor_factory_t::get_tensor_size(const bbts::tensor_meta_t &_meta) {

  // find the function and return the size
  if(_meta.fmt_id < _fmt_fs.size()) {
    return _fmt_fs[_meta.fmt_id].get_size(_meta);
  }

  // check if the format exits
  throw std::runtime_error("Requested get_tensor_size for a format not registered with the system.");
}

bbts::tensor_factory_t::tensor_factory_t() {

  // register the dense format
  register_fmt("dense", dense_tensor_t::get_creation_fs());
}

bbts::tfid_t bbts::tensor_factory_t::get_tensor_ftm(const std::string &_fmt_name) {

  // check if the format exits
  auto it = _reg_fmts.find(_fmt_name);
  if(it != _reg_fmts.end()) {
    return it->second;
  }
  return -1;
}
