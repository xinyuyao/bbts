#include "ffnn_types.h"
#include <iostream>
#include <sstream>

namespace bbts {

tensor_creation_fs_t bbts::ffnn_dense_t::get_creation_fs() {

  // return the init function
  auto init = [](void *here, const tensor_meta_t &_meta) -> tensor_t & {
    auto &t = *(ffnn_dense_t *) here;
    auto &m = *(ffnn_dense_meta_t * ) & _meta;
    t.meta() = m;
    return t;
  };

  // return the size
  auto size = [](const tensor_meta_t &_meta) {
    auto &m = *(ffnn_dense_meta_t *) &_meta;

    auto num_elements = m.m().num_cols * m.m().num_rows;
    num_elements += m.m().has_bias ? m.m().num_cols : 0;

    // we add the bias if we have it to the size
    return sizeof(tensor_meta_t) + num_elements * sizeof(float);
  };

  auto pnt = [](const void *here, std::stringstream &ss) {
    
    // get the tensor
    auto &t = *(ffnn_dense_t *) here;

    // extract the info
    auto num_rows = t.meta().m().num_rows;
    auto num_cols = t.meta().m().num_cols;
    auto data = t.data();

    // print the tensor
    for(int i = 0; i < num_rows; i++) {
      ss << "[";
      for(int j = 0; j < num_cols; j++) {
        ss << data[i * num_cols + j] << ((j == num_cols - 1) ? "" : ",");
      }
      ss << "]\n";
    }

  };

  auto deserialize_meta = [](tensor_meta_t& _meta, tfid_t id, const char *data) {
    auto &m = *(ffnn_dense_meta_t *) &_meta;
    m.fmt_id = id;

    auto s = std::string(data);
    std::string delimiter = "|";
    size_t pos = s.find(delimiter);
    std::string num_rows = s.substr(0, s.find(delimiter));
    s.erase(0, pos + delimiter.length());
    std::string num_columns = s.substr(0, s.find(delimiter));

    m.m().num_rows = std::atoi(num_rows.c_str());
    m.m().num_cols = std::atoi(num_columns.c_str());
    // QUESTION: Do we need to add bias? Since we got num_rows and num_cols from input data. Maybe we don't need to consider it? The data itself already contains bias?

  };

  auto deserialize_tensor = [](tensor_t* here, tfid_t id, const char *data) -> tensor_t& {

    auto &a = here->as<ffnn_dense_t>();
    // set meta data
    // tfid
    a.meta().fmt_id = id;

    // number of rows and columns
    auto s = std::string(data);
    std::string delimiter = "|";
    size_t pos = s.find(delimiter);
    std::string num_rows = s.substr(0, s.find(delimiter));
    s.erase(0, pos + delimiter.length());
    std::string num_columns = s.substr(0, s.find(delimiter));

    a.meta().m().num_rows = std::atoi(num_rows.c_str());
    a.meta().m().num_cols = std::atoi(num_columns.c_str());

    // put actual data inside tensor
    s.erase(0, pos + delimiter.length());
    std::string data_delimiter = " ";
    size_t data_pos = s.find(delimiter);

    for (auto row = 0; row < a.meta().m().num_rows; ++row) {
      for (auto col = 0; col < a.meta().m().num_cols; ++col) {
        data_pos = s.find(data_delimiter);
        std::string my_data = s.substr(0, data_pos);
        s.erase(0, data_pos + data_delimiter.length());
        auto temp = std::atof(my_data.c_str());
        a.data()[row * a.meta().m().num_cols + col] = temp;
      }
    }

    return a;
  };

  // return the tensor creation functions
  return tensor_creation_fs_t{.get_size = size, .init_tensor = init, .print = pnt, .deserialize_meta = deserialize_meta, .deserialize_tensor = deserialize_tensor};
}


tensor_creation_fs_t bbts::ffnn_sparse_t::get_creation_fs() {

  // return the init function
  auto init = [](void *here, const tensor_meta_t &_meta) -> tensor_t & {
    auto &t = *(ffnn_sparse_t *) here;
    auto &m = *(ffnn_sparse_meta_t * ) & _meta;
    t.meta() = m;
    return t;
  };

  // return the size
  auto size = [](const tensor_meta_t &_meta) {
    
    // we add the bias if we have it to the size
    auto &m = *(ffnn_sparse_meta_t *) &_meta;
    return sizeof(tensor_meta_t) + m.m().nnz * sizeof(ffnn_sparse_t::ffnn_sparse_value_t);
  };

  auto pnt = [](const void *here, std::stringstream &ss) {
    
    // get the tensor
    auto &t = *(ffnn_sparse_t *) here;

    // extract the info
    auto num_rows = t.meta().m().num_rows;
    auto num_cols = t.meta().m().num_cols;
    auto nnz = t.meta().m().nnz;
    auto data = t.data();

    // print the tensor 
    ss << "{ rows = " << num_rows << ", cols = " << num_cols << " }";
    ss << "[\n";
    for(int i = 0; i < nnz; i++) {
      ss << "(" << data[i].row << ", " << data[i].col << ", " << data[i].val << ")" << ((i == (nnz - 1)) ? "" : ",") << "\n";
    }
    ss << "]\n";

  };

  auto deserialize_meta = [](tensor_meta_t& _meta, tfid_t id, const char *data) {
    auto &m = *(ffnn_dense_meta_t *) &_meta;
    m.fmt_id = id;

    auto s = std::string(data);
    std::string delimiter = "|";
    size_t pos = s.find(delimiter);
    std::string num_rows = s.substr(0, s.find(delimiter));
    s.erase(0, pos + delimiter.length());
    std::string num_columns = s.substr(0, s.find(delimiter));

    m.m().num_rows = std::atoi(num_rows.c_str());
    m.m().num_cols = std::atoi(num_columns.c_str());
    // QUESTION: Do we need to add bias? Since we got num_rows and num_cols from input data. Maybe we don't need to consider it? The data itself already contains bias?

  };

  auto deserialize_tensor = [](tensor_t* here, tfid_t id, const char *data) -> tensor_t& {

    auto &a = here->as<ffnn_dense_t>();
    // set meta data
    // tfid
    a.meta().fmt_id = id;

    // number of rows and columns
    auto s = std::string(data);
    std::string delimiter = "|";
    size_t pos = s.find(delimiter);
    std::string num_rows = s.substr(0, s.find(delimiter));
    s.erase(0, pos + delimiter.length());
    std::string num_columns = s.substr(0, s.find(delimiter));

    a.meta().m().num_rows = std::atoi(num_rows.c_str());
    a.meta().m().num_cols = std::atoi(num_columns.c_str());

    // put actual data inside tensor
    s.erase(0, pos + delimiter.length());
    std::string data_delimiter = " ";
    size_t data_pos = s.find(delimiter);

    for (auto row = 0; row < a.meta().m().num_rows; ++row) {
      for (auto col = 0; col < a.meta().m().num_cols; ++col) {
        data_pos = s.find(data_delimiter);
        std::string my_data = s.substr(0, data_pos);
        s.erase(0, data_pos + data_delimiter.length());
        auto temp = std::atof(my_data.c_str());
        a.data()[row * a.meta().m().num_cols + col] = temp;
      }
    }

    return a;
  };
  

  // return the tensor creation functions
  // TODO: need to implement .deserialize_tensor, and .deserialize_meta to make it work. Since data loader change the format of tensor. Library is the 
  // combined of tensors and kernel func. I need to implement these two func to make it work.
  // There is a similar implementation for this two functions when implement data loader, check it
  return tensor_creation_fs_t{.get_size = size, .init_tensor = init, .print = pnt, .deserialize_meta = deserialize_meta, .deserialize_tensor = deserialize_tensor};
  
}

}