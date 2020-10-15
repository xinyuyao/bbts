#include "broadcast_op.h"

namespace bbts {

broadcast_op_t::broadcast_op_t(bbts::communicator_t &_comm,
                           bbts::tensor_factory_t &_factory, bbts::storage_t &_storage,
                           const std::vector<bbts::node_id_t> &_nodes,
                           int32_t _tag, bbts::tensor_t *_in): _comm(_comm),
                                                               _factory(_factory),
                                                               _storage(_storage),
                                                               _nodes(_nodes),
                                                               _tag(_tag),
                                                               _in(_in) {}

int32_t broadcast_op_t::get_num_nodes() const {
  return _nodes.size();
}

int32_t broadcast_op_t::get_local_rank() const {
  return std::distance(_nodes.begin(), std::find(_nodes.begin(), _nodes.end(), _comm.get_rank()));
}

int32_t broadcast_op_t::get_global_rank(int32_t local_rank) const {
  return _nodes[local_rank];
}

// runs the broadcast
bbts::tensor_t *broadcast_op_t::apply() {

  int size = get_num_nodes();
  int rank = get_local_rank();
  int vrank = (rank + size ) % size;

  int dim = opal_cube_dim(size);
  int hibit = opal_hibit(vrank, dim);
  --dim;

  // the root node has the vrank 0, if this is not the root node 
  // we need to recieve the broadcasted tensor
  if (vrank > 0) {

    // figure out the node we need to recieve the data from
    assert(hibit >= 0);
    int peer = ((vrank & ~(1 << hibit))) % size;

    // try to get the request
    auto req = _comm.expect_request_sync(get_global_rank(peer), _tag);

    // check if there is an error
    if (!req.success) {
      std::cout << "Error 1\n";
    }

    // allocate a buffer for the tensor
    _in = _storage.create_tensor(req.num_bytes);

    // recieve the request and check if there is an error
    if (!_comm.recieve_request_sync(_in, req)) {
      std::cout << "Error 2\n";
    }
  }

  // allocate the requests
  std::vector<communicator_t::async_request_t> requests;
  requests.reserve(size);

  // send the tensor to the right nodes
  for (int i = hibit + 1, mask = 1 << i; i <= dim; ++i, mask <<= 1) {
    int peer = vrank | mask;
    if (peer < size) {

      // figure out where we need to send it
      peer = peer % size;

      // return the size of the tensor
      auto output_size = _factory.get_tensor_size(_in->_meta);

      // send the tensor async
      requests.emplace_back(_comm.send_async(_in, output_size, get_global_rank(peer), _tag));

      // we failed here return null
      if(!requests.back().request) {
        return nullptr;
      }
    }
  }

  // wait on all requests
  bool success = true;
  if (!requests.empty()) {
    for(auto &r : requests) {
      
      // wait for request to finish
      if(!_comm.wait_async(r)) {
        success = false;
      }
    }
  }

  // if we failed return null
  if(!success) {
    return nullptr;
  }

  // return the tensor
  return _in;
}

}