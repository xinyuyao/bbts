#include "move_op.h"
#include <cassert>
#include <cstddef>
#include <cstdlib>

namespace bbts {

  // constructs the reduce operation
  move_op_t::move_op_t(bbts::communicator_t &_comm, int32_t _tag, 
                       bbts::tensor_t *_tensor, size_t _num_bytes, 
                       bbts::tensor_stats_t &_stats, tid_t _tid, 
                       bool _is_sender, bbts::storage_t &_storage, bbts::node_id_t _node) : _comm(_comm),
                                                                                            _tag(_tag),
                                                                                            _tensor(_tensor),
                                                                                            _num_bytes(_num_bytes),
                                                                                            _stats(_stats),
                                                                                            _tid(_tid),
                                                                                            _is_sender(_is_sender),
                                                                                            _storage(_storage),
                                                                                            _node(_node) {}

  // apply this operation
  bbts::tensor_t *move_op_t::apply() {
    
    // is this the sender, if so we initiate a send request
    if(_is_sender) {

      // do the sending
      if(!_comm.send_sync(_tensor, _num_bytes, _node, _tag)) {
        std::cout << "Failed to send the tensor, in a MOVE operation.\n";
        exit(-1);
      }
      
    } else {

      // allocate a buffer for the tensor
      _tensor = _storage.create_tensor(_tid, _num_bytes, _stats.is_gpu(_tid));

      // recieve the request and check if there is an error
      if (!_comm.receive_request_sync(_node, _tag, _tensor, _num_bytes)) {
        std::cout << "Failed to recieve the tensor, in a MOVE operation.\n";
        exit(-1);
      }
    }

    // return the tensor
    return _tensor;
  }

}