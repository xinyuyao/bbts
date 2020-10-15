#include "../communication/communicator.h"
#include "../tensor/builtin_formats.h"
#include "../storage/storage.h"
#include "../ud_functions/udf_manager.h"
#include <mpich/mpi.h>
#include <iostream>
#include <algorithm>

namespace bbts {

class move_op_t {
public:

  // the mpi communicator we are going to use to perform the communication
  bbts::communicator_t &_comm;

  // the tag we use to identify this reduce
  int32_t _tag; 

  // the tensor of this node, either the input or the output
  bbts::tensor_t *_tensor; 

  // is this node the sender
  bool _is_sender;

  // we use this to initialize the tensors, and the the size of the tensor
  bbts::tensor_factory_t &_factory;
  
  // we use this to allocate the tensors
  bbts::storage_t &_storage;

  // the node we are send or recieving from...
  bbts::node_id_t _node;

  // constructs the reduce operation
  move_op_t(bbts::communicator_t &_comm, int32_t _tag, bbts::tensor_t *_tensor,  
          bool _is_sender, bbts::tensor_factory_t &_factory, 
          bbts::storage_t &_storage, bbts::node_id_t _node);

  // apply this operation
  bbts::tensor_t *apply();
};

}