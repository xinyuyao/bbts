#pragma once

#include "../communication/communicator.h"
#include "../tensor/builtin_formats.h"
#include "../storage/storage.h"
#include "../ud_functions/udf_manager.h"
#include <mpich/mpi.h>
#include <iostream>
#include <algorithm>

namespace bbts {

// This code is based on the implementation from MPICH-1.
// Here's the algorithm.  Relative to the root, look at the bit pattern in
// my rank.  Starting from the right (lsb), if the bit is 1, send to
// the node with that bit zero and exit; if the bit is 0, receive from the
// node with that bit set and combine (as long as that node is within the
// group)

// Note that by receiving with source selection, we guarantee that we get
// the same bits with the same input.  If we allowed the parent to receive
// the children in any order, then timing differences could cause different
// results (roundoff error, over/underflows in some cases, etc).

// Because of the way these are ordered, if root is 0, then this is correct
// for both commutative and non-commutitive operations.  If root is not
// 0, then for non-commutitive, we use a root of zero and then send
// the result to the root.  To see this, note that the ordering is
// mask = 1: (ab)(cd)(ef)(gh)            (odds send to evens)
// mask = 2: ((ab)(cd))((ef)(gh))        (3,6 send to 0,4)
// mask = 4: (((ab)(cd))((ef)(gh)))      (4 sends to 0)

// Comments on buffering.
// If the datatype is not contiguous, we still need to pass contiguous
// data to the user routine.
// In this case, we should make a copy of the data in some format,
// and send/operate on that.

// In general, we can't use MPI_PACK, because the alignment of that
// is rather vague, and the data may not be re-usable.  What we actually
// need is a "squeeze" operation that removes the skips.
class reduce {
public:

  // the mpi communicator we are going to use to perform the communication
  bbts::mpi_communicator_t &_comm;

  // we use the tensor factory to initialize the tensors and calculate the required size
  bbts::tensor_factory_t &_factory; 

  // the storage we use this to allocate the output and the intermediate tensors
  bbts::storage_t &_storage;

  // the nodes
  const std::vector<bbts::node_id_t> &_nodes;

  // the tag we use to identify this reduce
  int32_t _tag; 

  // the input tensor of this node
  bbts::tensor_t &_in; 

  // the reduce operation
  const bbts::ud_impl_t &_reduce_op;

  // constructs the reduce operation
  reduce(bbts::mpi_communicator_t &_comm, bbts::tensor_factory_t &_factory, 
         bbts::storage_t &_storage, const std::vector<bbts::node_id_t> &_nodes,
         int32_t _tag, bbts::tensor_t &_in, const bbts::ud_impl_t &_reduce_op);

  // get the number of nodes
  int32_t get_num_nodes() const;

  // get local rank
  int32_t get_local_rank() const;

  // get global rank
  int32_t get_global_rank(int32_t local_rank) const;

  // apply this operation
  bbts::tensor_t *apply();
  
};

}