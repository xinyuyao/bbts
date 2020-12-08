#pragma once

#include <vector>
#include <assert.h> 
#include "../tensor/tensor.h"
#include "../server/node.h"
#include "../ud_functions/ud_function.h"

namespace bbts {

// the impl_id of the operation, this is unique globally across all processes
using command_id_t = int32_t;

struct command_t {

  enum op_type_t {
    APPLY = 0,
    REDUCE = 1,
    MOVE = 2,
    DELETE = 3
  };

  // specifies exactly what tensor on which node we refer to
  struct tid_node_id_t {
    tid_t tid; 
    node_id_t node;
  };

  // is this a delete
  [[nodiscard]] bool is_delete() const { return _type == op_type_t::DELETE; }
  
  // is this a move
  [[nodiscard]] bool is_move() const { return _type == op_type_t::MOVE; }
  
  // is this an apply
  [[nodiscard]] bool is_apply() const { return _type == op_type_t::APPLY; }

  // tells us the node that is going to initiate the command
  [[nodiscard]] node_id_t get_root_node() const {

    switch (_type) {

      // for the delete node that is the node where we are deleting 
      case op_type_t::DELETE: {
        assert(!_input_tensors.empty());
        return _input_tensors.front().node;
      }
      case op_type_t::APPLY: {
        
        // for the apply the inputs and outputs are on the same node
        // so we just grab one of them
        assert(!_input_tensors.empty());
        return _input_tensors.front().node;
      }
      case op_type_t::REDUCE: {

        // the reduce op has to have all the oputs on the same node
        assert(!_output_tensors.empty());
        return _output_tensors.front().node;
      }
      case op_type_t::MOVE: {
        
        // for the move we assume that the node with the tensors initiates the move,
        // as it knows when they are available
        assert(!_input_tensors.empty());
        return _input_tensors.front().node;
      }
      default: {

        // this is never supposed to happen
        throw std::runtime_error("Unknown operation type.\n");
      }
    }

  }

  // the impl_id of the operation
  command_id_t _id;

  // the type of operation
  op_type_t _type;

  // the input tensors
  std::vector<tid_node_id_t> _input_tensors;

  // the output tensors
  std::vector<tid_node_id_t> _output_tensors;

  // the function we want to execute
  ud_impl_id_t _fun_id = {-1, -1};
};

// the command
using command_ptr_t = std::unique_ptr<command_t>; 

}