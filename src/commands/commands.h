#pragma once

#include <vector>
#include "../tensor/tensor.h"
#include "../ud_functions/ud_function.h"

namespace bbts {

// the id of the operation, this is unique globally across all processes
using command_id_t = int32_t;

struct command_t {

  enum op_type_t {
    APPLY,
    MOVE,
    DELETE
  };

  // the id of the operation
  command_id_t _id;

  // the type of operation
  op_type_t _type;

  // the input tensors
  std::vector<tid_t> _input_tensors;

  // the output tensors
  std::vector<tid_t> _output_tensors;

  // the function we want to execute
  ud_id_t _fun_id = -1;
};

}