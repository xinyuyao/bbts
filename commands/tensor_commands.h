#pragma once

#include <vector>
#include "../tensor/tensor.h"
#include "../ud_functions/ud_function.h"

namespace bbts {

struct command_t {

  enum op_type_t {
    APPLY,
    MOVE,
    BROADCAST,
    DELETE
  };

  // the type of operation
  op_type_t type;

  // the input tensors
  std::vector<tid_t> input_tensors;

  // the output tensors
  std::vector<tid_t> output_tensors;

  // the function we want to execute
  ud_id_t fun_id = -1;
};

}