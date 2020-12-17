#pragma once

#include <vector>
#include <assert.h> 
#include "../server/node_config.h"
#include "../ud_functions/ud_function.h"

namespace bbts {

// the impl_id of the operation, this is unique globally across all processes
using command_id_t = int32_t;

// pre-declare the command ptr type as well as a deleter for it
struct command_t;
struct command_deleter_t { void operator()(command_t* p) { delete[] ((char*) p); }};
using command_ptr_t = std::unique_ptr<bbts::command_t, command_deleter_t>;

// the commands we execute, they can be copied directly with a memcpy as they are layered out flat
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

  // returns the input tensor
  [[nodiscard]] tid_node_id_t &get_input(int32_t idx) {
    return _tensors[idx];
  }

  // return the input but constant
  [[nodiscard]] const tid_node_id_t &get_input(int32_t idx) const {
    return _tensors[idx];
  }

  // returns the output tensor
  [[nodiscard]] tid_node_id_t &get_output(int32_t idx) {
    return _tensors[_num_inputs + idx];
  }

  // return the output tensor but constant
  [[nodiscard]] const tid_node_id_t &get_output(int32_t idx) const {
    return _tensors[_num_inputs + idx];
  }

  // return the number of input tensors
  [[nodiscard]] int32_t get_num_inputs() const { return _num_inputs; }

  // return the number of output tensors
  [[nodiscard]] int32_t get_num_outputs() const { return _num_outputs; }

  // is this a delete
  [[nodiscard]] bool is_delete() const { return type == op_type_t::DELETE; }

  // return the number of bytes
  [[nodiscard]] size_t num_bytes() const { return sizeof(bbts::command_t) +
                                                  (_num_inputs + _num_outputs) * sizeof(tid_node_id_t); }

  // is this a move
  [[nodiscard]] bool is_move() const { return type == op_type_t::MOVE; }
  
  // is this an apply
  [[nodiscard]] bool is_apply() const { return type == op_type_t::APPLY; }

  // check if command uses a particular node
  [[nodiscard]] bool uses_node(node_id_t _node) const {

    // go and check every tensor if it is located on a node
    int32_t n = get_num_inputs() + get_num_outputs();
    for(int32_t idx = 0; idx < n; ++idx) {
      if(_tensors[idx].node == _node) {
        return true;
      }
    }

    return false;
  }

  // tells us the node that is going to initiate the command
  [[nodiscard]] node_id_t get_root_node() const {

    switch (type) {

      // for the delete node that is the node where we are deleting 
      case op_type_t::DELETE: {
        assert(_num_inputs != 0);
        return get_input(0).node;
      }
      case op_type_t::APPLY: {
        
        // for the apply the inputs and outputs are on the same node
        // so we just grab one of them
        assert(_num_inputs != 0);
        return get_input(0).node;
      }
      case op_type_t::REDUCE: {

        // the reduce op has to have all the outputs on the same node
        assert(_num_outputs != 0);
        return get_output(0).node;
      }
      case op_type_t::MOVE: {
        
        // for the move we assume that the node with the tensors initiates the move,
        // as it knows when they are available
        assert(_num_inputs != 0);
        return get_input(0).node;
      }
      default: {

        // this is never supposed to happen
        throw std::runtime_error("Unknown operation type.\n");
      }
    }
  }

  // the impl_id of the operation
  command_id_t id;

  // the type of operation
  op_type_t type;

  // the function we want to execute
  ud_impl_id_t fun_id = {-1, -1};

  // clone the command
  command_ptr_t clone() {

    // create a new command
    auto tmp = create_unique(_num_inputs, _num_outputs);

    // set the id type and function
    tmp->id = id;
    tmp->type = type;
    tmp->fun_id = fun_id;

    // fill-up the inputs
    for(auto idx = 0; idx < _num_inputs; idx++) {
      tmp->get_input(idx) = get_input(idx);
    }

    // fill-up the outputs
    for(auto idx = 0; idx < _num_outputs; idx++) {
      tmp->get_output(idx) = get_output(idx);
    }

    // return the the clone
    return std::move(tmp);
  }

  // create the command
  static command_ptr_t create_unique(size_t num_inputs, size_t num_outputs) {

    // allocate the memory
    std::unique_ptr<char[]> p(new char[sizeof(bbts::command_t) + (num_inputs + num_outputs) * sizeof(tid_node_id_t)]);

    // construct the command
    new (p.get()) bbts::command_t(num_inputs, num_outputs);

    auto pReleased = p.release();
    auto pDerived = (bbts::command_t *)(pReleased);
    auto d = std::unique_ptr<bbts::command_t, command_deleter_t>(pDerived);

    // move the command
    return std::move(d);
  }

  static command_ptr_t create_unique(command_id_t _id,
                                     op_type_t _type,
                                     ud_impl_id_t _fun_id,
                                     const std::vector<tid_node_id_t> &_inputs,
                                     const std::vector<tid_node_id_t> &_outputs) {

    // create the output
    auto tmp = create_unique(_inputs.size(), _outputs.size());

    // set the id type and function
    tmp->id = _id;
    tmp->type = _type;
    tmp->fun_id = _fun_id;

    // fill-up the inputs
    for(auto idx = 0; idx < _inputs.size(); idx++) {
      tmp->get_input(idx) = _inputs[idx];
    }

    // fill-up the outputs
    for(auto idx = 0; idx < _outputs.size(); idx++) {
      tmp->get_output(idx) = _outputs[idx];
    }

    // return the created pointer
    return std::move(tmp);
  }

private:

  // make the default constructor private so that people have to call create manually
  command_t(size_t num_inputs, size_t num_output) : _num_inputs(num_inputs), _num_outputs(num_output) {}

  // the number of input tensors
  size_t _num_inputs;

  // the number of output tensors
  size_t _num_outputs;

  // the tensors input and output
  tid_node_id_t _tensors[0];
};

}