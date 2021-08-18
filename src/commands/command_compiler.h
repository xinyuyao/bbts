#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include "abstract_command.h"
#include "cost_model.h"
#include "../commands/command.h"
#include "../tensor/tensor.h"
#include "../tensor/tensor_factory.h"
#include "../ud_functions/ud_function.h"
#include "../ud_functions/udf_manager.h"


namespace bbts {

class command_compiler_t {
public:

command_compiler_t(cost_model_ptr_t cost_model, size_t num_nodes) : cost_model(cost_model),
                                                                    num_nodes(num_nodes) {}
  
  struct node_cost_t {

    // the total cost of transfering all the nodes
    float transfer_cost = 0;

    // the total compute cost
    float compute_cost = 0;
  };

  std::vector<bbts::command_ptr_t> compile(const std::vector<abstract_command_t> &commands,
                                           std::vector<std::unordered_set<tid_t>> &tensor_locations) {
    
    float max_compute_cost = 0;
    float max_transfer_cost = 0;

    // the costs so far
    std::vector<node_cost_t> costs(num_nodes);

    // go compile all the commands
    command_id_t cur_cmd = 0;
    std::vector<bbts::command_ptr_t> out_commands;
    for(auto &c : commands) {

      // if the command is an APPLY then we need to make sure that all the tensors are on the same node
      if(c.type == abstract_command_type_t::APPLY) {

        // store the command
        generate_apply(cur_cmd,
                       out_commands,
                       c, 
                       costs,
                       tensor_locations,
                       max_compute_cost,
                       max_transfer_cost);
      }
      else if(c.type == abstract_command_type_t::REDUCE) {

        // store the command
        generate_reduce(cur_cmd,
                        out_commands,
                        c, 
                        costs,
                        tensor_locations,
                        max_compute_cost,
                        max_transfer_cost);
      }
      else {
        
        // generate the delete commands
        generate_deletes(cur_cmd,
                         out_commands,
                         c, 
                         costs,
                         tensor_locations,
                         max_compute_cost,
                         max_transfer_cost);

      }
    }

    // insert the postprocess deletions
    _insert_deletions(out_commands, cur_cmd, tensor_locations);

    // return the commands
    return std::move(out_commands);
  }

private:

  void generate_deletes(command_id_t &cur_cmd,
                        std::vector<bbts::command_ptr_t> &out_commands,
                        const abstract_command_t &c, 
                        std::vector<node_cost_t> &costs,
                        std::vector<std::unordered_set<tid_t>> &tensor_locations,
                        float &max_compute_cost,
                        float &max_transfer_cost) {
    
    for(node_id_t node = 0; node < num_nodes; ++node) {

      // check if there is something
      std::vector<command_t::tid_node_id_t> inputs;
      for(auto &in : c.input_tids) {

        // create the command and remove
        auto it = tensor_locations[node].find(in);
        if(it != tensor_locations[node].end()) {
          inputs.push_back(command_t::tid_node_id_t{.tid = in, .node = node});
          tensor_locations[node].erase(it);
        }
      }

      if(!inputs.empty()) {
        // make the apply
        auto cmd = command_t::create_delete(cur_cmd++, inputs);

        // store the command
        out_commands.push_back(std::move(cmd));
      }
    }
  }

  void generate_reduce(command_id_t &cur_cmd,
                       std::vector<bbts::command_ptr_t> &out_commands,
                       const abstract_command_t &c, 
                       std::vector<node_cost_t> &costs,
                       std::vector<std::unordered_set<tid_t>> &tensor_locations,
                       float &max_compute_cost,
                       float &max_transfer_cost) {

    // get the compute cost of running this
    command_param_list_t raw_param = {._data = c.params.data(), ._num_elements = c.params.size()};
    auto ud_info = cost_model->get_reduce_cost(c.ud_id, 
                                               bbts::ud_impl_t::tensor_params_t{._params = raw_param}, 
                                               c.input_tids,
                                               c.output_tids);

    // check if the reduce can be local
    node_id_t local_node = _get_can_be_local(c, tensor_locations);
    if(local_node != -1) {

      // init the inputs
      std::vector<command_t::tid_node_id_t> inputs(c.input_tids.size());
      for(int32_t idx = 0; idx < c.input_tids.size(); ++idx) {
        inputs[idx] = (command_t::tid_node_id_t{.tid = c.input_tids[idx], .node = local_node});
      }

      // init the parameters
      std::vector<command_param_t> params(c.params.size());
      for(int32_t idx = 0; idx < c.params.size(); ++idx) {
        params[idx] = c.params[idx];
      }

      // make the apply
      auto cmd = command_t::create_reduce(cur_cmd++, 
                                          ud_info.ud->impl_id, 
                                          ud_info.is_gpu, 
                                          params, 
                                          inputs, 
                                          command_t::tid_node_id_t{.tid = c.output_tids[0], 
                                                                   .node = local_node});

      // mark that we are creating a tensor here
      tensor_locations[local_node].insert(c.output_tids[0]);
      out_commands.push_back(std::move(cmd));
      return;
    }

    // first find the best node to send the data for each input
    std::vector<node_id_t> assigned_nodes(c.input_tids.size());
    for(int32_t idx = 0; idx < c.input_tids.size(); ++idx) {
      auto tid = c.input_tids[idx];
      assigned_nodes[idx] = _find_node_to_fetch(tid, tensor_locations, costs);
    }

    // find the node with the cheapest compute cost as the node where we reduce to
    // it is not really a good heuristic but whatever...
    float currBestCost = std::numeric_limits<float>::max();
    node_id_t best_node = 0;

    // go through all the nodes and figure out the compute overhead
    for(node_id_t node = 0; node < num_nodes; ++node) {

      // if this node is not one of the inputs nodes skip it... TODO make this faster 
      if(std::find(assigned_nodes.begin(), assigned_nodes.end(), node) == assigned_nodes.end()) {
        continue;
      }

      // the transfer overhead
      auto compute_overhead = std::max<float>(costs[node].compute_cost + ud_info.cost - max_compute_cost, 0);
      if(compute_overhead < currBestCost) {
        currBestCost = compute_overhead;
        best_node = node;
      }
    }

    // we got some compute (do it only in this one node, technically more of them have compute by we need to load balance)
    costs[best_node].compute_cost += ud_info.cost;
 
    // init the inputs
    std::vector<command_t::tid_node_id_t> inputs(c.input_tids.size());
    for(int32_t idx = 0; idx < c.input_tids.size(); ++idx) {
      
      // if the tensor is local no need to fetch
      auto it = tensor_locations[best_node].find(c.input_tids[idx]);
      if(it == tensor_locations[best_node].end()) {
        inputs[idx] = (command_t::tid_node_id_t{.tid = c.input_tids[idx], .node = assigned_nodes[idx]});
        continue;
      }

      // we are fetching it
      inputs[idx] = (command_t::tid_node_id_t{.tid = c.input_tids[idx], .node = best_node});
    }

    // init the parameters
    std::vector<command_param_t> params(c.params.size());
    for(int32_t idx = 0; idx < c.params.size(); ++idx) {
      params[idx] = c.params[idx];
    }

    // make the reduce
    auto cmd = command_t::create_reduce(cur_cmd++, 
                                        ud_info.ud->impl_id, 
                                        ud_info.is_gpu, 
                                        params, 
                                        inputs, 
                                        command_t::tid_node_id_t{.tid = c.output_tids[0], 
                                                                 .node = best_node});

    
    // mark that we are creating a tensor here
    tensor_locations[best_node].insert(c.output_tids[0]);

    // return the command
    out_commands.push_back(std::move(cmd));
  }


  void generate_apply(command_id_t &cur_cmd,
                      std::vector<bbts::command_ptr_t> &out_commands,
                      const abstract_command_t &c, 
                      std::vector<node_cost_t> &costs,
                      std::vector<std::unordered_set<tid_t>> &tensor_locations,
                      float &max_compute_cost,
                      float &max_transfer_cost) {
    
    // unique inputs
    std::vector<std::tuple<tid_t, size_t>> unique_inputs; unique_inputs.resize(10);

    // get the compute cost of running this
    command_param_list_t raw_param = {._data = c.params.data(), ._num_elements = c.params.size()};
    auto ud_info = cost_model->get_ud_cost(c.ud_id, 
                                           bbts::ud_impl_t::tensor_params_t{._params = raw_param}, 
                                           c.input_tids);
    
    // find all the unique inputs
    _find_unique_inputs(c.input_tids, unique_inputs);

    // the best node
    float currBestCost = std::numeric_limits<float>::max();
    node_id_t best_node = 0;

    // go through all the nodes and try to assign 
    for(node_id_t node = 0; node < num_nodes; ++node) {
      
      // find the transfer cost for this node
      float total_transfer_cost = 0;
      for(auto &u : unique_inputs) {
        
        // if it is not here we need to move it
        if(tensor_locations[node].find(std::get<0>(u)) == tensor_locations[node].end()) {
          total_transfer_cost += std::get<1>(u);
        }
      }

      // the transfer overhead
      auto compute_overhead = std::max<float>(costs[node].compute_cost + ud_info.cost - max_compute_cost, 0);
      auto transfer_overhead = std::max<float>(costs[node].transfer_cost + total_transfer_cost - max_transfer_cost, 0);

      // store the best
      if(currBestCost > (compute_overhead + transfer_overhead)) {
        best_node = node;
        currBestCost = compute_overhead + transfer_overhead;
      }
    }

    // we figured out the best node, now we need to make all the move ops if necessary
    for(auto &u : unique_inputs) {
        
        // if it is not here we need to move it
        if(tensor_locations[best_node].find(std::get<0>(u)) == tensor_locations[best_node].end()) {
          
          auto it = _moved_tensors.find(std::get<0>(u));
          if(it == _moved_tensors.end()) {

            // find the node to fetch from
            auto from_node = _find_node_to_fetch(std::get<0>(u), tensor_locations, costs);

            // create the move command to this node
            _moved_tensors[std::get<0>(u)] = cur_cmd;
            auto cmd = command_t::create_move(cur_cmd++, command_t::tid_node_id_t{.tid = std::get<0>(u), .node = from_node},
                                                         command_t::tid_node_id_t{.tid = std::get<0>(u), .node = best_node});

            // increase the transfer cost on this node
            costs[from_node].transfer_cost += std::get<1>(u);
            costs[best_node].transfer_cost += std::get<1>(u);

            // store the command
            out_commands.push_back(std::move(cmd));
            tensor_locations[best_node].insert(std::get<0>(u));
          }
          else  {
            
            // the command 
            auto cmd_id = it->second;

            // update the move op
            _update_move(out_commands, cmd_id, best_node);
            costs[best_node].transfer_cost += std::get<1>(u);

            // mark the location
            tensor_locations[best_node].insert(std::get<0>(u));
          }
        }
    }

    // init the inputs
    std::vector<command_t::tid_node_id_t> inputs(c.input_tids.size());
    for(int32_t idx = 0; idx < c.input_tids.size(); ++idx) {
      inputs[idx] = command_t::tid_node_id_t{.tid = c.input_tids[idx], .node = best_node};
    }

    // init the outputs
    std::vector<command_t::tid_node_id_t> outputs(c.output_tids.size());
    for(int32_t idx = 0; idx < c.output_tids.size(); ++idx) {

      // store the output location
      outputs[idx] = command_t::tid_node_id_t{.tid = c.output_tids[idx], .node = best_node};

      // mark that we are creating a tensor here
      tensor_locations[best_node].insert(c.output_tids[idx]);
    }

    // init the parameters
    std::vector<command_param_t> params(c.params.size());
    for(int32_t idx = 0; idx < c.params.size(); ++idx) {
      params[idx] = c.params[idx];
    }

    // make the apply
    auto cmd = command_t::create_apply(cur_cmd++, 
                                        ud_info.ud->impl_id, 
                                        ud_info.is_gpu, 
                                        params, 
                                        inputs, 
                                        outputs);
    
    // we got some compute
    costs[best_node].compute_cost += ud_info.cost;

    // update the meta for the ud function
    raw_param = {._data = c.params.data(), ._num_elements = c.params.size()};
    cost_model->update_meta_for_ud(c.ud_id, 
                                   ud_info.is_gpu, 
                                   bbts::ud_impl_t::tensor_params_t{._params = raw_param}, 
                                   c.input_tids, 
                                   c.output_tids);

    // move the command
    out_commands.push_back(std::move(cmd));
  }

  node_id_t _find_node_to_fetch(tid_t id,
                                const std::vector<std::unordered_set<tid_t>> &tensor_locations,
                                const std::vector<node_cost_t> &costs) {
    
    // init to find the best
    float currBestCost = std::numeric_limits<float>::max();
    node_id_t bestNode = -1;

    // check every node
    for(tid_t node = 0; node < num_nodes; ++node) {

      // check if the tensor is on this node
      auto it = tensor_locations[node].find(id);
      if(it != tensor_locations[node].end()) {

        // is this the best option
        auto c = costs[node].transfer_cost;
        if(c < currBestCost) {
          currBestCost = c;
          bestNode = node;
        }
      }
    }

    // return the node
    return bestNode;
  }

  node_id_t _get_can_be_local(const abstract_command_t &c,
                              std::vector<std::unordered_set<tid_t>> &tensor_locations) {
    
    // go through every node
    for(tid_t node = 0; node < num_nodes; ++node) {
      
      // check if every tensor is present on the node
      bool fine = true;
      for(auto i : c.input_tids) {
        
        if(tensor_locations[node].find(i) == tensor_locations[node].end()) {
          fine = false;
          break;
        }
      }

      // are they all on the same node
      if(fine) {
        return node;
      }
    }

    return -1;
  }

  // find all the unique input tids
  void _find_unique_inputs(const std::vector<tid_t> &input, std::vector<std::tuple<tid_t, size_t>> &output)  {
    output.clear();
    for(auto t : input) {
      if(std::find_if(output.begin(), output.end(), [&](const std::tuple<tid_t, size_t> item) { return std::get<0>(item) == t; }) == output.end()) {
        output.push_back({t, cost_model->get_transfer_cost(t)});
      }
    }
  }

  // update the move op
  void _update_move(std::vector<bbts::command_ptr_t> &out_commands, command_id_t &cmd_id, node_id_t best_node) {

    // store the previous command
    auto cmd = std::move(out_commands[cmd_id]);

    // copy the previous
    std::vector<command_t::tid_node_id_t> out; out.reserve(cmd->get_num_outputs() + 1);
    for(int32_t idx = 0; idx < cmd->get_num_outputs(); ++idx) {
      out.push_back(cmd->get_output(idx));
    }
    out.push_back(command_t::tid_node_id_t{.tid = cmd->get_input(0).tid, .node = best_node});

    // store the command
    out_commands[cmd_id] = command_t::create_broadcast(cmd_id, cmd->get_input(0), out);
  }

  void _insert_deletions(std::vector<bbts::command_ptr_t> &out_commands, 
                         command_id_t cur_cmd,
                         std::vector<std::unordered_set<tid_t>> &tensor_locations) {
    
    // 
    std::vector<command_t::tid_node_id_t> in(1);
    for(auto &it : _moved_tensors) {
      
      // resize the inputs
      auto cmd = out_commands[it.second].get();

      // make a delete for each of them
      for(size_t idx = 0; idx < cmd->get_num_outputs(); idx++) {

        // make sure the tensor is acutally located there
        in[0] = cmd->get_output(idx);
        auto jt = tensor_locations[in[0].node].find(in[0].tid);
        if(jt != tensor_locations[in[0].node].end()) {

          // delete it
          tensor_locations[in[0].node].erase(jt);
          out_commands.push_back(command_t::create_delete(cur_cmd++, in));
        }
      }
    }
  }

  // the cost model
  cost_model_ptr_t cost_model;

  // the number of nodes
  size_t num_nodes;

  // the tensors what are moved
  std::unordered_map<tid_t, command_id_t> _moved_tensors;
};

struct compile_source_file_t {

  // specifies all the functions used
  std::vector<abstract_ud_spec_t> function_specs;

  // specifies all the commands used
  std::vector<abstract_command_t> commands;

  void write_to_file(std::ofstream &file) {

    // write number of functions
    file << function_specs.size() << "\n";

    // write the function specs
    for(auto &fn : function_specs) {
      fn.write_to_file(file);
    }

    // write the command numbers
    file << commands.size() << '\n';

    // write the commands
    for(auto &cmd : commands) {
      cmd.write_to_file(file);
    }
  }

  void read_from_file(std::ifstream &file) {

    // get the number of functions
    std::size_t num_funs;
    file >> num_funs;

    // read all the functions
    function_specs.resize(num_funs);
    for(auto &fn : function_specs) {
      fn.read_from_file(file);
    }

    // read the number of commands
    std::size_t num_cmds;
    file >> num_cmds;

    // read all the commands
    commands.resize(num_cmds);
    for(auto &cmd : commands) {
      cmd.read_from_file(file);
    }
  }

};


}