#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../src/commands/command.h"
#include "../src/tensor/tensor.h"
#include "../src/tensor/tensor_factory.h"
#include "../src/ud_functions/ud_function.h"
#include "../src/ud_functions/udf_manager.h"


namespace bbts {

enum class abstract_command_type_t {
  APPLY,
  REDUCE
};

using abstract_ud_spec_id_t = int32_t;

// type:APPLY  fun:fun_id in:t1,t2,t2.. out:t3,t4.. 
// type:REDUCE fun:fun_id in:t1,t2,t2.. out:t3,t4.. 
// type:APPLY  fun:fun_id in:t1,t2,t2.. out:t3,t4.. 

struct abstract_ud_spec_t {

  // the id
  abstract_ud_spec_id_t id;

  // the name of the ud fuction
  std::string ud_name;

  // the input types of the ud function
  std::vector<std::string> input_types;
 
  // the output types
  std::vector<std::string> output_types; 
};

class cost_model_t {
private:
  // the matched function
  struct function_match_t {

    // the cpu implementation
    ud_impl_t* cpu;

    // the gpu implementation
    ud_impl_t* gpu;
  };

public:

  // the choice we made
  struct ud_choice_t {

    // the cost
    float cost;

    // is this using the gpu
    bool is_gpu;

    // the pointer
    ud_impl_t* ud;
  };

  cost_model_t(std::unordered_map<tid_t, tensor_meta_t> meta,
               const std::vector<abstract_ud_spec_t> &funs,
               tensor_factory_ptr_t tf, 
               udf_manager_ptr manager, 
               float gpu_transfer_cost_per_byte, 
               float send_cost_per_byte) : meta(std::move(meta)),
                                           tf(std::move(tf)),
                                           udm(std::move(manager)),
                                           send_cost_per_byte(send_cost_per_byte),
                                           gpu_transfer_cost_per_byte(gpu_transfer_cost_per_byte) {
    
    
    for(const auto &f : funs) {

      // get the matcher
      auto matcher = manager->get_matcher_for(f.ud_name);

      // the inplementation
      auto impl_cpu = matcher->findMatch(f.input_types, f.output_types, false);
      auto impl_gpu = matcher->findMatch(f.input_types, f.output_types, true);

      if(impl_cpu == nullptr && impl_gpu == nullptr) {
        throw std::runtime_error("Could not find a matching ud function.");
      }

      // found the match store it
      matched_functions[f.id] = function_match_t{.cpu = impl_cpu, .gpu = impl_gpu};
    }
  }

  // figure out the kernel cost
  ud_choice_t get_ud_cost(abstract_ud_spec_id_t fn, 
                          const bbts::ud_impl_t::tensor_params_t &params,
                          const std::vector<tid_t> input_tids) {
    
    // find the implementations for this kernel
    auto impls = matched_functions[fn];

    // make the meta
    input_meta.resize(input_tids.size());
    for(size_t idx = 0; idx < input_tids.size(); ++idx) {

      auto it = meta.find(input_tids[idx]);
      if(it == meta.end()) {
        throw std::runtime_error("Failed to find the meta.");
      }

      input_meta.set(idx, it->second);
    }

    // do we only have one implementation
    if(impls.gpu == nullptr) {
      return ud_choice_t{.cost = (float) impls.cpu->get_complexity_hint(params, input_meta), .is_gpu = false, .ud = impls.cpu};
    }
    else if(impls.cpu == nullptr) {
      return ud_choice_t{.cost = (float) impls.gpu->get_complexity_hint(params, input_meta), .is_gpu = true, .ud = impls.gpu};
    }

    // figure out the cpu transfer cost
    float cpu_cost = (float) impls.cpu->get_complexity_hint(params, input_meta);

    // figure out the gpu compute cost
    float gpu_compute_cost = (float) impls.gpu->get_complexity_hint(params, input_meta);
    
    // figure out the gpu transfer cost
    float gpu_transfer_cost = 0;
    for(size_t idx = 0; idx < input_meta.num_args(); ++idx) {
      gpu_transfer_cost += gpu_transfer_cost_per_byte * this->tf->get_tensor_size(input_meta.get_by_idx(idx));
    }

    // check if we pick the cpu
    if(cpu_cost < std::max(gpu_transfer_cost, gpu_compute_cost)) {
      return ud_choice_t{.cost = cpu_cost, .is_gpu = false, .ud = impls.cpu};
    }

    // we pick the gpu
    return ud_choice_t{.cost = std::max(gpu_transfer_cost, gpu_compute_cost), .is_gpu = true, .ud = impls.gpu}; 
  } 

  float get_transfer_cost(tid_t tensor) {
    return send_cost_per_byte * this->tf->get_tensor_size(meta[tensor]);
  }

  void update_meta_for_ud(abstract_ud_spec_id_t fn,
                          bool use_gpu,
                          const bbts::ud_impl_t::tensor_params_t &params,
                          const std::vector<tid_t> input_tids,
                          const std::vector<tid_t> output_tids) {

    // make the meta
    input_meta.resize(input_tids.size());
    for(size_t idx = 0; idx < input_tids.size(); ++idx) {

      auto it = meta.find(input_tids[idx]);
      if(it == meta.end()) {
        throw std::runtime_error("Failed to find the meta.");
      }

      input_meta.set(idx, it->second);
    }

    // find the implementations for this kernel
    auto impls = matched_functions[fn];

    // are we using the gpu
    if(!use_gpu) {

      // resize the out meta
      assert(output_tids.size() == impls.cpu->outputTypes.size());
      out_meta.resize(impls.cpu->outputTypes.size());
      impls.cpu->get_out_meta(params, input_meta, out_meta);
    }
    else {

      // resize the out meta
      assert(output_tids.size() == impls.gpu->outputTypes.size());
      out_meta.resize(impls.gpu->outputTypes.size());
      impls.gpu->get_out_meta(params, input_meta, out_meta);
    }

    // store the meta
    for(auto idx = 0; idx < output_tids.size(); ++idx) {
      meta[output_tids[idx]] = out_meta.get_by_idx(idx);
    }
  }

private:

  // all the matched functions
  std::unordered_map<abstract_ud_spec_id_t, function_match_t> matched_functions;

  // the meta so far in the cost model
  std::unordered_map<tid_t, tensor_meta_t> meta;

  // the tensor factory
  tensor_factory_ptr_t tf;

  // the udf manager
  udf_manager_ptr udm;

  // what is the extimated cost of moving bytes
  float send_cost_per_byte;

  // what is the estimated cost of moving bytes to the gpu
  float gpu_transfer_cost_per_byte;

  // we use this so that we don't keep allocating 
  ud_impl_t::meta_args_t input_meta;

  // the output meta so that we don't keep allocating 
  ud_impl_t::meta_args_t out_meta;
};

using cost_model_ptr_t = std::shared_ptr<cost_model_t>;


struct abstract_command_t {

  // the type of the command (APPY or REDUCE)
  abstract_command_type_t type;

  // the input tids
  std::vector<tid_t> input_tids;

  // the output tids
  std::vector<tid_t> output_tids;

  // parameters
  bbts::ud_impl_t::tensor_params_t params;

  // the type 
  abstract_ud_spec_id_t ud_id;
};

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
    
    command_id_t cur_cmd = 0;

    // the costs so far
    std::vector<node_cost_t> costs(num_nodes);



    // go compile all the commands
    std::vector<bbts::command_ptr_t> out_commands;
    for(auto &c : commands) {

      // if the command is an APPLY then we need to make sure that all the tensors are on the same node
      if(c.type == abstract_command_type_t::APPLY) {

        // store the command
        out_commands.push_back(std::move(generate_apply(cur_cmd,
                                                        out_commands,
                                                        c, 
                                                        costs,
                                                        tensor_locations)));
      }
      else {

        // store the command
        out_commands.push_back(std::move(generate_reduce(cur_cmd,
                                                         out_commands,
                                                         c, 
                                                         costs,
                                                         tensor_locations)));
      }
    }

    return std::move(out_commands);
  }

private:

  command_ptr_t generate_reduce(command_id_t &cur_cmd,
                                std::vector<bbts::command_ptr_t> &out_commands,
                                const abstract_command_t &c, 
                                std::vector<node_cost_t> &costs,
                                std::vector<std::unordered_set<tid_t>> &tensor_locations) {

    // // get the compute cost of running this
    // auto ud_info = cost_model->get_reduce_cost(c.ud_id, c.params, c.input_tids);

    // // the reduce op does not need all the 
    // std::vector<command_t::tid_node_id_t> inputs(c.input_tids.size());
    // for(int32_t idx = 0; idx < c.input_tids.size(); ++idx) {
      
    //   auto u = c.input_tids[idx];
    //   if(tensor_locations[best_node].find(u) != tensor_locations[best_node].end()) {

    //     // 
    //     inputs.push_back(command_t::tid_node_id_t{.tid = c.input_tids[idx], .node = best_node});

    //     // we have some calculation here
    //     costs[best_node].compute_cost += ud_info.cost;
    //   }
    //   else {

    //       // find the node to fetch from
    //       auto from_node = _find_node_to_fetch(u, tensor_locations, costs);
    //       inputs.push_back(command_t::tid_node_id_t{.tid = c.input_tids[idx], .node = from_node});
          
    //       // increase the transfer cost on this node
    //       costs[from_node].transfer_cost += cost_model->get_transfer_cost(u);
    //       costs[best_node].transfer_cost += cost_model->get_transfer_cost(u);

    //       // we have some calculation here
    //       costs[from_node].compute_cost += ud_info.cost;
    //   }
    // }

    // // init the parameters
    // std::vector<command_param_t> params(c.params.num_parameters());
    // for(int32_t idx = 0; idx < c.params.num_parameters(); ++idx) {
    //   params[idx] = c.params.get_raw(idx);
    // }

    // // make the reduce
    // auto cmd = command_t::create_reduce(cur_cmd++, 
    //                                     ud_info.ud->impl_id, 
    //                                     ud_info.is_gpu, 
    //                                     params, 
    //                                     inputs, 
    //                                     { .tid = c.output_tids.front(), .node = best_node });

    // return std::move(cmd);
  }


  command_ptr_t generate_apply(command_id_t &cur_cmd,
                               std::vector<bbts::command_ptr_t> &out_commands,
                               const abstract_command_t &c, 
                               std::vector<node_cost_t> &costs,
                               std::vector<std::unordered_set<tid_t>> &tensor_locations) {
    
    float max_compute_cost = 0;
    float max_transfer_cost = 0;

    // unique inputs
    std::vector<std::tuple<tid_t, size_t>> unique_inputs; unique_inputs.resize(10);

    // get the compute cost of running this
    auto ud_info = cost_model->get_ud_cost(c.ud_id, c.params, c.input_tids);
    
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
      if(currBestCost < (compute_overhead + transfer_overhead)) {
        best_node = node;
        currBestCost = compute_overhead + transfer_overhead;
      }
    }

    // we figured out the best node, now we need to make all the move ops if necessary
    for(auto &u : unique_inputs) {
        
        // if it is not here we need to move it
        if(tensor_locations[best_node].find(std::get<0>(u)) == tensor_locations[best_node].end()) {
          
          // find the node to fetch from
          auto from_node = _find_node_to_fetch(std::get<0>(u), tensor_locations, costs);

          // create the move command to this node
          auto cmd = command_t::create_move(cur_cmd++, command_t::tid_node_id_t{.tid = std::get<0>(u), .node = from_node},
                                                      command_t::tid_node_id_t{.tid = std::get<0>(u), .node = best_node});

          // increase the transfer cost on this node
          costs[from_node].transfer_cost += std::get<1>(u);
          costs[best_node].transfer_cost += std::get<1>(u);

          // store the command
          out_commands.push_back(std::move(cmd));
        }
    }

    // init the inputs
    std::vector<command_t::tid_node_id_t> inputs(c.input_tids.size());
    for(int32_t idx = 0; idx < c.input_tids.size(); ++idx) {
      inputs.push_back(command_t::tid_node_id_t{.tid = c.input_tids[idx], .node = best_node});
    }

    // init the outputs
    std::vector<command_t::tid_node_id_t> outputs(c.output_tids.size());
    for(int32_t idx = 0; idx < c.output_tids.size(); ++idx) {
      outputs.push_back(command_t::tid_node_id_t{.tid = c.output_tids[idx], .node = best_node});
    }

    // init the parameters
    std::vector<command_param_t> params(c.params.num_parameters());
    for(int32_t idx = 0; idx < c.params.num_parameters(); ++idx) {
      params[idx] = c.params.get_raw(idx);
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
    cost_model->update_meta_for_ud(c.ud_id, 
                                   ud_info.is_gpu, 
                                   c.params, 
                                   c.input_tids, 
                                   c.output_tids);

    // move the command
    return std::move(cmd);
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

  // find all the unique input tids
  void _find_unique_inputs(const std::vector<tid_t> &input, std::vector<std::tuple<tid_t, size_t>> &output)  {
    output.clear();
    for(auto t : input) {
      if(std::find_if(output.begin(), output.end(), [&](const std::tuple<tid_t, size_t> item) { return std::get<0>(item) == t; }) == output.end()) {
        output.push_back({t, cost_model->get_transfer_cost(t)});
      }
    }
  }

  // the cost model
  cost_model_ptr_t cost_model;

  // the number of nodes
  size_t num_nodes;

};

TEST(TestCommandCompiler, Test1) {


}

}