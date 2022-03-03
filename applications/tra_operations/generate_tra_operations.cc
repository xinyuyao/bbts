
#include "../../src/commands/compile_source_file.h"
#include "../../src/commands/two_layer_compiler.h"
#include "../../src/tensor/tensor.h"
#include "../../src/tensor/tensor_factory.h"
#include "../../src/server/node.h"
#include "../../src/utils/terminal_color.h"

#include <cstdint>
#include <map>
#include <type_traits>
#include <unistd.h>

using namespace bbts;

tid_t current_tid = 0; // tensor_id

const int32_t UNFORM_ID = 0;
const int32_t ADD_ID = 1;
const int32_t MULT_ID = 2;

// commands here includes apply, reduce, and delete
void generate_matrix_commands(int32_t num_row, int32_t num_cols, int32_t row_split,
                     int32_t col_spilt,
                     std::vector<abstract_command_t> &commands,
                     std::map<std::tuple<int32_t, int32_t>, tid_t> &index) {

  std::vector<command_param_t> param_data = {
      command_param_t{.u = (std::uint32_t)(num_row / row_split)},
      command_param_t{.u = (std::uint32_t)(num_cols / col_spilt)},
      command_param_t{.f = 1.0f}, command_param_t{.f = 2.0f}};

  for (auto row_id = 0; row_id < row_split; row_id++) {
    for (auto col_id = 0; col_id < col_spilt; col_id++) {

      index[{row_id, col_id}] = current_tid;

      // store the command
      commands.push_back(
          abstract_command_t{.ud_id = UNFORM_ID,
                             .type = abstract_command_type_t::APPLY,
                             .input_tids = {},
                             .output_tids = {current_tid++},
                             .params = param_data});
    }
  }
}

void generate_aggregation_commands(int32_t num_row, int32_t num_cols, int32_t row_split,
                          int32_t col_spilt,
                          std::vector<abstract_command_t> &commands,
                          std::map<std::tuple<int32_t, int32_t>, tid_t> &index,
                          std::string dimension,
                          abstract_ud_spec_id_t kernel_func) { //Find a way to pass dynamic library

  std::vector<command_param_t> param_data = {};
  std::vector<tid_t> input_tids_list;

  if(dimension.compare("") == 0) {
    for (auto row_id = 0; row_id < row_split; row_id++) {
      for (auto col_id = 0; col_id < col_spilt; col_id++) {
        input_tids_list.push_back(index[{row_id, col_id}]);
      }
    }
    // store the command
    commands.push_back(
        abstract_command_t{.ud_id = kernel_func,
                           .type = abstract_command_type_t::APPLY,
                           .input_tids = input_tids_list,
                           .output_tids = {current_tid++},
                           .params = param_data});
  }
  else{
    for (auto row_id = 0; row_id < row_split; row_id++) {
      for (auto col_id = 0; col_id < col_spilt; col_id++) {
        input_tids_list.push_back(index[{row_id, col_id}]);
      }
      commands.push_back(
        abstract_command_t{.ud_id = kernel_func,
                           .type = abstract_command_type_t::APPLY,
                           .input_tids = input_tids_list,
                           .output_tids = {current_tid++},
                           .params = param_data});
      input_tids_list.clear();
    }
  }
}

  

void generate_join_commands(int32_t num_row, int32_t num_cols, int32_t row_split,
                   int32_t col_spilt,
                   std::vector<abstract_command_t> &commands,
                   std::map<std::tuple<int32_t, int32_t>, tid_t> &index,
                   std::string joinKeysL,
                   std::string joinKeysR,
                   abstract_ud_spec_id_t kernel_func) { //Find a way to pass dynamic library

  std::vector<command_param_t> param_data = {};

  for (auto row_id = 0; row_id < row_split; row_id++) {
    for (auto col_id = 0; col_id < col_spilt; col_id++) {
      for (auto sub_col_id = 0; sub_col_id < col_spilt; sub_col_id++) {
        std::vector<tid_t> input_tids_list;
        if(joinKeysL == "0"){
          if(joinKeysR == "0"){
            input_tids_list = {index[{col_id, row_id}], index[{col_id, sub_col_id}]};
          }
          else{
            input_tids_list = {index[{col_id, row_id}], index[{sub_col_id, col_id}]};
          }
        }
        else{
          if(joinKeysR == "0"){
            input_tids_list = {index[{row_id, col_id}], index[{col_id, sub_col_id}]};
          }
          else{
            input_tids_list = {index[{row_id, col_id}], index[{sub_col_id, col_id}]};
          }
        }
        commands.push_back(
        abstract_command_t{.ud_id = kernel_func,
                           .type = abstract_command_type_t::APPLY,
                           .input_tids = input_tids_list,
                           .output_tids = {current_tid++},
                           .params = param_data});
      }
    }
  }
}

//DO NOT NEED COMMAND
// void generate_reKey(){}
// void generate_filter(){}
// void generate_transform(){}
// void generate_tile(){}
// void generate_concat(){}

int main() {
  //*********************************** test aggregation ***********************************
  int32_t num_rows = 10;
  int32_t num_cols = 10;
  int32_t row_split = 2;
  int32_t col_split = 2;
  
  // the functions
  std::vector<abstract_ud_spec_t> funs_agg;

 
  // commands
  std::vector<abstract_command_t> commands_agg;

  std::map<std::tuple<int32_t, int32_t>, tid_t> index_agg;

  generate_matrix_commands(num_rows,num_cols, row_split, col_split, commands_agg, index_agg);

  generate_aggregation_commands(num_rows,num_cols, row_split, col_split, commands_agg, index_agg, "0" ,1);
  

  // specify functions
  funs_agg.push_back(abstract_ud_spec_t{.id = UNFORM_ID,
                                    .ud_name = "uniform",
                                    .input_types = {},
                                    .output_types = {"dense"}});
  
  std::vector<std::string> input_types_list(col_split,"dense");
  funs_agg.push_back(abstract_ud_spec_t{.id = ADD_ID,
                                    .ud_name = "matrix_add",
                                    .input_types = input_types_list,
                                    .output_types = {"dense"}});
  
  // write out the commands
  std::ofstream gen("TRA_commands_agg.sbbts");
  compile_source_file_t gsf{.function_specs = funs_agg, .commands = commands_agg};
  gsf.write_to_file(gen);
  gen.close();
  

  //*********************************** test join ***********************************
  // the functions
  std::vector<abstract_ud_spec_t> funs_join;

  // commands
  std::vector<abstract_command_t> commands_join;

  std::map<std::tuple<int32_t, int32_t>, tid_t> index_join;

  generate_matrix_commands(num_rows,num_cols, row_split, col_split, commands_join, index_join);

  generate_join_commands(num_rows,num_cols, row_split, col_split, commands_join, index_join, "0", "1", MULT_ID);

  // specify functions
  funs_join.push_back(abstract_ud_spec_t{.id = UNFORM_ID,
                                    .ud_name = "uniform",
                                    .input_types = {},
                                    .output_types = {"dense"}});

  funs_join.push_back(abstract_ud_spec_t{.id = MULT_ID,
                                    .ud_name = "matrix_mult",
                                    .input_types = {"dense", "dense"},
                                    .output_types = {"dense"}});
  
  // write out the commands
  std::ofstream gen_join("TRA_commands_join.sbbts");
  compile_source_file_t gsf_join{.function_specs = funs_join, .commands = commands_join};
  gsf_join.write_to_file(gen_join);
  gen_join.close();
  

  

  return 0;
}