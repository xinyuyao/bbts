
#include "../../src/commands/compile_source_file.h"
#include "../../src/commands/two_layer_compiler.h"
#include "../../src/tensor/tensor.h"
#include <cstdint>
#include <map>

using namespace bbts;

tid_t current_tid = 0;

const int32_t UNFORM_ID = 0;
const int32_t ADD_ID = 1;


// 4 x 4 
// (0, 0)  1 1 | (0, 1) 2 2
//         1 1 |        2 2
// ---------
// 3 3 | 4 4
// 3 3 | 4 4 

void generate_matrix(int32_t num_row, int32_t num_cols, int32_t row_split,
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

void generate_addition(int32_t num_row, int32_t num_cols, int32_t row_split,
                       int32_t col_spilt,
                       std::vector<abstract_command_t> &commands,
                       std::map<std::tuple<int32_t, int32_t>, tid_t> &a_index,
                       std::map<std::tuple<int32_t, int32_t>, tid_t> &b_index) {

  std::vector<command_param_t> param_data = {};

  for (auto row_id = 0; row_id < row_split; row_id++) {
    for (auto col_id = 0; col_id < col_spilt; col_id++) {

      // store the command
      commands.push_back(
          abstract_command_t{.ud_id = ADD_ID,
                             .type = abstract_command_type_t::APPLY,
                             .input_tids = {a_index[{row_id, col_id}], b_index[{row_id, col_id}]},
                             .output_tids = {current_tid++},
                             .params = param_data});
    }
  }
}

void sum_matrix() {}

int main() {

  // the functions
  std::vector<abstract_ud_spec_t> funs;

  // specify functions
  funs.push_back(abstract_ud_spec_t{.id = UNFORM_ID,
                                    .ud_name = "uniform",
                                    .input_types = {},
                                    .output_types = {"dense"}});

  funs.push_back(abstract_ud_spec_t{.id = ADD_ID,
                                    .ud_name = "matrix_add",
                                    .input_types = {"dense", "dense"},
                                    .output_types = {"dense"}});

  // commands
  std::vector<abstract_command_t> commands;

  std::map<std::tuple<int32_t, int32_t>, tid_t> a_index;
  std::map<std::tuple<int32_t, int32_t>, tid_t> b_index;
  generate_matrix(90, 90, 3, 3, commands, a_index);
  generate_matrix(90, 90, 3, 3, commands, b_index);
  generate_addition(90, 90, 3, 3, commands, a_index, b_index);

  // write out the commands
  std::ofstream gen("awesome_commands.sbbts");
  compile_source_file_t gsf{.function_specs = funs, .commands = commands};
  gsf.write_to_file(gen);
  gen.close();

  return 0;
}