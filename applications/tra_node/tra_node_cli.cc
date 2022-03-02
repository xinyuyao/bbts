#include <bits/stdint-intn.h>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <unistd.h>
#include <filesystem>
#include "../../src/tensor/tensor.h"
#include "../../src/tensor/tensor_factory.h"
#include "../../src/server/node.h"
#include "../../src/utils/terminal_color.h"


#include "sqlite3.h"

#include "../../third_party/cli/include/cli/cli.h"
#include "../../third_party/cli/include/cli/clifilesession.h"

#include <map>
#include <type_traits>

#include "../../src/commands/compile_source_file.h"
#include "../../src/commands/two_layer_compiler.h"

#include "cnpy.h"

#include <iostream>
#include <sstream>

#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "sqlite3_ops.cc"




using namespace cli;


static bbts::tid_t current_tid = 0; // tensor_id

const int32_t UNFORM_ID = 0;
const int32_t ADD_ID = 1;
const int32_t MULT_ID = 2;

bbts::udf_manager_ptr udf_manager;

std::thread loading_message(std::ostream &out, const std::string &s, std::atomic_bool &b) {

  auto t = std::thread([s, &out, &b]() {

    // as long as we load
    int32_t dot = 0;
    while(!b) {

      out << '\r' << s;
      for(int32_t i = 0; i < dot; ++i) { out << '.';}
      dot = (dot + 1) % 4;
      usleep(300000);
    }

    out << '\n';
  });

  return std::move(t);
}


//load user input data file
auto load_text_file(std::ostream &out, bbts::node_t &node, const std::string &file_path) {
  // kick off a loading message
  std::atomic_bool b; b = false;
  auto t = loading_message(out, "Loading the data text file", b);

  // try to open the file
  std::ifstream in(file_path, std::ifstream::ate);

  if(in.fail()) {
    // finish the loading message
    b = true; t.join();

    std::cout << bbts::red << "Failed to load the file " << file_path << '\n' << bbts::reset;
    return new char[0];
  }

  auto file_len = (size_t) in.tellg();
  in.seekg (0, std::ifstream::beg);

  auto file_bytes = new char[file_len];
  in.readsome(file_bytes, file_len);

  std::cout << "\nFile content:\n" << file_bytes << '\n';
  std::cout << "\nFile length is:" << file_len << '\n';

  // finish the loading message  
  b = true; t.join();

  return file_bytes;

}



bool load_shared_library(std::ostream &out, bbts::node_t &node, const std::string &file_path) {

  // kick off a loading message
  std::atomic_bool b; b = false;
  auto t = loading_message(out, "Loading the library file", b);

  // try to open the file
  std::ifstream in(file_path, std::ifstream::ate | std::ifstream::binary);

  if(in.fail()) {
    // finish the loading message
    b = true; t.join();

    std::cout << bbts::red << "Failed to load the file " << file_path << '\n' << bbts::reset;
    return false;
  }

  auto file_len = (size_t) in.tellg();
  in.seekg (0, std::ifstream::beg);

  auto file_bytes = new char[file_len];
  in.readsome(file_bytes, file_len);

  // finish the loading message  
  b = true; t.join();

  // kick off a registering message
  b = false;
  t = loading_message(out, "Registering the library", b);

  auto [did_load, message] = node.load_shared_library(file_bytes, file_len);
  delete[] file_bytes;

  // finish the registering message
  b = true; t.join();

  if(!did_load) {
    out << bbts::red << "Failed to register the library : \"" << message << "\"\n" << bbts::reset;
    return false;
  } else {
    out << bbts::green << message << bbts::reset;
    return true;
  }
}

bool generate_binary_file(std::ostream &out, bbts::node_t &node, const unsigned row_num, const unsigned col_num, const std::string &file){
  srand(0);
  //create random data
  std::vector<double> data(row_num * col_num);
  for(int i = 0;i < row_num * col_num;i++) data[i] = double(rand());

  //save it to file
  cnpy::npy_save(file,&data[0],{col_num, row_num},"w");

  
  return true;
}

cnpy::NpyArray load_binary_file(std::ostream &out, bbts::node_t &node, const std::string &file){
  //load it into a new array
  cnpy::NpyArray arr = cnpy::npy_load(file);
  
  size_t size_of_arr = 1;
  //print out shape of binary file
  out <<  "shape: ";
  out << "( ";
  for(size_t i: arr.shape){
    size_of_arr *= i;
    out << i << " ";
  }
  out << ")\n";

  //print out world size
  out <<  "word size: ";
  out << arr.word_size << "\n";

  double* loaded_data = arr.data<double>();
  
  out <<  "data: ";
  for(int i = 0; i < size_of_arr; i++) out<< loaded_data[i] << " ";
  out << "\n";

  
  //make sure the loaded data matches the saved data
  // assert(arr.word_size == sizeof(double));
  // assert(arr.shape.size() == 2 && arr.shape[0] == col_num && arr.shape[1] == row_num );
  // for(int i = 0; i < row_num * col_num;i++) assert(data[i] == loaded_data[i]);

  return arr;
}



// auto create_tensors_for_text_file(std::ostream &out, bbts::node_t &node, const std::string &file, const int32_t num_rows, 
//                     const int32_t num_cols, const int32_t row_split, const int32_t col_split){
  
//     std::string readline;
//     std::map<std::tuple<int32_t, int32_t>, std::vector<std::vector<int32_t>>> tensor_relation;

//     int32_t tensor_block[num_rows][num_cols];

//     std::ifstream data_file(file);

//     int32_t row_size = num_rows/row_split;
//     int32_t col_size = num_cols/col_split;
  
//     if (!data_file){
//       out << "There was an error opening the file.\n";
//     }
    
//     int row_idx = 0, col_idx = 0;
//     while(getline(data_file, readline)){
//       std::istringstream ss(readline);
//       int32_t readnum;
//       while(ss >> readnum){
//             tensor_block[row_idx][col_idx] = readnum;
//             // out << " ( " << row_idx << " , " << col_idx << " ): ";
//             // out << "number is: " << readnum << "\n";
//             col_idx++;
//             if(col_idx == num_cols){
//               row_idx++;
//               col_idx= 0;
//             }
       
//       }
//     }
    
//     for(int row_id = 0; row_id < row_split; row_id++){
//       for(int col_id = 0; col_id < row_split; col_id++){
//         std::tuple tensor_id = std::make_tuple(row_id, col_id);
//         // out << "\ntensor id: " << "( " << std::get<0>(tensor_id) << " , " << std::get<1>(tensor_id) << " )\n ";

//         std::vector<std::vector<int32_t>> subtensor_block;

//         for(int sub_row_id = 0; sub_row_id < row_size; sub_row_id++){
//           std::vector<int32_t> subtensor_block_row;
//           for(int sub_col_id = 0; sub_col_id < row_size; sub_col_id++){
//             subtensor_block_row.push_back(tensor_block[sub_row_id + row_id * row_size][sub_col_id + col_id * col_size]);
//           }
//           subtensor_block.push_back(subtensor_block_row);
//         }
//         tensor_relation.insert({tensor_id, subtensor_block});
//       }
//     }

//     for(std::map<std::tuple<int32_t, int32_t>, std::vector<std::vector<int32_t>>>::iterator iter = tensor_relation.begin(); iter != tensor_relation.end(); ++iter){
//       std::tuple key =  iter->first;
//       out << "\n( " << std::get<0>(key) << " , " << std::get<1>(key) << " ): ";
//       std::vector<std::vector<int32_t>> v = iter->second;
//       out << "[";
//       for(std::vector<int32_t> row: v){
//         out << "[";
//         for(int32_t col: row){
//           out <<  col << " ";
//         }
//         out << "]";
//       }
//       out << "]";
//     }
//     out << "\n\n\n";
// }


auto create_tensors(std::ostream &out, bbts::node_t &node, const std::string &file, const int32_t row_split, const int32_t col_split, const std::string &tensor_type){
  
  cnpy::NpyArray arr = load_binary_file(out, node, file);
  double* data = arr.data<double>();

  std::map<std::tuple<int32_t, int32_t>, std::vector<std::vector<double>>> tensor_relation;
  int32_t num_rows = arr.shape[0];
  int32_t num_cols = arr.shape[1];
  int32_t row_size = num_rows/row_split;
  int32_t col_size = num_cols/col_split;

  // create tensor relation map with (tra_id, tensor data)
  for(int r = 0; r < row_split; r++){
    for(int c = 0; c < col_split; c++){
      std::tuple tensor_id = std::make_tuple(r, c);
      std::vector<std::vector<double>> lists;
      for(int sub_r = 0; sub_r < row_size; sub_r++){
        std::vector<double> list;
        for(int sub_c = 0; sub_c < col_size; sub_c++){
          list.push_back(data[sub_r * 10 + sub_c + row_size*c + col_size*10*r]);
        }
        lists.push_back(list);
      }
      tensor_relation.insert({tensor_id, lists});
    }
  }

  mkdir("tensors", 0777);
  std::ofstream myfile;
  std::ofstream filelist;
  myfile.open("tensor_relation.txt");
  filelist.open("tensors/filelist.txt");
  

  //print out map
  for(std::map<std::tuple<int32_t, int32_t>, std::vector<std::vector<double>>>::iterator iter = tensor_relation.begin(); iter != tensor_relation.end(); ++iter){
      std::tuple key =  iter->first;
      std::string tensor_file = "t" + std::to_string(current_tid);
      std::ofstream tfile;
      tfile.open("tensors/" + tensor_file);
  
      out << "\n( " << std::get<0>(key) << " , " << std::get<1>(key) << " ): ";
      filelist << current_tid << "|" << tensor_type << "|" << tensor_file;
      myfile << current_tid++ << "|" << "2" << "|" << std::get<0>(key) << "|" << std::get<1>(key) << "|" << row_size << "|" << col_size << "|";
      tfile << row_size << "|" << col_size << "|";
      std::vector<std::vector<double>> v = iter->second;
      out << "[";
      for(std::vector<double> row: v){
        out << "[";
        for(double col: row){
          out <<  col << " ";
          myfile << col << " ";
          tfile << col << " ";
        }
        out << "]";
        // myfile << "|";
      }
      out << "]";
      myfile << "\n";
      filelist << "\n";
    }
    out << "\n\n\n";
  myfile.close();
   
}

void create_id_table(std::ostream &out, bbts::node_t &node,const std::string &db, std::map<bbts::tid_t, std::string> id_map){
  const char* db_name = db.c_str();
  // char* sql0 = "DROP TABLE TENSOR_IDS";
  // execute_command(sql0, db_name, 0);

  char* sql1 = "CREATE TABLE TENSOR_IDS(" \
              "TOS_ID INT PRIMARY KEY NOT NULL," \
              "TRA_ID VARCHAR(100) NOT NULL);";

  execute_command(sql1, db_name, 0);

  std::string sql_str;
  std::map<bbts::tid_t, std::string>::iterator it;
  for(it = id_map.begin(); it != id_map.end(); it++){
    sql_str += "INSERT INTO TENSOR_IDS (TOS_ID, TRA_ID)\n";
    sql_str += ("VALUES (\'" + std::to_string(it->first) + "\', \'" + it->second + "\'); \n");
  }

  const char* sql2 = sql_str.c_str();

  out << sql2;
  execute_command(sql2, db_name, 0);

}

void create_id_table(std::ostream &out, bbts::node_t &node, const std::string &tra_file,  const std::string &db){
  std::ofstream myfile;

  const char* db_name = db.c_str();
  // char* sql0 = "DROP TABLE TENSOR_IDS";
  // execute_command(sql0, db_name, 0);

  char* sql1 = "CREATE TABLE TENSOR_IDS(" \
              "TOS_ID INT PRIMARY KEY NOT NULL," \
              "TRA_ID VARCHAR(100) NOT NULL);";

  execute_command(sql1, db_name, 0);

  std::string readline;

  std::ifstream tra_data_file(tra_file);

  if (!tra_data_file){
    out << "There was an error opening the tra file.\n";
  }

  std::string sql_str;

  while(std::getline(tra_data_file, readline)){
    sql_str += "INSERT INTO TENSOR_IDS (TOS_ID, TRA_ID)\n";
    std::istringstream ss(readline);
    std::string read_number;
    std::string record;

    record += "\'";
    std::getline(ss, read_number,'|');
    record += read_number;
    record += "\', ";

    std::getline(ss, read_number, '|');
    out <<"read_number: " << read_number;

    int key_size = stoi(read_number);
    int i = 0;

    record += "\'";
    while(i < key_size && std::getline(ss, read_number, '|')){
      record += read_number;
      if(i != key_size -1){
        record += ",";
      }
      i++;
    }
    record += "\'";
    sql_str += ("VALUES (" + record + "); \n");
  }

  const char* sql2 = sql_str.c_str();

  out << sql2;
  execute_command(sql2, db_name, 0);


  myfile.close();

}

/***************************   implementing TRA operations for TOS.  *************************/
//generate TOS commands for generating matrix
void generate_matrix_commands(int32_t num_row, int32_t num_cols, int32_t row_split,
                     int32_t col_spilt,
                     std::vector<bbts::abstract_command_t> &commands,
                     std::map<std::tuple<int32_t, int32_t>, bbts::tid_t> &index) {
  
  std::ofstream myfile;
  myfile.open("tos_id.txt");

  std::vector<bbts::command_param_t> param_data = {
      bbts::command_param_t{.u = (std::uint32_t)(num_row / row_split)},
      bbts::command_param_t{.u = (std::uint32_t)(num_cols / col_spilt)},
      bbts::command_param_t{.f = 1.0f}, bbts::command_param_t{.f = 2.0f}};

  for (auto row_id = 0; row_id < row_split; row_id++) {
    for (auto col_id = 0; col_id < col_spilt; col_id++) {

      index[{row_id, col_id}] = current_tid;

      // store the command
      myfile << current_tid << "|";
      commands.push_back(
          bbts::abstract_command_t{.ud_id = UNFORM_ID,
                             .type = bbts::abstract_command_type_t::APPLY,
                             .input_tids = {},
                             .output_tids = {current_tid++},
                             .params = param_data});
      
    }
  }
  myfile.close();
}

//TRA operations: AGGREGATION//
// Generate commands for aggregation
void generate_aggregation_commands(std::ostream &out, int32_t num_rows, int32_t num_cols, int32_t row_split,
                          int32_t col_split,
                          std::vector<bbts::abstract_command_t> &commands,
                          std::map<std::tuple<int32_t, int32_t>, bbts::tid_t> &index,
                          std::string dimension,
                          bbts::abstract_ud_spec_id_t kernel_func, 
                          std::vector<bbts::tid_t> &output_tids_list,
                          std::vector<std::vector<bbts::tid_t>> &input_tids_list) { //Find a way to pass dynamic library
  

  // generate_matrix_commands(num_rows, num_cols, row_split, col_split, commands, index); 
  //TODO: find a way to keep up current_tid without calling generating matrix function inside aggregate

  std::vector<bbts::command_param_t> param_data = {};
  std::vector<bbts::tid_t> input_tids_sublist;

  // get the output result
  
  

  if(dimension.compare("") == 0) {
    for (auto row_id = 0; row_id < row_split; row_id++) {
      for (auto col_id = 0; col_id < col_split; col_id++) {
        input_tids_sublist.push_back(index[{row_id, col_id}]);
      }
    }
    // store the command
    output_tids_list.push_back(current_tid);
    commands.push_back(
        bbts::abstract_command_t{.ud_id = kernel_func,
                           .type = bbts::abstract_command_type_t::APPLY,
                           .input_tids = input_tids_sublist,
                           .output_tids = {current_tid++},
                           .params = param_data});
  }
  else if(dimension.compare("0") == 0){
    for (auto row_id = 0; row_id < row_split; row_id++) {
      for (auto col_id = 0; col_id < col_split; col_id++) {
        input_tids_sublist.push_back(index[{row_id, col_id}]);
      }
      output_tids_list.push_back(current_tid);
      commands.push_back(
        bbts::abstract_command_t{.ud_id = kernel_func,
                           .type = bbts::abstract_command_type_t::APPLY,
                           .input_tids = input_tids_sublist,
                           .output_tids = {current_tid++},
                           .params = param_data});
      input_tids_list.push_back(input_tids_sublist);
      input_tids_sublist.clear();
    }
  }
  else if(dimension.compare("1") == 0){
    for (auto col_id = 0; col_id < col_split; col_id++) {
      for (auto row_id = 0; row_id < row_split; row_id++) {
        input_tids_sublist.push_back(index[{row_id, col_id}]);
      }
      output_tids_list.push_back(current_tid);
      commands.push_back(
        bbts::abstract_command_t{.ud_id = kernel_func,
                           .type = bbts::abstract_command_type_t::APPLY,
                           .input_tids = input_tids_sublist,
                           .output_tids = {current_tid++},
                           .params = param_data});
      input_tids_list.push_back(input_tids_sublist);
      input_tids_sublist.clear();
    }
  }
}


void generate_aggregation_tra(std::ostream &out, bbts::node_t &node, std::vector<bbts::tid_t> output_tid_list, std::vector<std::vector<bbts::tid_t>> input_tid_list, const std::string &db, std::string dimension){

  const char* db_char= db.c_str();
  sqlite3_stmt * stmt;
  sqlite3 *sql_db;
  std::vector< std::vector < std:: string > > result;
  std::map<bbts::tid_t, std::string> id_map;
  std::map<bbts::tid_t, std::string> result_map;

  const char* sql = "SELECT * FROM TENSOR_IDS";
  // execute_command(sql, db_name, data);
  
  if(sqlite3_open(db_char, &sql_db) == SQLITE_OK){
     // preparing the statement
    int rc = sqlite3_prepare_v2(sql_db, sql,-1, &stmt, NULL);

    if(rc != SQLITE_OK){
      out << "ERRORS PREPARING THE STATEMENT";
      return;
    }
    // sqlite3_step(stmt); // executing the statement
    while((rc = sqlite3_step(stmt)) == SQLITE_ROW){
      bbts::tid_t tos_id = sqlite3_column_int(stmt, 0);
      std::string tra_id = std::string(reinterpret_cast< const char* >(sqlite3_column_text(stmt, 1)));
      id_map.insert(std::pair(tos_id, tra_id));
    }
  }
  else{
    std::cout << "Failed to open db\n";
  }

  sqlite3_finalize(stmt);
  sqlite3_close(sql_db);

  std::vector<std::vector<std::string>> input_tra_id_list;
  for (std::vector<bbts::tid_t> input_tid_sublist : input_tid_list){
    std::vector<std::string> input_tra_id_sublist;
    for(bbts::tid_t tid: input_tid_sublist){
      std::string tra_id = id_map.find(tid) -> second;
      input_tra_id_sublist.push_back(tra_id);
      out << "tra_id: " << tra_id << " ";
    }
    input_tra_id_list.push_back(input_tra_id_sublist);
  }

  

  if(dimension.compare("") == 0){
    std::string newKey = "";
    
    out << "new key: " << newKey << "\n";
    result_map.insert(std::pair(output_tid_list[0], newKey));
  }
  else if(dimension.compare("0") == 0){
    for (int i = 0; i < input_tra_id_list.size(); i++){
      std::string key = input_tra_id_list[i][0];
      std::string read_number;
      std::string newKey = "";
      std::istringstream ss(key);
      int j = 0;

      while(std::getline(ss, read_number, ',')){
        if(j == 0){
          newKey += read_number;
        }
        j++;
      }
      out << "new key: " << newKey << "\n";
      result_map.insert(std::pair(output_tid_list[i], newKey));
    } 
  }
  else if(dimension.compare("1") == 0){

   
   

    for (int i = 0; i < input_tra_id_list[0].size(); i++){
      std::string key = input_tra_id_list[0][i];
      std::istringstream ss(key);
      std::string newKey = "";
      std::string read_number;
      
      int j = 0;
      


      while(std::getline(ss, read_number, ',')){
        if(j == 1){
          newKey += read_number;
        }
        j++;
      }
      out << "new key: " << newKey << "\n";
      result_map.insert(std::pair(output_tid_list[i], newKey));
    } 
  }


  create_id_table(out, node, db, result_map);
}

  
//TRA operations: JOIN//
// Generate commands for aggregation
void generate_join_commands(std::ostream &out, bbts::node_t &node, int32_t num_rows, int32_t num_cols, int32_t row_split,
                            int32_t col_split,
                            std::vector<bbts::abstract_command_t> &commands,
                            std::map<std::tuple<int32_t, int32_t>, bbts::tid_t> &index,
                            std::string joinKeysL,
                            std::string joinKeysR,
                            bbts::abstract_ud_spec_id_t kernel_func, 
                            std::vector<bbts::tid_t> &output_tids_list,
                            std::vector<std::vector<bbts::tid_t>> &input_tids_list) { //Find a way to pass dynamic library
  
  std::vector<bbts::command_param_t> param_data = {};

  for (auto row_id = 0; row_id < row_split; row_id++) {
    for (auto col_id = 0; col_id < col_split; col_id++) {
      for (auto sub_col_id = 0; sub_col_id < col_split; sub_col_id++) {
        std::vector<bbts::tid_t> input_tids_sublist;
        if(joinKeysL == "0"){
          if(joinKeysR == "0"){
            input_tids_sublist = {index[{col_id, row_id}], index[{col_id, sub_col_id}]};
          }
          else{
            input_tids_sublist = {index[{col_id, row_id}], index[{sub_col_id, col_id}]};
          }
        }
        else{
          if(joinKeysR == "0"){
            input_tids_sublist = {index[{row_id, col_id}], index[{col_id, sub_col_id}]};
          }
          else{
            input_tids_sublist = {index[{row_id, col_id}], index[{sub_col_id, col_id}]};
          }
        }
        input_tids_list.push_back(input_tids_sublist);

        output_tids_list.push_back(current_tid);
        commands.push_back(
        bbts::abstract_command_t{.ud_id = kernel_func,
                           .type = bbts::abstract_command_type_t::APPLY,
                           .input_tids = input_tids_sublist,
                           .output_tids = {current_tid++},
                           .params = param_data});
      }
    }
  }
  out << "output_tids_list: ";
  for(bbts::tid_t output: output_tids_list){
    out << output << " ";
  }
  out << "\n";
  out << "input_tids_list: \n";
  for(std::vector<bbts::tid_t> input_sub: input_tids_list){
    for(bbts::tid_t id: input_sub){
      out << id << " ";
    }
    out << "\n";
  }
  out << "\n";
  
}



void generate_join_tra(std::ostream &out, bbts::node_t &node, std::vector<bbts::tid_t> output_tid_list, std::vector<std::vector<bbts::tid_t>> input_tid_list, const std::string &db, std::string joinKeysL, std::string joinKeysR){

  const char* db_char= db.c_str();
  sqlite3_stmt * stmt;
  sqlite3 *sql_db;
  std::vector< std::vector < std:: string > > result;
  std::map<bbts::tid_t, std::string> id_map;
  std::map<bbts::tid_t, std::string> result_map;

  const char* sql = "SELECT * FROM TENSOR_IDS";
  // execute_command(sql, db_name, data);
  
  if(sqlite3_open(db_char, &sql_db) == SQLITE_OK){
     // preparing the statement
    int rc = sqlite3_prepare_v2(sql_db, sql,-1, &stmt, NULL);

    if(rc != SQLITE_OK){
      out << "ERRORS PREPARING THE STATEMENT";
      return;
    }
    // sqlite3_step(stmt); // executing the statement
    while((rc = sqlite3_step(stmt)) == SQLITE_ROW){
      bbts::tid_t tos_id = sqlite3_column_int(stmt, 0);
      std::string tra_id = std::string(reinterpret_cast< const char* >(sqlite3_column_text(stmt, 1)));
      id_map.insert(std::pair(tos_id, tra_id));
    }
  }
  else{
    std::cout << "Failed to open db\n";
  }

  sqlite3_finalize(stmt);
  sqlite3_close(sql_db);

  std::vector<std::vector<std::string>> input_tra_id_list;
  out << "input_tra_id_list: \n";
  for (std::vector<bbts::tid_t> input_tid_sublist : input_tid_list){
    std::vector<std::string> input_tra_id_sublist;
    for(bbts::tid_t tid: input_tid_sublist){
      std::string tra_id = id_map.find(tid) -> second;
      input_tra_id_sublist.push_back(tra_id);
      out << tra_id << " ";
    }
    out << "\n";
    input_tra_id_list.push_back(input_tra_id_sublist);
  }

  for(int i = 0; i < input_tra_id_list.size(); i++){
    std::string keyL = input_tra_id_list[i][0];
    std::string keyR = input_tra_id_list[i][1];

    out << "keyL: " << keyL << "\n";
    out << "keyR: " << keyR << "\n";

    std::string joinKey;
    std::istringstream ssL(keyL);
    std::istringstream ssR(keyR);
    std::string read_number_L;
    std::string read_number_R;
    
    while(std::getline(ssL, read_number_L, ',')){
      out << "read_number_L: " << read_number_L << "\n";
      joinKey += (read_number_L + " ");
    }

    int j = 0;
    while(std::getline(ssR, read_number_R, ',')){
      out << "read_number_R: " << read_number_R << "\n";
      if(j != stoi(joinKeysR)){
        joinKey += read_number_R + " ";
      }
      j++;
    }

    out << "L: " << keyL << " R: " << keyR << "\n";
    out << "joinKey: " << joinKey << "\n";

    result_map.insert(std::pair(output_tid_list[i], joinKey));
    
  }
  create_id_table(out, node, db, result_map);
  

  // if(dimension.compare("0") == 0){
  //   for (int i = 0; i < input_tra_id_list.size(); i++){
  //     std::string key = input_tra_id_list[i][0];
  //     out << "old key: " << key << "\n";
  //     std::string newKey;
  //     std::istringstream ss(key);
  //     std::string read_number;
  //     std::vector<std::string> multiDimKey;
  //     int j = 0;

  //     while(std::getline(ss, read_number, ',')){
  //       if(j != 0){
  //         out << "read_number: " << read_number << "\n";
  //         multiDimKey.push_back(read_number);
  //       }
  //       j++;
  //     }
  //     for(int k = 0; k < multiDimKey.size(); k++){
  //       newKey += multiDimKey[k];
  //       if(k != multiDimKey.size() - 1){
  //         newKey += ",";
  //       }
  //     }
  //     out << "new key: " << newKey << "\n";
  //     result_map.insert(std::pair(output_tid_list[i], newKey));
  //   } 
  // }
  // else if(dimension.compare("1") == 0){
  //   for (int i = 0; i < input_tra_id_list[0].size(); i++){
  //     std::string key = input_tra_id_list[0][i];
  //     std::string newKey;
  //     std::istringstream ss(key);
  //     std::string read_number;
  //     std::vector<std::string> multiDimKey;
  //     int j = 0;

  //     while(std::getline(ss, read_number, ',')){
  //       if(j != 1){
  //         multiDimKey.push_back(read_number);
  //       }
  //     }
  //     for(int k = 0; k < multiDimKey.size(); k++){
  //       newKey += multiDimKey[k];
  //       if(k != multiDimKey.size() - 1){
  //         newKey += ",";
  //       }
  //     }
  //     result_map.insert(std::pair(output_tid_list[i], newKey));
  //   } 
  // }


  // create_id_table(out, node, db, result_map);
}

void reKey(std::ostream &out, bbts::node_t &node, const std::string &db, int keyFunc(int, int)){
  // update sqlite TENSOR_IDS table

  const char* db_char= db.c_str();
  sqlite3_stmt * stmt;
  sqlite3 *sql_db;
  std::vector< std::vector < std:: string > > result;

  const char* sql = "SELECT TRA_ID FROM TENSOR_IDS";
  // execute_command(sql, db_name, data);
  if(sqlite3_open(db_char, &sql_db) == SQLITE_OK){
    sqlite3_prepare(sql_db, sql,-1, &stmt, NULL); // preparing the statement
    // sqlite3_step(stmt); // executing the statement
    while(sqlite3_column_text(stmt, 0)){
      for(int i = 0; i < 3; i++){
        result[i].push_back(std::string((char*)sqlite3_column_text(stmt, i)));
        out << std::string((char*)sqlite3_column_text(stmt, i));
        sqlite3_step(stmt);
      }
    }
    std::cout << "not entering while loop";
  }
  else{
    std::cout << "Failed to open db\n";
  }
  

  sqlite3_finalize(stmt);
  sqlite3_close(sql_db);
  
 
  
  
  // std::string sql_str = "UPDATE TENSOR_IDS \n";


  
}


/**************************** compile and run commands from TOS API ***********************/
//compile commands
void compile_commands(std::ostream &out, bbts::node_t &node, const std::string &file_path) {

    // kick off a loading message
  std::atomic_bool b; b = false;
  auto t = loading_message(out, "Compiling commands", b);

  // compile the commands and load them
  auto [did_compile, message] = node.compile_commands(file_path);

  // finish the loading message  
  b = true; t.join();

  if(!did_compile) {
    out << bbts::red << "Failed to compile the : \"" << message << "\"\n" << bbts::reset;
  } else {
    out << bbts::green << message << bbts::reset;
  }
}

//run commands
void run_commands(std::ostream &out, bbts::node_t &node) {

  // kick of a loading message
  std::atomic_bool b; b = false;
  auto t = loading_message(out, "Running the commands", b);

  // run all the commands
  auto [did_load, message] = node.run_commands();

  // finish the loading message
  b = true; t.join();

  // did we fail
  if(!did_load) {
    out << bbts::red << "Failed to run commands : \"" << message << "\"\n" << bbts::reset;
  }
  else {
    out << bbts::green << message << bbts::reset;
  }
}


/*****************************  Functions for managing tra cli basic commands ***************************/
void shutdown(std::ostream &out, bbts::node_t &node) {

  // kick of a loading message
  std::atomic_bool b; b = false;
  auto t = loading_message(out, "Shutting down", b);

  // run all the commands
  auto [did_load, message] = node.shutdown_cluster();

  // finish the loading message
  b = true; t.join();

  // did we fail
  if(!did_load) {
    out << bbts::red << "Failed to shutdown : \"" << message << "\"\n" << bbts::reset;
  }
  else {
    out << bbts::green << message << bbts::reset;
  }
}

//clear the tensor operating system
void clear(std::ostream &out, bbts::node_t &node) {

  // kick of a loading message
  std::atomic_bool b; b = false;
  auto t = loading_message(out, "Clearing", b);

  // run all the commands
  auto [did_load, message] = node.clear();

  // finish the loading message
  b = true; t.join();

  // did we fail
  if(!did_load) {
    out << bbts::red << "Failed to clear : \"" << message << "\"\n" << bbts::reset;
  }
  else {
    out << bbts::green << message << bbts::reset;
  }
}

int keyFunc(int x, int y){
  return x+y;
}


// the prompt
void prompt(bbts::node_t &node) {
    

  std::cout << "\n";
  std::cout << "\t\t    \\   /  \\   /  \\   /  \\   /  \\   /  \\   /  \\   /  \\   /  \n";
  std::cout << "\t\t-----///----///----///----///----///----///----///----///-----\n";
  std::cout << "\t\t    /   \\  /   \\  /   \\  /   \\  /   \\  /   \\  /   \\  /   \\  \n";
  std::cout << "\n";
  std::cout << "\t\t\tWelcome to " << bbts::green << "BarbaTOS-TRA-API" << bbts::reset << ", the tensor operating system\n";
  std::cout << "\t\t\t\tVersion : 0.1 - Lupus Rex\n";
  std::cout << "\t\t\t\tEmail : xy38@rice.edu\n";
  std::cout << '\n';


  auto rootMenu = std::make_unique<Menu>("tra_cli");

  // set up load 
  auto loadSubMenu = std::make_unique<Menu>("load");
  

  loadSubMenu->Insert("text_file",[&](std::ostream &out, const std::string &file) {
  
    load_text_file(out, node, file);
  
  },"Load a data text file. Usage : load text_file <file>\n");

  loadSubMenu->Insert("library", [&](std::ostream &out, const std::string &file) {

    load_shared_library(out, node, file);  
  
  },"Load a shared object file. Usage : load library <file>\n");

  loadSubMenu->Insert("binary_file", [&](std::ostream &out, const std::string &file) {

    load_binary_file(out, node, file);  
  
  },"Load a binary file. Usage : load binary_file <file>\n");

  rootMenu->Insert(std::move(loadSubMenu));


  // set up create 


  // createSubMenu->Insert("tensors_for_text_file",[&](std::ostream &out, const std::string &file, const int32_t num_rows, const int32_t num_cols, const int32_t row_split, const int32_t col_split) {

  //   create_tensors_for_text_file(out, node, file, num_rows, num_cols, row_split, col_split);
  
  // },"Create tensors based on the text data file. Usage : create tensors_for_text_file <file> <num_rows> <num_cols> <row_split> <col_split>\n");

  



  // set up generate
  auto generateSubMenu = std::make_unique<Menu>("generate");


  generateSubMenu->Insert("matrix_to_binary", [&](std::ostream &out, const unsigned num_row, const unsigned num_col, const std::string &file) {

    generate_binary_file(out, node, num_row, num_col, file);  
  
  },"Generate matrix with specified rows and cols in binary file. Usage : generate matrix_to_binary <num_row> <num_col> <file>\n");


  generateSubMenu->Insert("tensors",[&](std::ostream &out, const std::string &file, const int32_t row_split, const int32_t col_split, const std::string &tensor_type) {

    create_tensors(out, node, file, row_split, col_split, tensor_type);
  
  },"Create tensors based on the binary data file. Usage : generate tensors <file> <row_split> <col_split> <tensor_type>\n");

  
  generateSubMenu->Insert("sqlite_db",[&](std::ostream &out, const std::string& db) {
    const char * db_name = db.c_str();

    create_db(db_name);
  
  },"Create sqlite table based on provided db name (eg.\"test.db\"). Usage : generate sqlite_db <db_name>\n");


  generateSubMenu->Insert("id_table",[&](std::ostream &out, const std::string &tra_file, const std::string &db) {

    create_id_table(out, node, tra_file, db);
  
  },"Create id table based on the tra text file and tos id file. Usage : generate id_table <tra_file> <db_name>\n");
  


  rootMenu->Insert(std::move(generateSubMenu));



  // set up run
  auto runSubMenu = std::make_unique<Menu>("run");
  

  runSubMenu->Insert("sqlite_command",[&](std::ostream &out, const std::string &db) {
    const char * db_name = db.c_str();

    char* sql = "CREATE TABLE COMPANY("  \
      "ID INT PRIMARY KEY     NOT NULL," \
      "NAME           TEXT    NOT NULL," \
      "AGE            INT     NOT NULL," \
      "ADDRESS        CHAR(50)," \
      "SALARY         REAL );";
    
    execute_command(sql, db_name, 0);

    sql = "INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) "  \
         "VALUES (1, 'Paul', 32, 'California', 20000.00 ); " \
         "INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) "  \
         "VALUES (2, 'Allen', 25, 'Texas', 15000.00 ); "     \
         "INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY)" \
         "VALUES (3, 'Teddy', 23, 'Norway', 20000.00 );" \
         "INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY)" \
         "VALUES (4, 'Mark', 25, 'Rich-Mond ', 65000.00 );";
    execute_command(sql, db_name, 0);
  
  },"Load a data text file. Usage : load text_file <file>\n");

  
  rootMenu->Insert(std::move(runSubMenu));



  // setup the info command
  rootMenu->Insert("info",[&](std::ostream &out, const std::string &what) {

    if(what == "cluster") {
      node.print_cluster_info(out);
    }
    else if(what == "storage") {
      
      auto [success, message] = node.print_storage_info();
      if(!success) {
        out << bbts::red << "[ERROR]\n";
      }
      out << message << '\n';
    }
    
  },"Returns information about the cluster. Usage : info [cluster, storage, tensor] \n ");

  rootMenu->Insert("info",[&](std::ostream &out, const std::string &what, int32_t id) {

    if(what == "tensor") {
      auto [success, message] = node.print_tensor_info(static_cast<bbts::tid_t>(id));
      if(!success) {
        out << bbts::red << "[ERROR]\n";
      }
      out << message << '\n';
    }

  },"Returns information about the cluster. Usage : info [cluster, storage, tensor] [tid] \n ");



  




  /************************************* generate matrix ************************************/
  rootMenu->Insert("generate_matrix_commands",[&](std::ostream &out, const int32_t num_rows, const int32_t num_cols, 
                                         const int32_t row_split,const int32_t col_split, const std::string &file_path) {
    // the functions
    std::vector<bbts::abstract_ud_spec_t> funs;

    // commands
    std::vector<bbts::abstract_command_t> commands;

    std::map<std::tuple<int32_t, int32_t>, bbts::tid_t> index;

    generate_matrix_commands(num_rows, num_cols, row_split, col_split, commands, index);

    funs.push_back(bbts::abstract_ud_spec_t{.id = UNFORM_ID,
                                    .ud_name = "uniform",
                                    .input_types = {},
                                    .output_types = {"dense"}});

    // std::string file_path = "TRA_commands_matrix_generation.sbbts";
    std::ofstream gen(file_path);
    bbts::compile_source_file_t gsf{.function_specs = funs, .commands = commands};
    gsf.write_to_file(gen);
    gen.close();
    load_text_file(out, node, file_path);
  
  },"Generate commands for generating matrix. Usage : generate_matrix_commands <num_rows> <num_cols> <row_split> <col_split> <file_path.sbbts>\n");




  /*************************************   aggregate  **********************************************/
  rootMenu->Insert("aggregate",[&](std::ostream &out, const int32_t num_rows, const int32_t num_cols, 
                                         const int32_t row_split,const int32_t col_split, std::string dimension,bbts::abstract_ud_spec_id_t kernel_func, const std::string &file_path, const std::string &db) {

    //generate commands and put into a sbbts file  ****************************
    // the functions
    std::vector<bbts::abstract_ud_spec_t> funs;

    // commands
    std::vector<bbts::abstract_command_t> commands;

    std::map<std::tuple<int32_t, int32_t>, bbts::tid_t> index;

    // for (auto row_id = 0; row_id < row_split; row_id++) {
    //   for (auto col_id = 0; col_id < col_split; col_id++) {
    //     index[{row_id, col_id}] = current_tid;
    //     current_tid++;
    //   }
    // }

    //REMOVE LATER
    generate_matrix_commands(num_rows, num_cols, row_split, col_split, commands, index);


    std::vector<bbts::tid_t> output_tid_list;
    std::vector<std::vector<bbts::tid_t>> input_tid_list;

    const char * db_name = db.c_str();

    generate_aggregation_commands(out, num_rows, num_cols, row_split, col_split, commands, index, dimension, kernel_func, output_tid_list, input_tid_list);
  
     

    generate_aggregation_tra(out, node, output_tid_list, input_tid_list, db_name, dimension);
    
    // //REMOVE LATER
    funs.push_back(bbts::abstract_ud_spec_t{.id = UNFORM_ID,
                                    .ud_name = "uniform",
                                    .input_types = {},
                                    .output_types = {"dense"}});


    std::vector<std::string> input_types_list(col_split,"dense");
    funs.push_back(bbts::abstract_ud_spec_t{.id = ADD_ID,
                                    .ud_name = "matrix_add",
                                    .input_types = input_types_list,
                                    .output_types = {"dense"}});
    
    // std::string file_path = "TRA_commands_aggregation.sbbts";
    std::ofstream gen(file_path);
    bbts::compile_source_file_t gsf{.function_specs = funs, .commands = commands};
    gsf.write_to_file(gen);
    gen.close();
    load_text_file(out, node, file_path);
    
    //compile commands
    compile_commands(out, node, file_path);
    //run commands
    run_commands(out, node);

    for (bbts::tid_t tid: output_tid_list){

      auto [success, message] = node.print_tensor_info(static_cast<bbts::tid_t>(tid));
      if(!success) {
        out << bbts::red << "[ERROR]\n";
      }
      out << "tid: " << tid << "\n";
      out << message << '\n';
      
    }
  
  },"Generate and run commands for aggregation. Usage : aggregate <num_rows> <num_cols> <row_split> <col_split> <dimension> <kernel_func> <file_path.sbbts> <db_name>\n");

  /*************************************   join  **********************************************/
  rootMenu->Insert("join",[&](std::ostream &out, const int32_t num_rows, const int32_t num_cols, 
                                         const int32_t row_split,const int32_t col_split, std::string joinKeysL, 
                                         std::string joinKeysR,bbts::abstract_ud_spec_id_t kernel_func, const std::string &file_path, const std::string &db) {
    // the functions
    std::vector<bbts::abstract_ud_spec_t> funs;

    // commands
    std::vector<bbts::abstract_command_t> commands;

    std::map<std::tuple<int32_t, int32_t>, bbts::tid_t> index;

    // for (auto row_id = 0; row_id < row_split; row_id++) {
    //   for (auto col_id = 0; col_id < col_split; col_id++) {
    //     index[{row_id, col_id}] = current_tid;
    //     current_tid++;
    //   }
    // }


    //REMOVE LATER
    generate_matrix_commands(num_rows, num_cols, row_split, col_split, commands, index);

    std::vector<bbts::tid_t> output_tid_list;
    std::vector<std::vector<bbts::tid_t>> input_tid_list;

    const char * db_name = db.c_str();

    generate_join_commands(out, node, num_rows, num_cols, row_split, col_split, commands, index, joinKeysL, joinKeysR, kernel_func, output_tid_list, input_tid_list);
    
    generate_join_tra(out, node, output_tid_list, input_tid_list, db_name, joinKeysL, joinKeysR);


    // //REMOVE LATER
    // funs.push_back(bbts::abstract_ud_spec_t{.id = UNFORM_ID,
    //                                 .ud_name = "uniform",
    //                                 .input_types = {},
    //                                 .output_types = {"dense"}});


    // funs.push_back(bbts::abstract_ud_spec_t{.id = MULT_ID,
    //                                 .ud_name = "matrix_mult", 
    //                                 .input_types = {"dense", "dense"},
    //                                 .output_types = {"dense"}}); //TODO: find a way to take user input for ud_name

    // // std::string file_path = "TRA_commands_join.sbbts";
    // std::ofstream gen(file_path);
    // bbts::compile_source_file_t gsf{.function_specs = funs, .commands = commands};
    // gsf.write_to_file(gen);
    // gen.close();
    // load_text_file(out, node, file_path);

    // //compile commands
    // compile_commands(out, node, file_path);
    // //run commands
    // run_commands(out, node);


    // for (bbts::tid_t tid: output_tid_list){
      
    //   auto [success, message] = node.print_tensor_info(static_cast<bbts::tid_t>(tid));
    //   if(!success) {
    //     out << bbts::red << "[ERROR]\n";
    //   }
    //   out << message << '\n';
      
    // }
  
  },"Generate and run commands for join. Usage : join <num_rows> <num_cols> <row_split> <col_split> <joinKeysL> <joinKeysR> <kernel_func> <file_path.sbbts> <db_name>\n");

  /*********************************************** reKey ******************************************/
  rootMenu->Insert("reKey",[&](std::ostream &out, const std::string &db) {

   reKey(out, node, db.c_str(), &keyFunc);
  
  },"Generate and run commands for join. Usage : join <num_rows> <num_cols> <row_split> <col_split> <joinKeysL> <joinKeysR> <kernel_func> <file_path.sbbts>\n");

  /*********************************************** filter ******************************************/


  /*********************************************** transform ******************************************/


  /*********************************************** tile ******************************************/


  /*********************************************** concat ******************************************/

  rootMenu->Insert("clear",[&](std::ostream &out) {

    clear(out, node);

  },"Clears the tensor operating system.\n");

  rootMenu->Insert("drop_table",[&](std::ostream &out, const std::string &db,const std::string &table) {
  const char* db_name = db.c_str();
  std::string  sql = "DROP TABLE " + table;
  const char* sql_char = sql.c_str();
  execute_command(sql_char, db_name, 0);
  
  }, "DROP table in sqlite; Usage drop_table <db_name> <table_name>");

  // init the command line interface
  Cli tra_cli(std::move(rootMenu));

  // global exit action
  tra_cli.ExitAction([&](auto &out) { shutdown(out, node); });

  // start the cli session
  CliFileSession input(tra_cli);
  input.Start();
}



int main(int argc, char **argv) {
  auto config = std::make_shared<bbts::node_config_t>(bbts::node_config_t{.argc=argc, .argv = argv, .num_threads = 8});

  // create the node
  bbts::node_t node(config);

  // init the node
  node.init();

  // sync everything
  node.sync();

  // kick off the prompt
  std::thread t;
  if (node.get_rank() == 0) {
    t = std::thread([&]() { prompt(node); });
  }

  // the node
  node.run();

  // wait for the prompt to finish
  if (node.get_rank() == 0) { t.join();}

  return 0;
}
