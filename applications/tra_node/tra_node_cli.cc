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
static bbts::abstract_ud_spec_id_t current_uid = 0; // udf_id

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


// for string delimiter
std::vector<std::string> split (std::string s, std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find (delimiter, pos_start)) != std::string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}



std::vector<bbts::tid_t> get_all_tid_from_all_nodes(std::ostream &out, bbts::node_t &node){
  
  auto [success, message] = node.print_all_tid_info();
  out << "message: " << message << "\n";
  if(!success) {
    out << bbts::red << "[ERROR]\n";
  }
  std::vector<std::string> all_tid = split(message, "\n");
  all_tid.pop_back();//remove the stoi

  std::vector<bbts::tid_t> tid_int_list;

  for(auto tid: all_tid){
    bbts::tid_t tid_int = std::stoi(tid);
    tid_int_list.push_back(tid_int);
  }

  std::sort(tid_int_list.begin(), tid_int_list.end());
  
  return tid_int_list;
} 



bool load_tensors(std::ostream &out, bbts::node_t &node, const std::string &file_list) {

  // kick off a loading message
  std::atomic_bool b; b = false;
  auto t = loading_message(out, "Loading tensors from a file", b);

  // try to open the file
  std::ifstream in(file_list);

  if(in.fail()) {
    // finish the loading message
    b = true; t.join();

    out << bbts::red << "Failed to load the filelist " << file_list << '\n' << bbts::reset;
    return false;
  }

  std::vector<std::tuple<bbts::tid_t, std::string, std::string>> parsed_file_list;
  std::string line;
  while(std::getline(in, line)){
    
    // split the file list
    auto values = split(line, "|");
    if(values.size() != 3) {

      // finish the loading message
      b = true; t.join();
      out << bbts::red << "The file list format must be <tid>|<format>|<file> \n" << bbts::reset;
      return false;
    }

    // make sure this is actually an integer
    std::string tid_string = values[0].c_str();
    // Check if tid is a non-negative integer
    for (int i = 0; i < tid_string.size(); i++){
      if (!isdigit(tid_string[i])){
        b = true; t.join();
        out << bbts::red << "\nThe tid must be an integer \n" << bbts::reset;
        return false;
      }  
    }
    bbts::tid_t parsed_tid = std::atoi(values[0].c_str());
    

    // turn the values[2] into a full path and make sure it exists

    // get the path of the filelist
    const std::filesystem::path filelist_path = file_list;
    std::string directory = filelist_path.parent_path();
    std::string file_relative_path = values[2];
    std::string concated_path = directory + "/" + file_relative_path;
    if(!std::filesystem::exists(concated_path)){
      b = true; t.join();
      out << bbts::red << "\nCould not find the tensor file: " << concated_path << " \n" << bbts::reset;
      return false;
    } 


    // right now it is hardcoded to add  tensors in front of it

    // store this <tid, type, filepath>
    parsed_file_list.push_back({parsed_tid, values[1], "tensors/" + values[2]});
  }

  auto [did_load, message] = node.load_tensor_list(parsed_file_list);

  // finish the registering message
  b = true; t.join();

  if(!did_load) {
    out << bbts::red << "Failed to load the tensor list : \"" << message << "\"\n" << bbts::reset;
    return false;
  } else {
    out << bbts::green << message << bbts::reset;
    return true;
  }

  return false;
}




bool generate_binary_file(const unsigned row_num, const unsigned col_num, const std::string &file){
  srand(0);
  //create random data
  std::vector<double> data(row_num * col_num);
  for(int i = 0;i < row_num * col_num;i++) data[i] = 0.1;

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
  // assert(arr.word_size == sizeof(long));
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
          list.push_back(data[sub_r * num_rows + sub_c + row_size*c + col_size*num_cols*r]);
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
  bbts::tid_t temp_current_tid = current_tid;//TODO: switch temp_current_tid back to current_tid after finish data loader
  for(std::map<std::tuple<int32_t, int32_t>, std::vector<std::vector<double>>>::iterator iter = tensor_relation.begin(); iter != tensor_relation.end(); ++iter){
      std::tuple key =  iter->first;
      std::string tensor_file = "t" + std::to_string(temp_current_tid);
      std::ofstream tfile;
      tfile.open("tensors/" + tensor_file);
  
      out << "\n( " << std::get<0>(key) << " , " << std::get<1>(key) << " ): ";
      filelist << temp_current_tid << "|" << tensor_type << "|" << tensor_file;
      myfile << temp_current_tid++ << "|" << "2" << "|" << std::get<0>(key) << "|" << std::get<1>(key) << "|" << row_size << "|" << col_size << "|";
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
  filelist.close();

   
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
  bbts::tid_t max_tid;
  for(it = id_map.begin(); it != id_map.end(); it++){
    sql_str += "INSERT INTO TENSOR_IDS (TOS_ID, TRA_ID)\n";
    sql_str += ("VALUES (" + std::to_string(it->first) + ", \'" + it->second + "\'); \n");
    max_tid = it->first > current_tid? it->first : current_tid;
  }
  if(max_tid > current_tid) current_tid = max_tid + 1;

  const char* sql2 = sql_str.c_str();

  out << sql2;
  execute_command(sql2, db_name, 0);

}


void create_kernel_table(std::ostream &out, bbts::node_t &node,const std::string &kernel_func_file, const std::string &db){
  const char* db_name = db.c_str();
  // char* sql0 = "DROP TABLE TENSOR_IDS";
  // execute_command(sql0, db_name, 0);

  char* sql1 = "CREATE TABLE KERNEL_FUNC(" \
              "KERNEL_ID INT PRIMARY KEY NOT NULL," \
              "KERNEL_NAME VARCHAR(100) NOT NULL);";

  execute_command(sql1, db_name, 0);

  std::string readline;

  std::ifstream kernel_file(kernel_func_file);

  if (!kernel_file){
    out << "There was an error opening the kernel file.\n";
  }

  std::string sql_str;

  std::getline(kernel_file, readline);
  std::istringstream ss(readline);
  std::string read_number;
  std::string record;

  while(std::getline(ss, read_number, '|')){
    sql_str += "INSERT INTO KERNEL_FUNC (KERNEL_ID, KERNEL_NAME)\n";
    record = "";
    record += std::to_string(current_uid);
    current_uid++;
    record += ",\'";
    record += read_number;
    record += "\'";
    sql_str += ("VALUES (" + record + "); \n");
    
  }
    
  const char* sql2 = sql_str.c_str();

  out << sql2;
  execute_command(sql2, db_name, 0);


  kernel_file.close();

}


void update_id_table(std::ostream &out, bbts::node_t &node,const std::string &db, std::map<bbts::tid_t, std::string> id_map){
  const char* db_name = db.c_str();
  
  // sql_str += "UPDATE TENSOR_IDS (TOS_ID, TRA_ID)\n";
  std::string sql_str;
  std::map<bbts::tid_t, std::string>::iterator it;

  bbts::tid_t max_tid;
  for(it = id_map.begin(); it != id_map.end(); it++){
    sql_str += "UPDATE TENSOR_IDS SET TRA_ID = " + it->second + " WHERE TOS_ID = " + std::to_string(it->first) + ";\n";
    max_tid = it->first > current_tid? it->first : current_tid;
  }
  if(max_tid > current_tid) current_tid = max_tid + 1;

  const char* sql2 = sql_str.c_str();

  out << sql2;
  execute_command(sql2, db_name, 0);

}

void update_id_table(std::ostream &out, bbts::node_t &node,const std::string &db, std::map<bbts::tid_t, bool> bool_map){
  const char* db_name = db.c_str();
  
  // sql_str += "UPDATE TENSOR_IDS (TOS_ID, TRA_ID)\n";
  std::string sql_str;
  std::map<bbts::tid_t, bool>::iterator it;
  for(it = bool_map.begin(); it != bool_map.end(); it++){
    if(!(it -> second)){
      sql_str += "DELETE FROM TENSOR_IDS WHERE TOS_ID = " + std::to_string(it->first) + ";\n";
    } 
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
void generate_matrix_commands(std::ostream &out, bbts::node_t &node,
                     int32_t num_row, int32_t num_cols, int32_t row_split,
                     int32_t col_spilt,
                     std::vector<bbts::abstract_command_t> &commands,
                     std::map<std::tuple<int32_t, int32_t>, bbts::tid_t> &index,
                     std::vector<bbts::abstract_ud_spec_t> funs,
                     bbts::abstract_ud_spec_id_t kernel_func,
                     const std::string &file_path,
                     const std::string &db) {
  

  std::vector<bbts::command_param_t> param_data = {
      bbts::command_param_t{.u = (std::uint32_t)(num_row / row_split)},
      bbts::command_param_t{.u = (std::uint32_t)(num_cols / col_spilt)},
      bbts::command_param_t{.f = 1.0f}, bbts::command_param_t{.f = 2.0f}};

  for (auto row_id = 0; row_id < row_split; row_id++) {
    for (auto col_id = 0; col_id < col_spilt; col_id++) {

      index[{row_id, col_id}] = current_tid;

      // store the command
      commands.push_back(
          bbts::abstract_command_t{.ud_id = kernel_func,
                             .type = bbts::abstract_command_type_t::APPLY,
                             .input_tids = {},
                             .output_tids = {current_tid++},
                             .params = param_data});
      
    }
  }

  funs.push_back(bbts::abstract_ud_spec_t{.id = kernel_func,
                                    .ud_name = "uniform",
                                    .input_types = {},
                                    .output_types = {"dense"}});

  // std::string file_path = "TRA_commands_matrix_generation.sbbts";
  std::ofstream gen(file_path);
  bbts::compile_source_file_t gsf{.function_specs = funs, .commands = commands};
  gsf.write_to_file(gen);
  gen.close();
  load_text_file(out, node, file_path);
}

//TRA operations: AGGREGATION//
// Generate commands for aggregation
void generate_aggregation_commands(std::ostream &out, bbts::node_t &node, int32_t num_rows, int32_t num_cols, int32_t row_split,
                          int32_t col_split,
                          std::vector<bbts::abstract_command_t> &commands,
                          std::map<std::tuple<int32_t, int32_t>, bbts::tid_t> &index,
                          std::vector<bbts::abstract_ud_spec_t> funs,
                          std::string dimension1,
                          std::string dimension2,
                          bbts::abstract_ud_spec_id_t kernel_func, 
                          std::vector<bbts::tid_t> &output_tids_list,
                          std::vector<std::vector<bbts::tid_t>> &input_tids_list,
                          const std::string &file_path,
                          const std::string &db) { //Find a way to pass dynamic library
  

  // generate_matrix_commands(num_rows, num_cols, row_split, col_split, commands, index); 
  //TODO: find a way to keep up current_tid without calling generating matrix function inside aggregate

  std::vector<bbts::command_param_t> param_data = {};
  std::vector<bbts::tid_t> input_tids_sublist;

  
  
  // get info from id_table
  const char* db_char= db.c_str();
  sqlite3_stmt * stmt;
  sqlite3 *sql_db;
  bbts::tid_t max_tid = 0;

  std::map<std::vector<std::string>, bbts::tid_t> id_map;
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
      bbts::tid_t tid = sqlite3_column_int(stmt, 0);
      max_tid = tid > max_tid? tid : max_tid;
      std::string tra_id = std::string(reinterpret_cast< const char* >(sqlite3_column_text(stmt, 1)));
      id_map.insert(std::pair<std::vector<std::string>, bbts::tid_t>(split(tra_id, ","), tid));
      out << "tid: " << tid << "\n";
      out << "tra_id: " << tra_id << "\n";
      // out << "map key: (" << split(tra_id, ",")[0] << " , " << split(tra_id, ",")[1] << ")" << " value: " << tid << "\n";
    }
  }
  else{
    std::cout << "Failed to open db\n";
  }

  sqlite3_close(sql_db);

  std::map<std::vector<std::string>, bbts::tid_t>::iterator it;
  for(it = id_map.begin(); it != id_map.end(); it++){
    max_tid = it->second > max_tid ? it->second : max_tid;
  }

  if(max_tid > current_tid) current_tid = max_tid + 1;

  if(dimension2.compare("-1") == 0){
    int i = 0;
    for (auto row_id = 0; row_id < row_split; row_id++) {
      for (auto col_id = 0; col_id < col_split; col_id++) {
        std::vector<std::string> key = {std::to_string(row_id), std::to_string(col_id)};
        index[{row_id, col_id}] = id_map.find(key)->second;
        i++;
      }
    }

    
    

    if(dimension1.compare("") == 0) {
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
    else if(dimension1.compare("0") == 0){
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
    else if(dimension1.compare("1") == 0){
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
  else{
    //assume dimension1 = 0, dimension2 = 2
    std::map<std::vector<int32_t>, bbts::tid_t> three_dim_index;
    for(auto dim0 = 0; dim0 < row_split; dim0++){
      for(auto dim1 = 0; dim1 < row_split; dim1++){
        for(auto dim2 = 0; dim2 < row_split; dim2++){
          std::vector<std::string> key = {std::to_string(dim0), std::to_string(dim1), std::to_string(dim2)};
          three_dim_index[{dim0, dim1, dim2}] = id_map.find(key)->second;
        }
      }
    }

    for(auto dim0 = 0; dim0 < row_split; dim0++){
      for(auto dim2 = 0; dim2 < row_split; dim2++){
        for(auto dim1 = 0; dim1 < row_split; dim1++){
          input_tids_sublist.push_back(three_dim_index[{dim0, dim1, dim2}]);
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

  

  // get ud_name from kernel_table
  sqlite3_stmt * stmt2;
  sqlite3 *sql_db2;
  std::string ud_name;
  std::string sql2_pre = "SELECT KERNEL_NAME FROM KERNEL_FUNC WHERE KERNEL_ID = " + std::to_string(kernel_func) + ";\n";
  const char* sql2 = sql2_pre.c_str();
  
  if(sqlite3_open(db_char, &sql_db2) == SQLITE_OK){
     // preparing the statement
    int rc = sqlite3_prepare_v2(sql_db2, sql2,-1, &stmt2, NULL);

    if(rc != SQLITE_OK){
      out << "ERRORS PREPARING THE STATEMENT";
      return;
    }
    // sqlite3_step(stmt); // executing the statement
    while((rc = sqlite3_step(stmt2)) == SQLITE_ROW){
      ud_name = std::string(reinterpret_cast< const char* >(sqlite3_column_text(stmt2, 0)));
    }
  }

  else{
    std::cout << "Failed to open db\n";
  }

  sqlite3_close(sql_db2);
  out << "ud_name: " << ud_name << "\n";

  std::vector<std::string> input_types_list(col_split,"dense");
  funs.push_back(bbts::abstract_ud_spec_t{.id = kernel_func,
                                  .ud_name = ud_name,
                                  .input_types = input_types_list,
                                  .output_types = {"dense"}});
  
  // std::string file_path = "TRA_commands_aggregation.sbbts";
  std::ofstream gen(file_path);
  bbts::compile_source_file_t gsf{.function_specs = funs, .commands = commands};
  gsf.write_to_file(gen);
  gen.close();
  load_text_file(out, node, file_path);
}


void generate_aggregation_tra(std::ostream &out, bbts::node_t &node, std::vector<bbts::tid_t> output_tid_list, std::vector<std::vector<bbts::tid_t>> input_tid_list, const std::string &db, std::string dimension1, std::string dimension2){

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
    }
    input_tra_id_list.push_back(input_tra_id_sublist);
  }


  
  //if there is only one dimension
  if(dimension2.compare("-1") == 0){
    if(dimension1.compare("") == 0){
      std::string newKey = "";
      
      result_map.insert(std::pair(output_tid_list[0], newKey));
    }
    else if(dimension1.compare("0") == 0){
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
        result_map.insert(std::pair(output_tid_list[i], newKey));
      } 
    }
    else if(dimension1.compare("1") == 0){

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
        result_map.insert(std::pair(output_tid_list[i], newKey));
      } 
    }
  }
  else{
    //assume only dimension1 =0, dimension2 = 2
    for (int i = 0; i < input_tra_id_list.size(); i++){
      std::string key = input_tra_id_list[i][0];
      std::istringstream ss(key);
      std::string newKey = "";
      std::string read_number;
      
      int j = 0;
      


      while(std::getline(ss, read_number, ',')){
        if(j == 0 || j == 2){
          newKey += read_number;
        }
        j++;
      }
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
                            std::vector<bbts::abstract_ud_spec_t> funs,
                            std::string joinKeysL,
                            std::string joinKeysR,
                            bbts::abstract_ud_spec_id_t kernel_func, 
                            std::vector<bbts::tid_t> &output_tids_list,
                            std::vector<std::vector<bbts::tid_t>> &input_tids_list,
                            const std::string &file_path,
                            const std::string &db) { //Find a way to pass dynamic library
  
  std::vector<bbts::command_param_t> param_data = {};
  
  // get info from id_table
  const char* db_char= db.c_str();
  sqlite3_stmt * stmt;
  sqlite3 *sql_db;
  bbts::tid_t max_tid;

  std::map<std::vector<std::string>, bbts::tid_t> id_map;
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
      bbts::tid_t tid = sqlite3_column_int(stmt, 0);
      max_tid = tid > max_tid? tid : max_tid;
      std::string tra_id = std::string(reinterpret_cast< const char* >(sqlite3_column_text(stmt, 1)));
      id_map.insert(std::pair<std::vector<std::string>, bbts::tid_t>(split(tra_id, ","), tid));
    }
  }
  else{
    std::cout << "Failed to open db\n";
  }

  int i = 0;
  for (auto row_id = 0; row_id < row_split; row_id++) {
    for (auto col_id = 0; col_id < col_split; col_id++) {
      std::vector<std::string> key = {std::to_string(row_id), std::to_string(col_id)};
      index[{row_id, col_id}] = id_map.find(key)->second;
      i++;
    }
  }
  //TODO: how to keep up current_tid
  if(max_tid > current_tid) current_tid = max_tid + 1;

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


  // get ud_name from id_table
  sqlite3_stmt * stmt2;
  sqlite3 *sql_db2;
  std::string ud_name = "";
  std::string sql2_pre = "SELECT KERNEL_NAME FROM KERNEL_FUNC WHERE KERNEL_ID = " + std::to_string(kernel_func) + ";\n";
  const char* sql2 = sql2_pre.c_str();

  if(sqlite3_open(db_char, &sql_db2) == SQLITE_OK){
     // preparing the statement
    int rc = sqlite3_prepare_v2(sql_db, sql2,-1, &stmt2, NULL);

    if(rc != SQLITE_OK){
      out << "ERRORS PREPARING THE STATEMENT";
      return;
    }
    // sqlite3_step(stmt); // executing the statement
    while((rc = sqlite3_step(stmt2)) == SQLITE_ROW){
      ud_name = std::string(reinterpret_cast< const char* >(sqlite3_column_text(stmt2, 0)));
    }
  }

  else{
    std::cout << "Failed to open db\n";
  }

  out << "ud_name: " << ud_name << "\n";
  
  funs.push_back(bbts::abstract_ud_spec_t{.id = kernel_func,
                                  .ud_name = ud_name, 
                                  .input_types = {"dense", "dense"},
                                  .output_types = {"dense"}}); 

  // std::string file_path = "TRA_commands_join.sbbts";
  std::ofstream gen(file_path);
  bbts::compile_source_file_t gsf{.function_specs = funs, .commands = commands};
  gsf.write_to_file(gen);
  gen.close();
  load_text_file(out, node, file_path);
  
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

  for (std::vector<bbts::tid_t> input_tid_sublist : input_tid_list){
    std::vector<std::string> input_tra_id_sublist;
    for(bbts::tid_t tid: input_tid_sublist){
      std::string tra_id = id_map.find(tid) -> second;
      input_tra_id_sublist.push_back(tra_id);
    }
    input_tra_id_list.push_back(input_tra_id_sublist);
  }

  for(int i = 0; i < input_tra_id_list.size(); i++){
    std::string keyL = input_tra_id_list[i][0];
    std::string keyR = input_tra_id_list[i][1];


    std::string joinKey;
    std::istringstream ssL(keyL);
    std::istringstream ssR(keyR);
    std::string read_number_L;
    std::string read_number_R;
    
    std::vector<std::string> temp_R;

    while(std::getline(ssL, read_number_L, ',')){
      joinKey += (read_number_L + ",");
    }

    int j = 0;
    while(std::getline(ssR, read_number_R, ',')){
      if(j != stoi(joinKeysR)){
        temp_R.push_back(read_number_R);
      }
      j++;
    }

    for(int i = 0; i < temp_R.size(); i++){
      joinKey += temp_R[i];
      if(i != temp_R.size() - 1){
         joinKey+= ",";
      }
    }

    result_map.insert(std::pair(output_tid_list[i], joinKey));
    
  }
  create_id_table(out, node, db, result_map);
  

}



void reKey(std::ostream &out, bbts::node_t &node, const std::string &db, int keyFunc(std::vector<int>)){

  const char* db_char= db.c_str();
  sqlite3_stmt * stmt;
  sqlite3 *sql_db;
  std::vector<std::vector<std:: string>> result;
  std::map<bbts::tid_t, std::string> id_map;

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
      out << "tos_id: " << std::to_string(tos_id) << " tra_id: " << tra_id << "\n";
      id_map.insert(std::pair(tos_id, tra_id));
    }
  }
  else{
    std::cout << "Failed to open db\n";
  }

  //change key based on keyFunc
  std::map<bbts::tid_t, std::string>::iterator it;
  for(it = id_map.begin(); it != id_map.end(); it++){
    bbts::tid_t tos_id = it -> first;
    std::string old_tra_id = it -> second;

    std::vector<int> tra_id_vec;
    std::istringstream ss(old_tra_id);
    std::string read_number;
    
    while(std::getline(ss, read_number, ',')){
      tra_id_vec.push_back(stoi(read_number));
    }
    
    std::string new_tra_id = std::to_string(keyFunc(tra_id_vec));
    it -> second = new_tra_id;
  }  

  sqlite3_finalize(stmt);
  sqlite3_close(sql_db);

  update_id_table(out, node, db, id_map);
 
}




void filter(std::ostream &out, bbts::node_t &node, const std::string &db, bool boolFunc(std::vector<int>)){

  const char* db_char= db.c_str();
  sqlite3_stmt * stmt;
  sqlite3 *sql_db;
  std::vector<std::vector<std:: string>> result;
  std::map<bbts::tid_t, std::string> id_map;
  std::map<bbts::tid_t, bool> bool_map;

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

  //change key based on keyFunc
  std::map<bbts::tid_t, std::string>::iterator it;
  for(it = id_map.begin(); it != id_map.end(); it++){
    bbts::tid_t tos_id = it -> first;
    std::string old_tra_id = it -> second;

    std::vector<int> tra_id_vec;
    std::istringstream ss(old_tra_id);
    std::string read_number;
    
    while(std::getline(ss, read_number, ',')){
      tra_id_vec.push_back(stoi(read_number));
    }
    
    bool tra_filter = boolFunc(tra_id_vec);
    bool_map.insert(std::pair(tos_id, tra_filter));
  }  

  sqlite3_finalize(stmt);
  sqlite3_close(sql_db);

  update_id_table(out, node, db, bool_map);
 
}

void generate_transform_commands(std::ostream &out, bbts::node_t &node,
                     int32_t num_row, int32_t num_cols, int32_t row_split,
                     int32_t col_spilt,
                     std::vector<bbts::abstract_command_t> &commands,
                     std::map<std::tuple<int32_t, int32_t>, bbts::tid_t> &index,
                     std::vector<bbts::abstract_ud_spec_t> funs,
                     bbts::abstract_ud_spec_id_t kernel_func,
                     const std::string &file_path,
                     const std::string &db) {
  

  std::vector<bbts::command_param_t> param_data = {
      bbts::command_param_t{.u = (std::uint32_t)(num_row / row_split)},
      bbts::command_param_t{.u = (std::uint32_t)(num_cols / col_spilt)},
      bbts::command_param_t{.f = 1.0f}, bbts::command_param_t{.f = 2.0f}};

  for (auto row_id = 0; row_id < row_split; row_id++) {
    for (auto col_id = 0; col_id < col_spilt; col_id++) {

      index[{row_id, col_id}] = current_tid;

      // store the command
      commands.push_back(
          bbts::abstract_command_t{.ud_id = kernel_func,
                             .type = bbts::abstract_command_type_t::APPLY,
                             .input_tids = {}, //TODO: get input_tid from info storage
                             .output_tids = {current_tid++},
                             .params = param_data});
      
    }
  }

  funs.push_back(bbts::abstract_ud_spec_t{.id = kernel_func,
                                    .ud_name = "uniform",//TODO: get kernel_func
                                    .input_types = {"dense"},
                                    .output_types = {"dense"}});

  // std::string file_path = "TRA_commands_matrix_generation.sbbts";
  std::ofstream gen(file_path);
  bbts::compile_source_file_t gsf{.function_specs = funs, .commands = commands};
  gsf.write_to_file(gen);
  gen.close();
  load_text_file(out, node, file_path);
}

void transform(std::ostream &out, bbts::node_t &node, const std::string &db, int32_t row_size, int32_t col_size, const std::string &tensor_type, bbts::tid_t transform_tid, std::string transformFunc(int32_t, int32_t, std::string)){
  //take a text file of data from transformFunc
  std::string file_name = "transform.npy";
  std::string file = transformFunc(row_size, col_size, file_name);
  cnpy::NpyArray arr = load_binary_file(out, node, file_name);
  double* data = arr.data<double>();


  //update filelist.txt and other file
  std::ofstream filelist;
  std::string tensor_file = "t" + std::to_string(current_tid);
  std::ofstream tfile;
  tfile.open("tensors/" + tensor_file);

  filelist.open("tensors/filelist.txt");
  filelist << current_tid << "|" << tensor_type << "|" << tensor_file;
  tfile << row_size << "|" << col_size << "|";

  //load_tensors
  load_tensors(out, node, "tensors/filelist.txt"); 

  //select old tid to get tra_id
  const char* db_char= db.c_str();
  sqlite3_stmt * stmt;
  sqlite3 *sql_db;
  bbts::tid_t old_tid = 0;
  std::string old_tra_id = "";
  bbts::tid_t max_tid = 0;
  // std::string sql_pre = "SELECT * FROM TENSOR_IDS WHERE TOS_ID = " + std::to_string(transform_tid) + ";\n";
  std::string sql_pre = "SELECT * FROM TENSOR_IDS;\n";
  const char* sql = sql_pre.c_str();

  if(sqlite3_open(db_char, &sql_db) == SQLITE_OK){
     // preparing the statement
    int rc = sqlite3_prepare_v2(sql_db, sql,-1, &stmt, NULL);

    if(rc != SQLITE_OK){
      out << "ERRORS PREPARING THE STATEMENT";
      return;
    }
    // sqlite3_step(stmt); // executing the statement
    while((rc = sqlite3_step(stmt)) == SQLITE_ROW){
      old_tid = sqlite3_column_int(stmt, 0);
      max_tid = old_tid > max_tid? old_tid : max_tid;
      if(old_tid == transform_tid){
        old_tra_id = std::string(reinterpret_cast< const char* >(sqlite3_column_text(stmt, 1)));
      }
      
    }
  }

  else{
    std::cout << "Failed to open db\n";
  }

  if(max_tid > current_tid) current_tid = max_tid + 1;

  //delete old tuple and insert (new tid, tra_id) into sqlite table
  std::string sql_str = "DELETE FROM TENSOR_IDS WHERE TOS_ID = " + std::to_string(transform_tid) + ";\n";
  sql_str += "INSERT INTO TENSOR_IDS (TOS_ID, TRA_ID)\n";
  sql_str += ("VALUES (\'" + std::to_string(current_tid) + "\', \'" + old_tra_id + "\'); \n");

  const char* sql2 = sql_str.c_str();
  out << sql2;

  execute_command(sql2, db_char, 0);
  current_tid++;
}











/**************************** compile and run commands from TOS API ***********************/

//load commands

void load_binary_command(std::ostream &out, bbts::node_t &node, const std::string &file_path) {

  // kick off a loading message
  std::atomic_bool b; b = false;
  auto t = loading_message(out, "Loading the file", b);

  // try to deserialize
  bbts::parsed_command_list_t cmd_list;
  bool success = cmd_list.deserialize(file_path);

  // finish the loading message
  b = true; t.join();

  // did we fail
  if(!success) {
    out << bbts::red << "Failed to load the file " << file_path << '\n' << bbts::reset;
    return;
  }

  // kick off a loading message
  b = false;
  t = loading_message(out, "Scheduling the loaded commands", b);

  // load the commands we just parsed
  auto [did_load, message] = node.load_commands(cmd_list);

  // finish the loading message
  b = true; t.join();

  // did we fail
  if(!did_load) {
    out << bbts::red << "Failed to schedule the loaded commands : \"" << message << "\"\n" << bbts::reset;
  }
  else {
    out << bbts::green << message << bbts::reset;
  }
}


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

void materialize_aggregation_commands(std::ostream &out, bbts::node_t &node, const std::string &file_path, 
                                      std::vector<bbts::tid_t> output_tid_list, std::vector<std::vector<bbts::tid_t>> input_tid_list,
                                      const std::string &db, std::string dimension1, std::string dimension2,
                                      bool store_result){

  //load binary commands
  // load_binary_command(out, node, file_path);
  //compile commands
  compile_commands(out, node, file_path);
  //run commands
  run_commands(out, node);

  if(store_result){
    generate_aggregation_tra(out, node, output_tid_list, input_tid_list, db, dimension1, dimension2);
  }
  
}


void materialize_join_commands(std::ostream &out, bbts::node_t &node, const std::string &file_path, 
                                      std::vector<bbts::tid_t> output_tid_list, std::vector<std::vector<bbts::tid_t>> input_tid_list,
                                      const std::string &db, std::string joinKeysL, std::string joinKeysR,
                                      bool store_result){

  //load binary commands
  // load_binary_command(out, node, file_path);
  //compile commands
  compile_commands(out, node, file_path);
  //run commands
  run_commands(out, node);

  if(store_result){
    generate_join_tra(out, node, output_tid_list, input_tid_list, db, joinKeysL, joinKeysR);
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

int keyFunc(std::vector<int> int_list){
  int res = 0;
  for(int i : int_list){
    res += i;
  }
  return res;
}

bool boolFunc(std::vector<int> int_list){
  int res = 0;
  for(int i : int_list){
    res += i;
  }
  if(res == 0) return false;
  
  return true;;
}

std::string transformFunc(int32_t row_size, int32_t col_size, std::string file_name){
  std::vector<double> data(row_size*col_size);
  for(int i = 0;i < row_size * col_size;i++) data[i] = 0.2;

  const unsigned row_num = row_size;
  const unsigned col_num = col_size;
  //save it to file
  cnpy::npy_save(file_name,&data[0],{col_num, row_num},"w");
  return file_name;
}

void read_table(std::ostream &out, const std::string &db,const std::string &table){
  const char* db_name = db.c_str();
  std::string  sql = "SELECT * FROM " + table + ";\n";
  const char* sql_char = sql.c_str();
  
  sqlite3_stmt * stmt;
  sqlite3 *sql_db;

  if(sqlite3_open(db_name, &sql_db) == SQLITE_OK){
     // preparing the statement
    int rc = sqlite3_prepare_v2(sql_db, sql_char,-1, &stmt, NULL);

    if(rc != SQLITE_OK){
      out << "ERRORS PREPARING THE STATEMENT";
      return;
    }
    // sqlite3_step(stmt); // executing the statement
    while((rc = sqlite3_step(stmt)) == SQLITE_ROW){
      bbts::tid_t id = sqlite3_column_int(stmt, 0);
      std::string string_name = std::string(reinterpret_cast< const char* >(sqlite3_column_text(stmt, 1)));
      if(table.compare("TENSOR_IDS") == 0){
        out << "tos_id: " << id << ", tra_id: " << string_name << "\n";
      }
      if(table.compare("KERNEL_FUNC") == 0){
        out << "kernel_id: " << id << ", kernel_name: " << string_name << "\n";
      }
    }
  }
  else{
    std::cout << "Failed to open db\n";
  }

  sqlite3_close(sql_db);
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
    out << "\n\n\n\n";
  
  },"Load a data text file. Usage : load text_file <file>\n");

  loadSubMenu->Insert("library", [&](std::ostream &out, const std::string &shared_lib_file, const std::string &kernel_func_file, const std::string &db) {

    load_shared_library(out, node, shared_lib_file);  
    out << "Successfully load shared library\n";

    create_kernel_table(out, node, kernel_func_file, db);
    read_table(out, db, "KERNEL_FUNC");
    out << "\n\n\n\n";
  
  },"Load a shared object file with a file holding all kernel functions. Usage : load library <file> <kernel_func_file> <db_name>\n");

  loadSubMenu->Insert("binary_file", [&](std::ostream &out, const std::string &file) {

    load_binary_file(out, node, file);  
    out << "\n\n\n\n";
  
  },"Load a binary file. Usage : load binary_file <file>\n");

  loadSubMenu->Insert("tensors",[&](std::ostream &out, const std::string &file_list) {

    load_tensors(out, node, file_list);
    out << "\n\n\n\n";

  },"Load tensors from filelist. Usage : load tensors <path_to_file_list>\n");


  rootMenu->Insert(std::move(loadSubMenu));


  // set up create 


  // createSubMenu->Insert("tensors_for_text_file",[&](std::ostream &out, const std::string &file, const int32_t num_rows, const int32_t num_cols, const int32_t row_split, const int32_t col_split) {

  //   create_tensors_for_text_file(out, node, file, num_rows, num_cols, row_split, col_split);
  
  // },"Create tensors based on the text data file. Usage : create tensors_for_text_file <file> <num_rows> <num_cols> <row_split> <col_split>\n");

  



  // set up generate
  auto generateSubMenu = std::make_unique<Menu>("generate");


  generateSubMenu->Insert("matrix_to_binary", [&](std::ostream &out, const unsigned num_row, const unsigned num_col, const std::string &file) {

    generate_binary_file(num_row, num_col, file);  
    load_binary_file(out, node, file);
    out << "\n\n\n\n";
  
  },"Generate matrix with specified rows and cols in binary file. Usage : generate matrix_to_binary <num_row> <num_col> <file>\n");


  generateSubMenu->Insert("tensors",[&](std::ostream &out, const std::string &file, const int32_t row_split, const int32_t col_split, const std::string &tensor_type, const std::string &db) {

    create_tensors(out, node, file, row_split, col_split, tensor_type);
    create_id_table(out, node, "tensor_relation.txt", db);
    out << "\n\n\n\n";
  
  },"(1) Create tensors based on the binary data file. (2)Generate data file format for data loader (3) Save relation into sqlite. Usage : generate tensors <file> <row_split> <col_split> <tensor_type> <db_name>\n");

  
  generateSubMenu->Insert("sqlite_db",[&](std::ostream &out, const std::string& db) {
    const char * db_name = db.c_str();

    create_db(db_name);
    out << "\n\n\n\n";
  
  },"Create sqlite table based on provided db name (eg.\"test.db\"). Usage : generate sqlite_db <db_name>\n");


  generateSubMenu->Insert("id_table",[&](std::ostream &out, const std::string &tra_file, const std::string &db) {

    create_id_table(out, node, tra_file, db);
    out << "\n\n\n\n";
  
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
    else if(what == "all_tid"){
      // auto [success, message] = node.print_all_tid_info();

      // if(!success) {
      //   out << bbts::red << "[ERROR]\n";
      // }
      // out << message << '\n';
      get_all_tid_from_all_nodes(out, node);
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
                                         const int32_t row_split,const int32_t col_split,  bbts::abstract_ud_spec_id_t kernel_func, const std::string &file_path, const std::string &db) {
    // the functions
    std::vector<bbts::abstract_ud_spec_t> funs;

    // commands
    std::vector<bbts::abstract_command_t> commands;

    std::map<std::tuple<int32_t, int32_t>, bbts::tid_t> index;

    generate_matrix_commands(out, node, num_rows, num_cols, row_split, col_split, commands, index,funs, kernel_func, file_path, db);
    //load binary commands
    // load_binary_command(out, node, file_path);
    //compile commands
    compile_commands(out, node, file_path);
    //run commands
    run_commands(out, node);
    out << "\n\n\n\n";
    
  
  },"Generate commands for generating matrix. Usage : generate_matrix_commands <num_rows> <num_cols> <row_split> <col_split> <kernel_func> <file_path.sbbts> <db_name>\n");




  /*************************************   aggregate  **********************************************/
  rootMenu->Insert("aggregate",[&](std::ostream &out, const int32_t num_rows, const int32_t num_cols, 
                                  const int32_t row_split,const int32_t col_split, std::string dimension1, 
                                  std::string dimension2,bbts::abstract_ud_spec_id_t kernel_func, 
                                  const std::string &file_path, const std::string &db, bool store_result) {
    


    //generate commands and put into a sbbts file  ****************************
    // the functions
    std::vector<bbts::abstract_ud_spec_t> funs;

    // commands
    std::vector<bbts::abstract_command_t> commands;

    std::map<std::tuple<int32_t, int32_t>, bbts::tid_t> index;

    

    //REMOVE LATER
    // generate_matrix_commands(num_rows, num_cols, row_split, col_split, commands, index);


    std::vector<bbts::tid_t> output_tid_list;
    std::vector<std::vector<bbts::tid_t>> input_tid_list;
    

    generate_aggregation_commands(out, node, num_rows, num_cols, row_split, col_split, commands, index, funs, dimension1, dimension2, kernel_func, output_tid_list, input_tid_list, file_path, db);


    materialize_aggregation_commands(out, node, file_path, output_tid_list, input_tid_list, db, dimension1, dimension2, store_result);


    

    for (bbts::tid_t tid: output_tid_list){

      auto [success, message] = node.print_tensor_info(static_cast<bbts::tid_t>(tid));
      if(!success) {
        out << bbts::red << "[ERROR]\n";
      }
      out << "tid: " << tid << "\n";
      out << message << '\n';
      
    }
    out << "\n\n\n\n";
  
  },"Generate and run commands for aggregation. Usage : aggregate <num_rows> <num_cols> <row_split> <col_split> <dimension1> <dimension2> <kernel_func> <file_path.sbbts> <db_name> <store_commands>\n");

  /*************************************   join  **********************************************/
  rootMenu->Insert("join",[&](std::ostream &out, const int32_t num_rows, const int32_t num_cols, 
                                         const int32_t row_split,const int32_t col_split, std::string joinKeysL, 
                                         std::string joinKeysR,bbts::abstract_ud_spec_id_t kernel_func, 
                                         const std::string &file_path, const std::string &db, bool store_result) {
    // the functions
    std::vector<bbts::abstract_ud_spec_t> funs;

    // commands
    std::vector<bbts::abstract_command_t> commands;

    std::map<std::tuple<int32_t, int32_t>, bbts::tid_t> index;

    for (auto row_id = 0; row_id < row_split; row_id++) {
      for (auto col_id = 0; col_id < col_split; col_id++) {
        index[{row_id, col_id}] = current_tid;
        current_tid++;
      }
    }



    std::vector<bbts::tid_t> output_tid_list;
    std::vector<std::vector<bbts::tid_t>> input_tid_list;


    generate_join_commands(out, node, num_rows, num_cols, row_split, col_split, commands, index, funs, joinKeysL, joinKeysR, kernel_func, output_tid_list, input_tid_list, file_path, db);


    materialize_join_commands(out, node, file_path, output_tid_list, input_tid_list, db, joinKeysL, joinKeysR, store_result);




    for (bbts::tid_t tid: output_tid_list){
      
      auto [success, message] = node.print_tensor_info(static_cast<bbts::tid_t>(tid));
      if(!success) {
        out << bbts::red << "[ERROR]\n";
      }
      out << message << '\n';
      
    }
    out << "\n\n\n\n";
  
  },"Generate and run commands for join. Usage : join <num_rows> <num_cols> <row_split> <col_split> <joinKeysL> <joinKeysR> <kernel_func> <file_path.sbbts> <db_name> <store_commands>\n");

  /*********************************************** reKey ******************************************/
  rootMenu->Insert("rekey",[&](std::ostream &out, const std::string &db) {

   reKey(out, node, db.c_str(), &keyFunc);
   read_table(out, db, "TENSOR_IDS");
   out << "\n\n\n\n";
  
  },"Perform reKey. Usage : rekey <db_name>\n");

  /*********************************************** filter ******************************************/

  rootMenu->Insert("filter",[&](std::ostream &out, const std::string &db) {

   filter(out, node, db.c_str(), &boolFunc);
   read_table(out, db, "TENSOR_IDS");
   out << "\n\n\n\n";
  
  },"Perform filter. Usage : filter <db_name>\n");


  /*********************************************** transform ******************************************/
  rootMenu->Insert("transform",[&](std::ostream &out, const std::string &db, const int32_t row_size, const int32_t col_size, const std::string &tensor_type, bbts::tid_t transform_tid) {

   transform(out, node, db, row_size, col_size, tensor_type, transform_tid, &transformFunc);
   read_table(out, db, "TENSOR_IDS");
   out << "\n\n\n\n";
  
  },"Perform transform. Usage : transform <db_name> <row_size> <col_size> <tensor_type> <tid>\n");

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
  current_tid = 0;
  out << "\n\n\n\n";
  
  }, "DROP table in sqlite; Usage drop_table <db_name> <table_name>");

  rootMenu->Insert("read_table",[&](std::ostream &out, const std::string &db,const std::string &table) {

  read_table(out, db, table);
  
  }, "Read table in sqlite; Usage read_table <db_name> <table_name>");

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
