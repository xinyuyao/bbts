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
#include <filesystem>




using namespace cli;


static bbts::tid_t current_tid = 0; // tensor_id

const int32_t UNFORM_ID = 0;
const int32_t ADD_ID = 1;
const int32_t MULT_ID = 2;

// to store the relation before materialize
std::map<std::string, std::map<std::string, bbts::tid_t>> stored_R;

std::unordered_set<std::string> math_op = {"add", "subtract", "multiply", "divide", "mod", "+", "-", "*", "/", "%"};

std::unordered_set<std::string> collective_op = {"sum"};

// bbts::udf_manager_ptr udf_manager;

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
  // std::atomic_bool b; b = false;
  // auto t = loading_message(out, "Loading the data text file", b);

  // try to open the file
  std::ifstream in(file_path, std::ifstream::ate);

  if(in.fail()) {
    // finish the loading message
    // b = true; t.join();

    std::cout << bbts::red << "Failed to load the file " << file_path << '\n' << bbts::reset;
    return new char[0];
  }

  auto file_len = (size_t) in.tellg();
  in.seekg (0, std::ifstream::beg);

  auto file_bytes = new char[file_len];
  in.readsome(file_bytes, file_len);

  // std::cout << "\nFile content:\n" << file_bytes << '\n';
  // std::cout << "\nFile length is:" << file_len << '\n';

  // finish the loading message  
  // b = true; t.join();

  return file_bytes;

}



bool load_shared_library(std::ostream &out, bbts::node_t &node, const std::string &file_path) {

  // kick off a loading message
  // std::atomic_bool b; b = false;
  // auto t = loading_message(out, "Loading the library file", b);

  // try to open the file
  std::ifstream in(file_path, std::ifstream::ate | std::ifstream::binary);


  if(in.fail()) {
    // finish the loading message
    // b = true; t.join();

    std::cout << bbts::red << "Failed to load the file " << file_path << '\n' << bbts::reset;
    return false;
  }


  auto file_len = (size_t) in.tellg();
  in.seekg (0, std::ifstream::beg);


  auto file_bytes = new char[file_len];
  in.readsome(file_bytes, file_len);


  // finish the loading message  
  // b = true; t.join();

  // kick off a registering message
  // b = false;
  // t = loading_message(out, "Registering the library", b);


  auto [did_load, message] = node.load_shared_library(file_bytes, file_len);
  delete[] file_bytes;


  // finish the registering message
  // b = true; t.join();


  if(!did_load) {
    out << bbts::red << "Failed to register the library : \"" << message << "\"\n" << bbts::reset;
    return false;
  } else {
    out << bbts::green << "Sucessfully register the library: \"" << message << "\"\n" << bbts::reset;
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

std::string stick(std::vector<std::string> s, std::string delimiter){
  std::string stick_str;
  for(int i = 0; i < s.size(); i++){
    stick_str += s[i];
    if(i != s.size() - 1){
      stick_str += ",";
    }
  }
  return stick_str;
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
  // std::atomic_bool b; b = false;
  // auto t = loading_message(out, "Loading tensors from a file", b);

  // try to open the file
  std::ifstream in(file_list);

  if(in.fail()) {
    // finish the loading message
    // b = true; t.join();

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
      // b = true; t.join();
      out << bbts::red << "The file list format must be <tid>|<format>|<file> \n" << bbts::reset;
      return false;
    }

    // make sure this is actually an integer
    std::string tid_string = values[0].c_str();
    // Check if tid is a non-negative integer
    for (int i = 0; i < tid_string.size(); i++){
      if (!isdigit(tid_string[i])){
        // b = true; t.join();
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
      // b = true; t.join();
      out << bbts::red << "\nCould not find the tensor file: " << concated_path << " \n" << bbts::reset;
      return false;
    } 


    // right now it is hardcoded to add  tensors in front of it

    // store this <tid, type, filepath>
    parsed_file_list.push_back({parsed_tid, values[1], "tensors/" + values[2]});
  }

  auto [did_load, message] = node.load_tensor_list(parsed_file_list);

  // finish the registering message
  // b = true; t.join();

  if(!did_load) {
    out << bbts::red << "Failed to load the tensor list : \"" << message << "\"\n" << bbts::reset;
    return false;
  } else {
    // out << bbts::green << message << bbts::reset;
    return true;
  }

  return false;
}






bool generate_binary_file(const unsigned row_num, const unsigned col_num, const std::string &file){
  srand(0);
  //create random data
  std::vector<double> data(row_num * col_num);
  for(int i = 0;i < row_num * col_num;i++){
    if(i % 2 == 0) data[i] = 0.1;
    else data[i] = - 0.5;
  }

  //save it to file
  cnpy::npy_save(file,&data[0],{col_num, row_num},"w");

  
  return true;
}

cnpy::NpyArray load_binary_file(std::ostream &out, bbts::node_t &node, const std::string &file){
  //load it into a new array
  cnpy::NpyArray arr = cnpy::npy_load(file);
  
  size_t size_of_arr = 1;
  //print out shape of binary file
  // out <<  "shape: ";
  // out << "( ";
  // for(size_t i: arr.shape){
  //   size_of_arr *= i;
  //   out << i << " ";
  // }
  // out << ")\n";

  // //print out world size
  // out <<  "word size: ";
  // out << arr.word_size << "\n";

  double* loaded_data = arr.data<double>();
  
  // out <<  "data: ";
  // for(int i = 0; i < size_of_arr; i++) out<< loaded_data[i] << " ";
  // out << "\n";

  
  //make sure the loaded data matches the saved data
  // assert(arr.word_size == sizeof(long));
  // assert(arr.shape.size() == 2 && arr.shape[0] == col_num && arr.shape[1] == row_num );
  // for(int i = 0; i < row_num * col_num;i++) assert(data[i] == loaded_data[i]);

  return arr;
}


std::map<std::string, bbts::tid_t> load_tensor_prep(std::ostream &out, bbts::node_t &node, std::map<std::tuple<int32_t, int32_t>, std::vector<std::vector<double>>> tensor_relation,const std::string &tensor_type,int32_t row_size, int32_t col_size){
  mkdir("tensors", 0777);
  std::map<std::string, bbts::tid_t> id_map;
  std::ofstream filelist;

  filelist.open("tensors/filelist.txt");
  if(!filelist.is_open()){
    out << "Could not open file tensors/filelist.txt.\n";
    return id_map;
  }
  
  

  //print out map
  for(std::map<std::tuple<int32_t, int32_t>, std::vector<std::vector<double>>>::iterator iter = tensor_relation.begin(); iter != tensor_relation.end(); ++iter){
      std::tuple key =  iter->first;
      std::string tensor_file = "t" + std::to_string(current_tid);
      std::ofstream tfile;
      tfile.open("tensors/" + tensor_file);
      if(!tfile.is_open()){
        out << "Could not open file tensors/" << tensor_file << ".\n";
        return id_map;
      }
  
      out << "\n( " << std::get<0>(key) << " , " << std::get<1>(key) << " ): ";
      filelist << current_tid << "|" << tensor_type << "|" << tensor_file;
      std::string key_str = std::to_string(std::get<0>(key)) + "," + std::to_string(std::get<1>(key));
      id_map.insert(std::pair(key_str, current_tid++));
      tfile << row_size << "|" << col_size << "|";
      std::vector<std::vector<double>> v = iter->second;
      out << "[";
      for(std::vector<double> row: v){
        out << "[";
        for(double col: row){
          out <<  col << " ";
          tfile << col << " ";
        }
        out << "]";
      }
      out << "]";
      filelist << "\n";
    }
    out << "\n\n\n";
    
  filelist.close();

  return id_map;
}




void create_id_table(std::ostream &out, bbts::node_t &node, const std::string &db, std::map<std::string, bbts::tid_t> id_map, const std::string &table_name){
  const char* db_name = db.c_str();
  // char* sql0 = "DROP TABLE TENSOR_IDS";
  // execute_command(sql0, db_name, 0);

  std::string sql1 = "CREATE TABLE " + table_name + "(" \
              "TOS_ID INT PRIMARY KEY NOT NULL," \
              "TRA_ID VARCHAR(100) NOT NULL);";

  const char* sql1_char = sql1.c_str();
  execute_command(sql1_char, db_name, 0);

  std::string sql_str;
  std::map< std::string, bbts::tid_t>::iterator it;
  bbts::tid_t max_tid;
  for(it = id_map.begin(); it != id_map.end(); it++){
    sql_str += "INSERT INTO " + table_name + " (TOS_ID, TRA_ID)\n";
    sql_str += ("VALUES (" + std::to_string(it->second) + ", \'" + it->first + "\'); \n");
    max_tid = it->second > current_tid? it->second : current_tid;
  }
  if(max_tid > current_tid) current_tid = max_tid + 1;

  const char* sql2 = sql_str.c_str();

  // out << sql2;
  execute_command(sql2, db_name, 0);

}


void create_udf_table(std::ostream &out, bbts::node_t &node, const std::string &db){
  const char* db_name = db.c_str();
  std::unordered_map<std::string, std::tuple<bbts::ud_id_t, bool, bool, size_t, size_t>> udfs_info = node.get_udfs_info();

  char* sql1 = "CREATE TABLE KERNEL_FUNC(" \
              "KERNEL_NAME VARCHAR(100) PRIMARY KEY NOT NULL," \
              "KERNEL_ID INT NOT NULL," \
              "IS_ASS BOOLEAN NOT NULL," \
              "IS_COMM BOOLEAN NOT NULL," \
              "NUM_IN INT NOT NULL," \
              "NUM_OUT INT NOT NULL);";

  execute_command(sql1, db_name, 0);

  std::string record;
  std::string sql_str;
  std::unordered_map<std::string, std::tuple<bbts::ud_id_t, bool, bool, size_t, size_t>>::iterator udfs_info_it;
  for(udfs_info_it = udfs_info.begin(); udfs_info_it != udfs_info.end(); udfs_info_it++){
    sql_str += "INSERT INTO KERNEL_FUNC (KERNEL_NAME, KERNEL_ID, IS_ASS, IS_COMM, NUM_IN, NUM_OUT)\n";
    record = "\'";
    record += udfs_info_it->first;
    record += "\',";
    record += std::to_string(std::get<0>(udfs_info_it->second));
    record += ", ";
    record += std::to_string(std::get<1>(udfs_info_it->second));
    record += ", ";
    record += std::to_string(std::get<2>(udfs_info_it->second));
    record += ", ";
    record += std::to_string(std::get<3>(udfs_info_it->second));
    record += ", ";
    record += std::to_string(std::get<4>(udfs_info_it->second));
    record += " ";
    sql_str += ("VALUES (" + record + "); \n");
  }
    
  const char* sql2 = sql_str.c_str();

  // out << sql2;
  execute_command(sql2, db_name, 0);


}


void update_tra_id_in_id_table(std::ostream &out, bbts::node_t &node,const std::string &db, std::map<std::string, bbts::tid_t> id_map, const std::string &tr_name){
  const char* db_name = db.c_str();
  
  std::string sql_str;
  std::map<std::string, bbts::tid_t>::iterator it;

  bbts::tid_t max_tid;
  for(it = id_map.begin(); it != id_map.end(); it++){
    sql_str += "UPDATE " + tr_name + " SET TRA_ID = \'" + it->first + "\' WHERE TOS_ID = " + std::to_string(it->second) + ";\n";
    max_tid = it->second > current_tid? it->second : current_tid;
  }
  if(max_tid > current_tid) current_tid = max_tid + 1;

  const char* sql2 = sql_str.c_str();

  // out << sql2;
  execute_command(sql2, db_name, 0);

}



void delete_record_from_id_table(std::ostream &out, bbts::node_t &node,const std::string &db, std::map<bbts::tid_t, bool> bool_map, const std::string tr_name){
  const char* db_name = db.c_str();
  
  // sql_str += "UPDATE TENSOR_IDS (TOS_ID, TRA_ID)\n";
  std::string sql_str;
  std::map<bbts::tid_t, bool>::iterator it;
  for(it = bool_map.begin(); it != bool_map.end(); it++){
    if(!(it -> second)){
      sql_str += "DELETE FROM " + tr_name + " WHERE TOS_ID = " + std::to_string(it->first) + ";\n";
    } 
  }

  const char* sql2 = sql_str.c_str();

  // out << sql2;
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

  // out << sql2;
  execute_command(sql2, db_name, 0);


  myfile.close();
  
}

void update_tos_id_in_id_table(std::ostream &out, bbts::node_t &node,const std::string &db, std::vector<bbts::tid_t> input_tid_list, std::vector<bbts::tid_t> output_tid_list, int max_tid){
  // get info from id_table
  sqlite3_stmt * stmt;
  sqlite3 *sql_db;
  const char* db_name = db.c_str();

  std::map<std::string, bbts::tid_t> id_map;
  std::map<bbts::tid_t, bool> bool_map;
  
  std::vector<std::string> tra_id_list;
  const char* sql = "SELECT TRA_ID FROM TENSOR_IDS";
  // execute_command(sql, db_name, data);

  if(sqlite3_open(db_name, &sql_db) == SQLITE_OK){
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
      tra_id_list.push_back(tra_id);
    }
  }
  else{
    std::cout << "Failed to open db\n";
  }

  sqlite3_close(sql_db);

  
  for(int i = 0; i < input_tid_list.size(); i++){
    bool_map.insert(std::pair<bbts::tid_t, bool>(input_tid_list[i], false));
  }
  delete_record_from_id_table(out, node, db, bool_map, "TENSOR_IDS");

  for(int i = 0; i < output_tid_list.size(); i++){
    id_map.insert(std::pair<std::string, bbts::tid_t>(tra_id_list[i], output_tid_list[i]));
  }


  // create_id_table(out, node, db, id_map);

}



auto create_tensors(std::ostream &out, bbts::node_t &node, const std::string &file, const int32_t row_split, const int32_t col_split, const std::string &tensor_type, const std::string &table_name, const std::string &db){
  
  cnpy::NpyArray arr = load_binary_file(out, node, file);
  double* data = arr.data<double>();
  std::map<std::string, bbts::tid_t> id_map;



  std::map<std::tuple<int32_t, int32_t>, std::vector<std::vector<double>>> tensor_relation;
  int32_t num_rows = arr.shape[0];
  int32_t num_cols = arr.shape[1];

  if(row_split > num_rows || col_split > num_cols){
    out << "Can not split a matrix with split number more than row/col number.\n";
    return;
  }
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

  id_map = load_tensor_prep(out, node, tensor_relation, tensor_type, row_size, col_size);
  load_tensors(out, node, "tensors/filelist.txt");
  create_id_table(out, node, db, id_map, table_name);
}

std::map<std::string, bbts::tid_t> get_id_map_from_sqlite(std::ostream &out,bbts::node_t &node, const char* db_char, bbts::tid_t max_tid, const std::string &tr_name, bool &relation_exist_flag){
  // get info from id_table
  sqlite3_stmt * stmt;
  sqlite3 *sql_db;

  std::map<std::string, bbts::tid_t> id_map;
  const std::string sql_str = "SELECT * FROM " + tr_name + ";\n";
  const char* sql = sql_str.c_str();
  // execute_command(sql, db_name, data);

  if(sqlite3_open(db_char, &sql_db) == SQLITE_OK){
     // preparing the statement
    int rc = sqlite3_prepare_v2(sql_db, sql,-1, &stmt, NULL);

    if(rc != SQLITE_OK){
      out << "Relation \'" << tr_name << "\' does not exist.\n";
      relation_exist_flag = false;
      return id_map;
    }
    // sqlite3_step(stmt); // executing the statement
    while((rc = sqlite3_step(stmt)) == SQLITE_ROW){
      bbts::tid_t tid = sqlite3_column_int(stmt, 0);
      max_tid = tid > max_tid? tid : max_tid;
      std::string tra_id = std::string(reinterpret_cast< const char* >(sqlite3_column_text(stmt, 1)));
      id_map.insert(std::pair<std::string, bbts::tid_t>(tra_id, tid));
      // out << "tid: " << tid << "\n";
      // out << "tra_id: " << tra_id << "\n";
      // out << "map key: (" << split(tra_id, ",")[0] << " , " << split(tra_id, ",")[1] << ")" << " value: " << tid << "\n";
    }
  }
  else{
    std::cout << "Failed to open db\n";
  }

  sqlite3_close(sql_db);
  return id_map;
}



bbts::abstract_ud_spec_id_t get_udf_id_from_sqlite(std::ostream &out,bbts::node_t &node, const char* db_char, const std::string &kernel_name, bool &udf_exist_flag){
  // get udf_name from id_table
  sqlite3_stmt * stmt;
  sqlite3 *sql_db;
  bbts::abstract_ud_spec_id_t udf_id;
  std::string sql_str = "SELECT KERNEL_ID FROM KERNEL_FUNC WHERE KERNEL_NAME = \'" + kernel_name + "\';\n";
  const char* sql = sql_str.c_str();

  if(sqlite3_open(db_char, &sql_db) == SQLITE_OK){
     // preparing the statement
    int rc = sqlite3_prepare_v2(sql_db, sql,-1, &stmt, NULL);

    if(rc != SQLITE_OK){
      out << "ERROR preparing sql statement";
      return -1;
    }
    // exist_cnt check whether kernel func is inside the table
    int exist_cnt = 0;
    while((rc = sqlite3_step(stmt)) == SQLITE_ROW){
      udf_id = sqlite3_column_int(stmt, 0);
      exist_cnt++;
    }
    if(exist_cnt == 0){
      out << "Kernel function \'" << kernel_name << "\' does not exist.\n";
      udf_exist_flag = false;
      return -1;
    }
  }

  else{
    std::cout << "Failed to open db\n";
  }
  out << "udf_id: " << udf_id << "\n";
  return udf_id;
}

void check_input_output_size_for_kernel_func(std::ostream &out,bbts::node_t &node, const char* db_char, 
                                            const std::string &kernel_func, int input_size, int output_size){
  // get udf_name from id_table
  sqlite3_stmt * stmt;
  sqlite3 *sql_db;
  std::string udf_name = "";
  std::string sql_pre = "SELECT * FROM KERNEL_FUNC WHERE KERNEL_NAME = \'" + kernel_func + "\';\n";
  const char* sql = sql_pre.c_str();


  if(sqlite3_open(db_char, &sql_db) == SQLITE_OK){
     // preparing the statement
    int rc = sqlite3_prepare_v2(sql_db, sql,-1, &stmt, NULL);

    if(rc != SQLITE_OK){
      out << "Kernel function " << kernel_func << "does not exist.\n";
      return;
    }
    // sqlite3_step(stmt); // executing the statement
    while((rc = sqlite3_step(stmt)) == SQLITE_ROW){
      std::string string_name = std::string(reinterpret_cast< const char* >(sqlite3_column_text(stmt, 0)));
        bbts::tid_t id = sqlite3_column_int(stmt, 1);
        std::string is_ass = std::string(reinterpret_cast< const char* >(sqlite3_column_text(stmt, 2)));
        std::string is_comm = std::string(reinterpret_cast< const char* >(sqlite3_column_text(stmt, 3)));
        int num_in = sqlite3_column_int(stmt, 4);
        int num_out = sqlite3_column_int(stmt, 5);
        // out << "kernel_name: " << string_name << ",   kernel_id: " << id << ",   is_ass: " <<  is_ass << 
        //     ",   is_comm: " << is_comm << ",   num_in: " << num_in << ",   num_out: " << num_out << "\n";

      if(input_size != num_in || output_size != num_out){
        out << "input_size: " << input_size << "| num_in: " << num_in << "| output_size: " << output_size << " | num_out: " << num_out << "\n";
        out << "The input size or output size does not match to the provided kernel function\n";
        return;
      }
      // if(int(check_ass) != stoi(is_ass) || int(check_comm) != stoi(is_comm) ){
      //   out << "Failed to satify the condition of associative and commutative for the kernel function " << kernel_func << ".\n";
      //   return;
      // }
    }
  }

  else{
    std::cout << "Failed to open db\n";
  }

  
  return;
}


void rename_sqlite_table(const std::string &db, const std::string &old_table_name, const std::string &new_table_name){
  const char* db_char= db.c_str();
  sqlite3_stmt * stmt;
  sqlite3 *sql_db;

  const std::string &sql_str = "ALTER TABLE " + old_table_name + " RENAME TO " + new_table_name + ";\n";
  const char* sql = sql_str.c_str();
  
  execute_command(sql, db_char, 0);
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
  
  const char* db_char= db.c_str();
  
  check_input_output_size_for_kernel_func(out,node, db_char, "uniform", funs[0].input_types.size(), funs[0].output_types.size());
 

  // std::string file_path = "TRA_commands_matrix_generation.sbbts";
  std::ofstream gen(file_path);
  bbts::compile_source_file_t gsf{.function_specs = funs, .commands = commands};
  gsf.write_to_file(gen);
  gen.close();
  load_text_file(out, node, file_path);
}



// Generate commands for aggregation/join/transform
void generate_tra_op_commands(std::ostream &out, bbts::node_t &node,
                          const std::string &kernel_name, 
                          std::map<std::string, bbts::tid_t> &output_mapping, 
                          std::map<std::string, std::vector<bbts::tid_t>> &input_mapping,
                          const std::string &file_path,
                          const std::string &db,
                          bool &relation_exist_flag) {
  
  
  if(!relation_exist_flag){
    return;
  }
  std::vector<bbts::abstract_ud_spec_t> funs;

  // commands
  std::vector<bbts::abstract_command_t> commands;
  
  bool udf_exist_flag = true;

  // get udf_name from id_table
  const char* db_char= db.c_str();
  bbts::abstract_ud_spec_id_t udf_id = get_udf_id_from_sqlite(out, node, db_char, kernel_name, udf_exist_flag);

  if(!udf_exist_flag){
    return;
  }
  // out << "udf_id: " << udf_id << "\n";
  out << "intput_mapping.size(): " << input_mapping.size() << "\n";
  out << "output_mapping.size(): " << output_mapping.size() << "\n";

  std::map<std::string, std::vector<bbts::tid_t>>::iterator input_it;

  for(input_it = input_mapping.begin(); input_it != input_mapping.end(); input_it++){
    out << "input: " << input_it->second[0] << "\n";
    out << "output: " << output_mapping.find(input_it->first)->second << "\n";
    commands.push_back(
        bbts::abstract_command_t{.ud_id = udf_id,
                                  .type = bbts::abstract_command_type_t::APPLY,
                                  .input_tids = input_it->second,
                                  .output_tids = {output_mapping.find(input_it->first)->second},
                                  .params = {}});
    check_input_output_size_for_kernel_func(out,node, db_char, kernel_name, commands[commands.size() - 1].input_tids.size(), 
                                            commands[commands.size() - 1].output_tids.size());
  }
    
  
  std::vector<std::string> input_types_list(input_mapping.begin()->second.size(),"dense");
  funs.push_back(bbts::abstract_ud_spec_t{.id = udf_id,
                                          .ud_name = kernel_name,
                                          .input_types = input_types_list,
                                          .output_types = {"dense"}});


  
  // std::string file_path = "TRA_commands_aggregation.sbbts";
  std::ofstream gen(file_path);
  bbts::compile_source_file_t gsf{.function_specs = funs, .commands = commands};
  gsf.write_to_file(gen);
  gen.close();
  load_text_file(out, node, file_path);
}

std::map<std::string, bbts::tid_t> generate_id_map_for_all_current_node_from_cluster(std::ostream &out, bbts::node_t &node, const std::string &db){

  const char* db_char= db.c_str();
  sqlite3_stmt * stmt;
  sqlite3 *sql_db;
  std::vector< std::vector < std:: string > > result;
  std::map<std::string, bbts::tid_t> id_map;
  std::map<bbts::tid_t, std::string> result_map;
  int max_tid = 0;

  // get all tid that is inside cluster
  std::vector<bbts::tid_t> tid_list = get_all_tid_from_all_nodes(out, node);

  for(auto tid: tid_list){
    max_tid = tid > max_tid ? tid : max_tid;

    std::string sql_str = "SELECT * FROM TENSOR_IDS WHERE TOS_ID = " + std::to_string(tid) + ";\n";
    const char* sql_cmd = sql_str.c_str();

    if(sqlite3_open(db_char, &sql_db) == SQLITE_OK){
      // preparing the statement
      int rc = sqlite3_prepare_v2(sql_db, sql_cmd,-1, &stmt, NULL);

      if(rc != SQLITE_OK){
        out << "ERRORS PREPARING THE STATEMENT";
        return id_map;
      }
      
      while((rc = sqlite3_step(stmt)) == SQLITE_ROW){
        std::string tra_id = std::string(reinterpret_cast< const char* >(sqlite3_column_text(stmt, 0)));
        id_map.insert(std::pair(tra_id, tid));
      }
    }
    else{
      std::cout << "Failed to open db\n";
    }

    sqlite3_finalize(stmt);
  }
  if(max_tid > current_tid) current_tid = max_tid + 1;
  
  sqlite3_close(sql_db);
  return id_map;

}

std::map<std::string, bbts::tid_t> generate_id_map_for_all_current_node_from_sqlite(std::ostream &out, bbts::node_t &node, const std::string &db, const std::string &tr_name, bool &relation_exist_bool_flag){

  const char* db_char= db.c_str();
  sqlite3_stmt * stmt;
  sqlite3 *sql_db;
  std::vector< std::vector < std:: string > > result;
  std::map<std::string, bbts::tid_t> id_map;
  std::map<bbts::tid_t, std::string> result_map;
  int max_tid = 0;


  

  std::string sql_str = "SELECT * FROM " + tr_name + ";\n";
  const char* sql_cmd = sql_str.c_str();

  if(sqlite3_open(db_char, &sql_db) == SQLITE_OK){
    // preparing the statement
    int rc = sqlite3_prepare_v2(sql_db, sql_cmd,-1, &stmt, NULL);

    if(rc != SQLITE_OK){
      out << "Relation \'" << tr_name << "\' does not exist.\n";
      relation_exist_bool_flag = false;
      return id_map;
    }
    
    while((rc = sqlite3_step(stmt)) == SQLITE_ROW){
      bbts::tid_t tid = sqlite3_column_int(stmt, 0);
      max_tid = tid > max_tid ? tid : max_tid;
      std::string tra_id = std::string(reinterpret_cast< const char* >(sqlite3_column_text(stmt, 1)));
      id_map.insert(std::pair(tra_id, tid));
    }
  }
  else{
    std::cout << "Failed to open db\n";
  }

  sqlite3_finalize(stmt);
  
  if(max_tid > current_tid) current_tid = max_tid + 1;
  
  sqlite3_close(sql_db);
  return id_map;
}


void generate_aggregation_tra(std::ostream &out, bbts::node_t &node, std::map<std::string, bbts::tid_t> &output_mapping, 
                              std::map<std::string, std::vector<bbts::tid_t>> &input_mapping, const std::string &db, 
                              const std::vector<std::string> dimension_list, const std::string &tr_name, 
                              const std::string &stored_tr_name, bool &relation_exist_flag){
  
  std::map<std::string, bbts::tid_t> id_map = generate_id_map_for_all_current_node_from_sqlite(out, node, db, tr_name, relation_exist_flag);

  if(!relation_exist_flag){
    return; 
  }
  // Generate Aggregate commands pairs based on aggregate dimension list
  
  std::map<std::string, bbts::tid_t>::iterator map_it;
  std::vector<bbts::tid_t> removed_tid_list;
  std::vector<std::string> input_tr_name = {tr_name};

  if(id_map.size() == 0){
    out << "There is no id pairs stored for relation " << tr_name << ".\n";
    return;
  }

  for(map_it = id_map.begin(); map_it!= id_map.end(); map_it++){
    removed_tid_list.push_back(map_it->second);
    std::string agg_key;
    std::vector<std::string> split_key = split(map_it->first,",");
    
    for(int i = 0; i < dimension_list.size(); i++){
      agg_key += (split_key[stoi(dimension_list[i])]);
      if(i != dimension_list.size() - 1){
        agg_key += ",";
      } 
    }

    std::map<std::string, std::vector<bbts::tid_t>>::iterator it = input_mapping.find(agg_key);
    if(it == input_mapping.end()){
      std::vector<bbts::tid_t> mapping_list;
      mapping_list.push_back(map_it->second);
      input_mapping.insert(std::pair(agg_key, mapping_list));
      output_mapping.insert(std::pair(agg_key, current_tid++));
    }
    else{
      it->second.push_back(map_it->second);
    }
  }
  
  stored_R.insert(std::pair(stored_tr_name, output_mapping));

}




void generate_join_tra(std::ostream &out, bbts::node_t &node, std::map<std::string, bbts::tid_t> &output_mapping, 
                      std::map<std::string, std::vector<bbts::tid_t>> &input_mapping, const std::string &db, 
                      std::vector<std::string> dimension_list_l, std::vector<std::string> dimension_list_r, 
                      const std::string &tr_name_l, const std::string &tr_name_r, const std::string &stored_tr_name,
                      bool &relation_exist_flag){
  

  std::map<std::string, bbts::tid_t> id_map_l = generate_id_map_for_all_current_node_from_sqlite(out, node, db, tr_name_l, relation_exist_flag);
  std::map<std::string, bbts::tid_t> id_map_r = generate_id_map_for_all_current_node_from_sqlite(out, node, db, tr_name_r, relation_exist_flag);

  // return if relation does not exist
  if(!relation_exist_flag){
    return;
  }
  // out << "id_map_l.size: " << id_map_l.size() << " | id_map_r.size: " << id_map_r.size() << "\n";

  // Generate Aggregate commands pairs based on aggregate dimension list
  
  std::map<std::string, bbts::tid_t>::iterator map_it1;
  std::map<std::string, bbts::tid_t>::iterator map_it2;


  // check if join size for two relations are the same
  if(dimension_list_l.size() != dimension_list_r.size()){
    out << "join dimension mismatch\n";
    return;
  }
  

  //
  for(map_it1 = id_map_l.begin(); map_it1!= id_map_l.end(); map_it1++){
    for(map_it2 = id_map_r.begin(); map_it2!= id_map_r.end(); map_it2++){
    
      std::vector<std::string> split_key_l = split(map_it1->first,",");
      std::vector<std::string> split_key_r = split(map_it2->first,",");
      
      // check if left key is equal to right key on all join dimensions
      bool join_key_match = true;
      for(int i = 0; i < dimension_list_l.size(); i++){
        if(split_key_l[stoi(dimension_list_l[i])].compare(split_key_r[stoi(dimension_list_r[i])]) != 0){
          join_key_match = false;
          break;
        }
      }
      // if two keys match on their join dimensions
      if(join_key_match){
        
        // create new key for join
        std::vector<std::string> join_key_list;
        int dim_idx = 0;

        for(int i = 0; i < split_key_l.size(); i++){
          join_key_list.push_back(split_key_l[i]);
        }
        for(int i = 0; i < split_key_r.size(); i++){
          out<< "i: " << i << "  , dimension_list_r[dim_idx]: " << dimension_list_r[dim_idx] << "\n";
          if(i == stoi(dimension_list_r[dim_idx])){
            if(dim_idx < dimension_list_r.size() - 1){
              dim_idx++;
            }
          }
          else{
            join_key_list.push_back(split_key_r[i]);
          }
        }
        std::string join_key;
        for(int i = 0; i < join_key_list.size(); i++){
          join_key += join_key_list[i];
          if(i != join_key_list.size() - 1){
            join_key += ",";
          }
        }
        out << "joinkey: " << join_key << "\n";
        std::vector<bbts::tid_t> mapping_list;
        mapping_list.push_back(map_it1->second);
        mapping_list.push_back(map_it2->second);
        input_mapping.insert(std::pair(join_key, mapping_list));
        output_mapping.insert(std::pair(join_key, current_tid++));
      
      }

    }
  }
  
  stored_R.insert(std::pair(stored_tr_name, output_mapping));
}
  



// void reKey(std::ostream &out, bbts::node_t &node, const std::string &db, int keyFunc(std::vector<int>),const std::string &tr_name, const std::string &stored_tr_name, const std::string &kernel_name){

//   const char* db_char= db.c_str();
//   sqlite3_stmt * stmt;
//   sqlite3 *sql_db;
//   std::vector<std::vector<std:: string>> result;
//   std::map<bbts::tid_t, std::string> id_map;
//   std::map<std::string, bbts::tid_t> output_mapping;

//   const std::string sql_str = "SELECT * FROM " + tr_name + ";\n";
//   const char* sql = sql_str.c_str();
//   // execute_command(sql, db_name, data);
  
//   if(sqlite3_open(db_char, &sql_db) == SQLITE_OK){
//      // preparing the statement
//     int rc = sqlite3_prepare_v2(sql_db, sql,-1, &stmt, NULL);

//     if(rc != SQLITE_OK){
//       out << "ERRORS PREPARING THE STATEMENT";
//       return;
//     }
//     // sqlite3_step(stmt); // executing the statement
//     while((rc = sqlite3_step(stmt)) == SQLITE_ROW){
//       bbts::tid_t tos_id = sqlite3_column_int(stmt, 0);
//       std::string tra_id = std::string(reinterpret_cast< const char* >(sqlite3_column_text(stmt, 1)));
//       out << "tos_id: " << std::to_string(tos_id) << " tra_id: " << tra_id << "\n";
//       id_map.insert(std::pair(tos_id, tra_id));
//     }
//   }
//   else{
//     std::cout << "Failed to open db\n";
//   }

//   //change key based on keyFunc
//   std::map<bbts::tid_t, std::string>::iterator it;
//   for(it = id_map.begin(); it != id_map.end(); it++){
//     bbts::tid_t tos_id = it -> first;
//     std::string old_tra_id = it -> second;

//     std::vector<int> tra_id_vec;
//     std::istringstream ss(old_tra_id);
//     std::string read_number;
    
//     while(std::getline(ss, read_number, ',')){
//       tra_id_vec.push_back(stoi(read_number));
//     }
    
//     std::string new_tra_id = std::to_string(keyFunc(tra_id_vec));
//     it -> second = new_tra_id;
//     output_mapping.insert(std::pair(new_tra_id, tos_id));
//   }  

//   sqlite3_finalize(stmt);
//   sqlite3_close(sql_db);

 
//   stored_R.insert(std::pair(stored_tr_name, output_mapping));
 
// }




// void filter(std::ostream &out, bbts::node_t &node, const std::string &db, bool boolFunc(std::vector<int>), const std::string &tr_name, const std::string &stored_tr_name, const std::string &kernel_name){

//   const char* db_char= db.c_str();
//   sqlite3_stmt * stmt;
//   sqlite3 *sql_db;
//   std::vector<std::vector<std:: string>> result;
//   std::map<bbts::tid_t, std::string> id_map;
//   std::map<bbts::tid_t, bool> bool_map;
//    std::map<std::string, bbts::tid_t> output_mapping;



//   const std::string sql_str = "SELECT * FROM " + tr_name + ";\n";
//   const char* sql = sql_str.c_str();
//   // execute_command(sql, db_name, data);
  
//   if(sqlite3_open(db_char, &sql_db) == SQLITE_OK){
//      // preparing the statement
//     int rc = sqlite3_prepare_v2(sql_db, sql,-1, &stmt, NULL);

//     if(rc != SQLITE_OK){
//       out << "ERRORS PREPARING THE STATEMENT";
//       return;
//     }
//     // sqlite3_step(stmt); // executing the statement
//     while((rc = sqlite3_step(stmt)) == SQLITE_ROW){
//       bbts::tid_t tos_id = sqlite3_column_int(stmt, 0);
//       std::string tra_id = std::string(reinterpret_cast< const char* >(sqlite3_column_text(stmt, 1)));
//       id_map.insert(std::pair(tos_id, tra_id));
//     }
//   }
//   else{
//     std::cout << "Failed to open db\n";
//   }

//   //change key based on keyFunc
//   std::map<bbts::tid_t, std::string>::iterator it;
//   for(it = id_map.begin(); it != id_map.end(); it++){
//     bbts::tid_t tos_id = it -> first;
//     std::string tra_id = it -> second;

//     std::vector<int> tra_id_vec;
//     std::istringstream ss(tra_id);
//     std::string read_number;
    
//     while(std::getline(ss, read_number, ',')){
//       tra_id_vec.push_back(stoi(read_number));
//     }
    
//     bool tra_filter = boolFunc(tra_id_vec);
//     if(tra_filter){
//       output_mapping.insert(std::pair(tra_id, tos_id));
//     }
//     // bool_map.insert(std::pair(tos_id, tra_filter));
    
//   }  

//   sqlite3_finalize(stmt);
//   sqlite3_close(sql_db);


//   // delete_record_from_id_table(out, node, db, bool_map);
 
// }


void generate_transform_tra(std::ostream &out, bbts::node_t &node, std::map<std::string, bbts::tid_t> &output_mapping, 
                            std::map<std::string, std::vector<bbts::tid_t>> &input_mapping, const std::string &db, 
                            const std::string &stored_tr_name, std::vector<bbts::tid_t> removed_tid_list, 
                            const std::string &tr_name, bool &relation_exist_flag){
  
  // update_tos_id_in_id_table(out, node, db, input_tid_list, output_tid_list, max_tid);
  
  std::map<std::string, bbts::tid_t> id_map = generate_id_map_for_all_current_node_from_sqlite(out, node, db, tr_name, relation_exist_flag);

  if(!relation_exist_flag){
    return;
  }

  std::map<std::string, bbts::tid_t>::iterator map_it;
  std::vector<std::string> input_tr_name = {tr_name};

  for(map_it = id_map.begin(); map_it != id_map.end(); map_it++){
    std::vector<bbts::tid_t> mapping_list;
    mapping_list.push_back(map_it->second);
    out << "input_mapping: ( " << map_it->first << " )\n";
    for(auto i: mapping_list){
      out << i << "  "; 
    }
    out << "\n";
    input_mapping.insert(std::pair(map_it->first, mapping_list));
    out << "output mapping: " << current_tid << "\n";
    output_mapping.insert(std::pair(map_it->first, current_tid++));
  }

  stored_R.insert(std::pair(stored_tr_name, output_mapping));

}

void math_op_prep(std::ostream &out, const std::string &num_or_dim, std::string &num_or_dim_type, std::vector<std::string> &num_or_dim_list){
  if(num_or_dim.front() == '<' and num_or_dim.back() == '>'){
    num_or_dim_type = "dimension";
    // out << "num_or_dim.substr(1, num_or_dim.find_last_of('>') - 1): " << num_or_dim.substr(1, num_or_dim.find_last_of('>') - 1) << "\n";
    num_or_dim_list = split(num_or_dim.substr(1, num_or_dim.find_last_of('>') - 1), ",");
  }
  else{
    num_or_dim_type = "number";
    num_or_dim_list = split(num_or_dim, ",");
  }
}

void handle_math_op_for_rekey(std::ostream &out, bbts::node_t &node, const std::string &tr_name, const std::string &op, const std::string &num_or_dim_1, const std::string &num_or_dim_2, const std::string &db){

  int max_tid = 0;
  bool relation_exist_flag = true;
  std::map<std::string, bbts::tid_t> old_id_map = get_id_map_from_sqlite(out, node, db.c_str(), max_tid, tr_name, relation_exist_flag);

  if(!relation_exist_flag){
    return;
  }

  std::map<std::string, bbts::tid_t>::iterator id_map_it;
  std::map<std::string, bbts::tid_t> new_id_map;
  std::string num_or_dim_1_type;
  std::string num_or_dim_2_type;
  std::vector<std::string> num_or_dim_1_list;
  std::vector<std::string> num_or_dim_2_list;

  
  math_op_prep(out, num_or_dim_1, num_or_dim_1_type, num_or_dim_1_list);
  math_op_prep(out, num_or_dim_2, num_or_dim_2_type, num_or_dim_2_list);

  
  if(num_or_dim_1_type != "dimension"){
    out << num_or_dim_1 << "is not a dimension.\n";
    assert(num_or_dim_1_type == "dimension");
  }

  for(id_map_it = old_id_map.begin(); id_map_it != old_id_map.end(); id_map_it++){
    std::vector<std::string> tra_id = split(id_map_it->first, ",");
    std::vector<std::string> new_tra_id;
    for(int idx = 0; idx < tra_id.size(); idx++){
      int i;
      for(i = 0; i < num_or_dim_1_list.size();i++){
        if(idx == stoi(num_or_dim_1_list[i])){
          int changed_number;
          if(num_or_dim_2_list.size() == 1){
            if(num_or_dim_2_type == "dimension"){
              changed_number = stoi(tra_id[stoi(num_or_dim_2_list[0])]);
            }
            else if(num_or_dim_2_type == "number"){
              changed_number = stoi(num_or_dim_2_list[0]);
            }
            else{
              out << "Input has to be numbers of dimensions.\n";
            }
          }
          else if(num_or_dim_1_list.size() != num_or_dim_2_list.size()){
            out << "Two dim/num list has to be same size or one list has size 1.\n";
            return;
          }
          else{
            if(num_or_dim_2_type == "dimension"){
              changed_number = stoi(tra_id[stoi(num_or_dim_2_list[i])]);
            }
            else if(num_or_dim_2_type == "number"){
              changed_number = stoi(num_or_dim_2_list[i]);
            }
            else{
              out << "Input has to be numbers of dimensions.\n";
            }
          }

          if(op == "add" || op == "+"){
            new_tra_id.push_back(std::to_string(stoi(tra_id[stoi(num_or_dim_1_list[i])]) + changed_number));
          }
          else if(op == "subtract" || op == "-"){
            new_tra_id.push_back(std::to_string(stoi(tra_id[stoi(num_or_dim_1_list[i])]) - changed_number));
          }
          else if(op == "multiply" || op == "*"){
            new_tra_id.push_back(std::to_string(stoi(tra_id[stoi(num_or_dim_1_list[i])]) * changed_number));
          }
          else if(op == "divide" || op == "/"){
            if(changed_number == 0 || stoi(tra_id[stoi(num_or_dim_1_list[i])]) % changed_number != 0){
              out << "Error for division. Either denominator is 0 or the remainder of division does not equal to 0.\n";
              return;
            }
            new_tra_id.push_back(std::to_string(stoi(tra_id[stoi(num_or_dim_1_list[i])]) / changed_number));
          }
          else if(op == "mod" || op == "%"){
            if(changed_number == 0){
              out << "Error for mod. Denominator could not be 0.\n";
            }
            new_tra_id.push_back(std::to_string(stoi(tra_id[stoi(num_or_dim_1_list[i])]) % changed_number));
          }
          break;
        }
        
      }
      if(i == num_or_dim_1_list.size()){
        new_tra_id.push_back(tra_id[idx]);
      }
    }
    std::string new_tra_id_str = stick(new_tra_id, ",");
    if(new_id_map.find(new_tra_id_str) != new_id_map.end()){
      out << "Illegal! TRA id is repetitive. \n";
      return;
    }
    new_id_map.insert(std::pair(new_tra_id_str, id_map_it->second));
    // out << "inserted into new_id_map: ( " << new_tra_id_str << " , " << id_map_it->second << " )\n";
  }

  update_tra_id_in_id_table(out, node, db, new_id_map, tr_name);
  
}


void handle_collective_op_for_rekey(std::ostream &out, bbts::node_t &node, const std::string &tr_name, const std::string &op, 
                          const std::string &num_or_dim_1, const std::string &num_or_dim_2, const std::string &db){
  int max_tid = 0;
  bool relation_exist_flag = true;
  std::map<std::string, bbts::tid_t> old_id_map = get_id_map_from_sqlite(out, node, db.c_str(), max_tid, tr_name, relation_exist_flag);

  if(!relation_exist_flag){
    return;
  }

  std::map<std::string, bbts::tid_t>::iterator id_map_it;
  std::map<std::string, bbts::tid_t> new_id_map;
  std::string num_or_dim_1_type;
  std::string num_or_dim_2_type;
  std::vector<std::string> num_or_dim_1_list;
  std::vector<std::string> num_or_dim_2_list;


  
  math_op_prep(out, num_or_dim_1, num_or_dim_1_type, num_or_dim_1_list);
  math_op_prep(out, num_or_dim_2, num_or_dim_2_type, num_or_dim_2_list);
 
  
  if(num_or_dim_1_type != "dimension" || num_or_dim_2_type != "dimension"){
    out << "At least one of the two input lists is not dimension list.\n";
    return;
  }
  if(num_or_dim_2_list.size() != 1){
    out << "The dimension list for the result of collective operation must be size of 1.\n";
    return;
  }

  for(id_map_it = old_id_map.begin(); id_map_it != old_id_map.end(); id_map_it++){
    std::vector<std::string> tra_id = split(id_map_it->first, ",");
    std::vector<std::string> new_tra_id;

    int result = 0;
    for(int i = 0; i < num_or_dim_1_list.size();i++){
      if(op == "sum"){
        result += stoi(tra_id[stoi(num_or_dim_1_list[i])]);
      }
    }
    for(int i = 0; i < tra_id.size(); i++){
      if(i == stoi(num_or_dim_2_list[0])){
        new_tra_id.push_back(std::to_string(result));
        continue;
      }
      int j;
      for(j = 0; j < num_or_dim_1_list.size(); j++){
        if(i == stoi(num_or_dim_1_list[j])){
          break;
        }
      }
      if(j == num_or_dim_1_list.size()){
        new_tra_id.push_back(tra_id[i]);
      }
      
    }

    std::string new_tra_id_str = stick(new_tra_id, ",");
    if(new_id_map.find(new_tra_id_str) != new_id_map.end()){
      out << "Illegal! TRA id is repetitive. \n";
      return;
    }
    new_id_map.insert(std::pair(new_tra_id_str, id_map_it->second));
    // out << "inserted into new_id_map: ( " << new_tra_id_str << " , " << id_map_it->second << " )\n";
  }

  update_tra_id_in_id_table(out, node, db, new_id_map, tr_name);
}

void drop_dimensions_from_relation(std::ostream &out, bbts::node_t &node, const std::string &tr_name, const std::string &drop_dimensions, const std::string &db){
  int max_tid;
  bool relation_exist_flag = true;
  std::map<std::string, bbts::tid_t> old_id_map = get_id_map_from_sqlite(out, node, db.c_str(), max_tid, tr_name, relation_exist_flag);

  if(!relation_exist_flag){
    return;
  }

  std::map<std::string, bbts::tid_t>::iterator id_map_it;
  std::map<std::string, bbts::tid_t> new_id_map;

  std::string drop_dimensions_type;
  std::vector<std::string> drop_dimensions_list;

  
  math_op_prep(out, drop_dimensions, drop_dimensions_type, drop_dimensions_list);

  if(drop_dimensions_type != "dimensions"){
    out << "Input has to be dimensions for drop.\n";
  }

  for(id_map_it = old_id_map.begin(); id_map_it != old_id_map.end(); id_map_it++){
    std::vector<std::string> tra_id = split(id_map_it->first, ",");
    std::vector<std::string> new_tra_id;

    for(int i = 0; i < tra_id.size(); i++){
      int j;
      for(j = 0; j < drop_dimensions_list.size(); j++){
        if(i == stoi(drop_dimensions_list[j])){
          break;
        }
      }
      if(j == drop_dimensions_list.size()){
        new_tra_id.push_back(tra_id[i]);
      }
    }
    std::string new_tra_id_str = stick(new_tra_id, ",");
    new_id_map.insert(std::pair(new_tra_id_str, id_map_it->second));
    // out << "inserted into new_id_map: ( " << new_tra_id_str << " , " << id_map_it->second << " )\n";
  }
  update_tra_id_in_id_table(out, node, db, new_id_map, tr_name);
}

void append_dimension_to_relation(std::ostream &out, bbts::node_t &node, const std::string &tr_name, 
                                  const std::string &append_dimension, const std::string &num_or_dim_1, 
                                  const std::string &op, const std::string &num_or_dim_2, const std::string &db){
  int max_tid = 0;
  bool relation_exist_flag = true;
  std::map<std::string, bbts::tid_t> old_id_map = get_id_map_from_sqlite(out, node, db.c_str(), max_tid, tr_name, relation_exist_flag);

  if(!relation_exist_flag){
    return;
  }

  std::map<std::string, bbts::tid_t>::iterator id_map_it;
  std::map<std::string, bbts::tid_t> new_id_map;

  std::string append_dimension_type;
  std::string num_or_dim_1_type;
  std::string num_or_dim_2_type;

  std::vector<std::string> append_dimension_list;
  std::vector<std::string> num_or_dim_1_list;
  std::vector<std::string> num_or_dim_2_list;

  math_op_prep(out, append_dimension, append_dimension_type, append_dimension_list);
  math_op_prep(out, num_or_dim_1, num_or_dim_1_type, num_or_dim_1_list);
  math_op_prep(out, num_or_dim_2, num_or_dim_2_type, num_or_dim_2_list);

  
  if(append_dimension_type != "dimension"){
    out << append_dimension << "is not a dimension.\n";
    return;
  }
  if(append_dimension_list.size() != 1 || num_or_dim_1_list.size() != 1 || num_or_dim_2_list.size() != 1){
    out << "You can only append one dimension at a time.\n";
    return;
  }

  for(id_map_it = old_id_map.begin(); id_map_it != old_id_map.end(); id_map_it++){
    std::vector<std::string> tra_id = split(id_map_it->first, ",");
    std::vector<std::string> new_tra_id;
    for(int idx = 0; idx < tra_id.size() + 1; idx++){
      int i;
      if(idx == stoi(append_dimension_list[0])){
        int changed_number_left;
        int changed_number_right;
        if(num_or_dim_1_type == "dimension"){
          if(num_or_dim_2_type == "dimension"){
            changed_number_left = stoi(tra_id[stoi(num_or_dim_1_list[0])]);
            out << "changed_number_left: " << changed_number_left << "\n";
            changed_number_right = stoi(tra_id[stoi(num_or_dim_2_list[0])]);
            out << "changed_number_right: " << changed_number_right << "\n";
          }
          else if(num_or_dim_2_type == "number"){
            changed_number_left = stoi(tra_id[stoi(num_or_dim_1_list[0])]);
            changed_number_right = stoi(num_or_dim_2_list[0]);
          }
          else{
            out << "Input has to be number or dimension.\n";
          }
        }
        else if(num_or_dim_1_type == "number"){
          if(num_or_dim_2_type == "dimension"){
            changed_number_left = stoi(num_or_dim_1_list[0]);
            changed_number_right = stoi(tra_id[stoi(num_or_dim_2_list[0])]);
          }
          else if(num_or_dim_2_type == "number"){
            changed_number_left = stoi(num_or_dim_1_list[0]);
            changed_number_right = stoi(num_or_dim_2_list[0]);
          }
          else{
            out << "Input has to be number or dimension.\n";
          }
        }
        else{
          out << "Input has to be number or dimension.\n";
        }

        if(op == "add" || op == "+"){
          new_tra_id.push_back(std::to_string(changed_number_left + changed_number_right));
        }
        else if(op == "subtract" || op == "-"){
          new_tra_id.push_back(std::to_string(changed_number_left - changed_number_right));
        }
        else if(op == "multiply" || op == "*"){
          new_tra_id.push_back(std::to_string(changed_number_left * changed_number_right));
        }
        else if(op == "divide" || op == "/"){
          if(changed_number_right == 0 || changed_number_left % changed_number_right != 0){
            out << "Error for division. Either denominator is 0 or the remainder of division does not equal to 0.\n";
            return;
          }
          new_tra_id.push_back(std::to_string(changed_number_left / changed_number_right));
        }
        else if(op == "mod" || op == "%"){
          if(changed_number_right == 0){
            out << "Error for mod. Denominator could not be 0.\n";
          }
          new_tra_id.push_back(std::to_string(changed_number_left % changed_number_right));
        }
        break;
      }
      
      if(idx == tra_id.size()){
        out << "Input appended index value is invalid.\n";
        return;
      }
      new_tra_id.push_back(tra_id[idx]);

    }
    std::string new_tra_id_str = stick(new_tra_id, ",");
    if(new_id_map.find(new_tra_id_str) != new_id_map.end()){
      out << "Illegal! TRA id is repetitive. \n";
      return;
    }
    new_id_map.insert(std::pair(new_tra_id_str, id_map_it->second));
    // out << "inserted into new_id_map: ( " << new_tra_id_str << " , " << id_map_it->second << " )\n";
  }

  update_tra_id_in_id_table(out, node, db, new_id_map, tr_name);
}




/**************************** compile and run commands from TOS API ***********************/

//load commands

void load_binary_command(std::ostream &out, bbts::node_t &node, const std::string &file_path) {

  // kick off a loading message
  // std::atomic_bool b; b = false;
  // auto t = loading_message(out, "Loading the file", b);

  // try to deserialize
  bbts::parsed_command_list_t cmd_list;
  bool success = cmd_list.deserialize(file_path);

  // finish the loading message
  // b = true; t.join();

  // did we fail
  if(!success) {
    out << bbts::red << "Failed to load the file " << file_path << '\n' << bbts::reset;
    return;
  }

  // kick off a loading message
  // b = false;
  // t = loading_message(out, "Scheduling the loaded commands", b);

  // load the commands we just parsed
  auto [did_load, message] = node.load_commands(cmd_list);

  // finish the loading message
  // b = true; t.join();

  // did we fail
  if(!did_load) {
    out << bbts::red << "Failed to schedule the loaded commands : \"" << message << "\"\n" << bbts::reset;
  }
  else {
    // out << bbts::green << message << bbts::reset;
  }
}


//compile commands
void compile_commands(std::ostream &out, bbts::node_t &node, const std::string &file_path) {

    // kick off a loading message
  // std::atomic_bool b; b = false;
  // auto t = loading_message(out, "Compiling commands", b);

  // compile the commands and load them
  std::cout << "please hit me!!!\n";
  auto [did_compile, message] = node.compile_commands(file_path);

  // finish the loading message  
  // b = true; t.join();

  if(!did_compile) {
    out << bbts::red << "Failed to compile the : \"" << message << "\"\n" << bbts::reset;
  } else {
    // out << bbts::green << message << bbts::reset;
  }
}

//run commands
void run_commands(std::ostream &out, bbts::node_t &node) {

  // kick of a loading message
  // std::atomic_bool b; b = false;
  // auto t = loading_message(out, "Running the commands", b);

  // run all the commands
  auto [did_load, message] = node.run_commands();

  // finish the loading message
  // b = true; t.join();

  // did we fail
  if(!did_load) {
    out << bbts::red << "Failed to run commands : \"" << message << "\"\n" << bbts::reset;
  }
  else {
    // out << bbts::green << message << bbts::reset;
  }
}

void materialize_commands(std::ostream &out, bbts::node_t &node, const std::string &file_path, 
                          const std::string &tr_name,
                          const std::string &db){
  if(std::filesystem::exists(file_path)){
    //compile commands
    compile_commands(out, node, file_path);
    //run commands
    run_commands(out, node);
    std::remove(file_path.c_str());
  }
  else{
    out << file_path << " does not exist";
    return;
  }
  
  // rename_sqlite_table(db, old_table_name, new_table_name);
  create_id_table(out, node, db, stored_R.find(tr_name)->second, tr_name);
}

void delete_tensor_from_cluster(std::ostream &out, bbts::node_t &node, std::vector<bbts::tid_t> id_list, const std::string &file_path, bool &relation_exist_flag){

  std::vector<bbts::abstract_ud_spec_t> funs;

  std::vector<bbts::abstract_command_t> commands = {bbts::abstract_command_t{.ud_id = -1,
                                                                .type = bbts::abstract_command_type_t::DELETE,
                                                                .input_tids = {id_list},
                                                                .output_tids = {},
                                                                .params = {}}};
  
  if(!relation_exist_flag){
    return;
  }
  std::ofstream gen(file_path);
  bbts::compile_source_file_t gsf{.function_specs = funs, .commands = commands};
  gsf.write_to_file(gen);
  gen.close();
  // load_text_file(out, node, file_path);

  if(std::filesystem::exists(file_path)){
    //compile commands
    compile_commands(out, node, file_path);
    //run commands
    run_commands(out, node);
    std::remove(file_path.c_str());
  }
  else{
    out << file_path << " does not exist";
    return;
  }
}




void filter_relations(std::ostream &out, bbts::node_t &node, const std::string &tr_name, const std::string &num_or_dim_1, const std::string &op, 
                      const std::string &num_or_dim_2, const std::string &num_or_dim_3, const std::string &db){
  int max_tid = 0;
  bool relation_exist_flag = true;
  std::map<std::string, bbts::tid_t> old_id_map = get_id_map_from_sqlite(out, node, db.c_str(), max_tid, tr_name, relation_exist_flag);

  if(!relation_exist_flag){
    return;
  }
  std::map<std::string, bbts::tid_t>::iterator id_map_it;
  std::map<bbts::tid_t, bool> bool_id_map;
  std::map<bbts::tid_t, bool>::iterator bool_id_map_it;
  std::vector<bbts::tid_t> removed_tid_list;

  std::string num_or_dim_1_type;
  std::string num_or_dim_2_type;
  std::string num_or_dim_3_type;

  std::vector<std::string> num_or_dim_1_list;
  std::vector<std::string> num_or_dim_2_list;
  std::vector<std::string> num_or_dim_3_list;

  math_op_prep(out, num_or_dim_1, num_or_dim_1_type, num_or_dim_1_list);
  math_op_prep(out, num_or_dim_2, num_or_dim_2_type, num_or_dim_2_list);
  math_op_prep(out, num_or_dim_3, num_or_dim_3_type, num_or_dim_3_list);


  if(num_or_dim_1_list.size() != 1 || num_or_dim_2_list.size() != 1 || num_or_dim_3_list.size() != 1){
    out << "Filter only support filtering on one dimension.\n";
  }

  if(num_or_dim_1_type == "number" && num_or_dim_2_type == "number" && num_or_dim_3_type == "number"){
    out << "Illegal input. Can't take three numbers.\n";
  }

  for(id_map_it = old_id_map.begin(); id_map_it != old_id_map.end(); id_map_it++){
    std::vector<std::string> tra_id = split(id_map_it->first, ",");
    int left_side, right_side;

    if(stoi(num_or_dim_1_list[0]) >= tra_id.size()){
      out << "TRA_ID index out of range.\n";
      return;
    }

    if(stoi(num_or_dim_2_list[0]) >= tra_id.size()){
      out << "TRA_ID index out of range.\n";
      return;
    }

    if(num_or_dim_1_type == "dimension" && num_or_dim_2_type == "dimension"){
      left_side = stoi(tra_id[stoi(num_or_dim_1_list[0])]);
      right_side = stoi(tra_id[stoi(num_or_dim_2_list[0])]);
    }
    else if(num_or_dim_1_type == "dimension" && num_or_dim_2_type == "number"){
      left_side = stoi(tra_id[stoi(num_or_dim_1_list[0])]);
      right_side = stoi(num_or_dim_2_list[0]);
    }
    else if(num_or_dim_1_type == "number" && num_or_dim_2_type == "dimension"){
      left_side = stoi(num_or_dim_1_list[0]);
      right_side = stoi(tra_id[stoi(num_or_dim_2_list[0])]);
    }
    else if(num_or_dim_1_type == "number" && num_or_dim_2_type == "number"){
      left_side = stoi(num_or_dim_1_list[0]);
      right_side = stoi(num_or_dim_2_list[0]);
    }
    else{
      out << "Input has to be either dimension or number.\n";
      return;
    }
    
    int compare_result;
    if(op == "add" || op == "+"){
      compare_result = left_side + right_side;
    }
    else if(op == "subtract" || op == "-"){
      compare_result = left_side - right_side;
    }
    else if(op == "multiply" || op == "*"){
      compare_result = left_side * right_side;
    }
    else if(op == "divide" || op == "/"){
      if(right_side == 0 || left_side % right_side != 0){
        out << "Error for division. Either denominator is 0 or the remainder of division does not equal to 0.\n";
        return;
      }
      compare_result = left_side - right_side;
    }
    else if(op == "mod" || op == "%"){
      if(right_side == 0){
        out << "Error for mod. Denominator could not be 0.\n";
      }
      compare_result = left_side % right_side;
    }
    else{
      out << "Only support operations: add(+), subtract(-), multiply(*), divide(/).\n";
      return;
    }

    int check_equal_result;
    if(num_or_dim_3_type == "dimension"){
      if(stoi(num_or_dim_3_list[0]) >= tra_id.size()){
        out << "TRA_ID index out of range.\n";
        return;
      }
      check_equal_result = stoi(tra_id[stoi(num_or_dim_3_list[0])]);
    }
    else if(num_or_dim_3_type == "number"){
      check_equal_result = stoi(num_or_dim_3_list[0]);
    }
    else{
      out << "Input has to be either dimension or number.\n";
      return;
    }

    if(compare_result == check_equal_result){
      bool_id_map.insert(std::pair(id_map_it->second, true));
    }
    else{
      bool_id_map.insert(std::pair(id_map_it->second, false));
      removed_tid_list.push_back(id_map_it->second);
    }

  }
  delete_record_from_id_table(out, node, db, bool_id_map, tr_name);


  const std::string &file_path = "delete" + tr_name + ".sbbts";
  delete_tensor_from_cluster(out, node, removed_tid_list, file_path, relation_exist_flag);
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
      out << "Relation \'" << table << "\' does not exist.\n";
      return;
    }
    // sqlite3_step(stmt); // executing the statement
    while((rc = sqlite3_step(stmt)) == SQLITE_ROW){
      if(table.compare("KERNEL_FUNC") == 0){
        std::string string_name = std::string(reinterpret_cast< const char* >(sqlite3_column_text(stmt, 0)));
        bbts::tid_t id = sqlite3_column_int(stmt, 1);
        std::string is_ass = std::string(reinterpret_cast< const char* >(sqlite3_column_text(stmt, 2)));
        std::string is_comm = std::string(reinterpret_cast< const char* >(sqlite3_column_text(stmt, 3)));
        bbts::tid_t num_in = sqlite3_column_int(stmt, 4);
        bbts::tid_t num_out = sqlite3_column_int(stmt, 5);
        out << "kernel_name: " << string_name << ",   kernel_id: " << id << ",   is_ass: " <<  is_ass << 
            ",   is_comm: " << is_comm << ",   num_in: " << num_in << ",   num_out: " << num_out << "\n";
      }
      else{
        bbts::tid_t id = sqlite3_column_int(stmt, 0);
        std::string string_name = std::string(reinterpret_cast< const char* >(sqlite3_column_text(stmt, 1)));

        out << "tos_id: " << id << ", tra_id: " << string_name << "\n";
      }
    }
  }
  else{
    std::cout << "Failed to open db\n";
  }

  sqlite3_close(sql_db);
}

void export_data_to_file(std::ostream &out, bbts::node_t &node, const std::string &db, const std::string &tr_name, const std::string &file_path){
  
  bool relation_exist_flag = true;
  std::map<std::string, bbts::tid_t> id_map = generate_id_map_for_all_current_node_from_sqlite(out, node, db, tr_name, relation_exist_flag);
  std::map<std::string, bbts::tid_t>::iterator it;
  
  if(!relation_exist_flag){
    return; 
  }
  for(it = id_map.begin(); it != id_map.end(); it++){
    auto [success, message] = node.print_tensor_info(static_cast<bbts::tid_t>(it->second));
    if(!success) {
      out << bbts::red << "[ERROR]\n";
    }
    std::ofstream data_file;
    data_file.open(file_path);
    message.erase(0,26);
    data_file << message;
  }
}

void drop_table(std::ostream &out, bbts::node_t &node, const std::string &table_name, const std::string &db, bool &relation_exist_flag){

  const char* db_char= db.c_str();
  sqlite3_stmt * stmt;
  sqlite3 *sql_db;

  std::string  sql1 = "DROP TABLE " + table_name;
  const char* sql_drop = sql1.c_str();

  std::string sql2 = "SELECT * FROM " + table_name + ";\n";
  const char* sql_select = sql2.c_str();

  if(sqlite3_open(db_char, &sql_db) == SQLITE_OK){
    // preparing the statement
    int rc = sqlite3_prepare_v2(sql_db, sql_select,-1, &stmt, NULL);

    if(rc != SQLITE_OK){
      out << "Relation \'" << table_name << "\' does not exist.\n";
      relation_exist_flag = false;
      return;
    }
  }
  else{
    std::cout << "Failed to open db\n";
  }
  sqlite3_finalize(stmt);

  execute_command(sql_drop, db_char, 0);
  sqlite3_close(sql_db);
  // out << "\n\n\n\n";
}

void display_all_table(std::ostream &out, bbts::node_t &node, const std::string &db){

  const char* db_name = db.c_str();
  std::string  sql = "SELECT * FROM sqlite_master where type='table';";
  const char* sql_char = sql.c_str();
  sqlite3_stmt * stmt;
  sqlite3 *sql_db;

  if(sqlite3_open(db_name, &sql_db) == SQLITE_OK){
     // preparing the statement
    int rc = sqlite3_prepare_v2(sql_db, sql_char,-1, &stmt, NULL);

    // sqlite3_step(stmt); // executing the statement
    out<< "\n Current tables in tra.db are the following: \n";
    while((rc = sqlite3_step(stmt)) == SQLITE_ROW){
      std::string table_info = std::string(reinterpret_cast< const char* >(sqlite3_column_text(stmt, 1)));
      out << table_info << "\n";
    }
    out << "\n";
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

  
  },"Load a data text file. Usage : load text_file <file>\n");

  loadSubMenu->Insert("library", [&](std::ostream &out, const std::string &shared_lib_file) {
    const std::string &db = "tra.db";
    load_shared_library(out, node, shared_lib_file);
    create_udf_table(out, node, db);  
    out << "Successfully load shared library\n";


  
  },"Load a shared object file with a file holding all kernel functions. Usage : load library <file>\n");

  loadSubMenu->Insert("binary_file", [&](std::ostream &out, const std::string &file) {

    load_binary_file(out, node, file);  
  
  },"Load a binary file. Usage : load binary_file <file>\n");

  loadSubMenu->Insert("tensors",[&](std::ostream &out, const std::string &file_list) {

    load_tensors(out, node, file_list);

  },"Load tensors from filelist. Usage : load tensors <path_to_file_list>\n");


  rootMenu->Insert(std::move(loadSubMenu));





  // set up generate
  auto generateSubMenu = std::make_unique<Menu>("generate");


  generateSubMenu->Insert("matrix_to_binary", [&](std::ostream &out, const unsigned num_row, const unsigned num_col, const std::string &file) {

    generate_binary_file(num_row, num_col, file);  
    // load_binary_file(out, node, file);
  
  },"Generate matrix with specified rows and cols in binary file. Usage : generate matrix_to_binary <num_row> <num_col> <file>\n");


  rootMenu ->Insert("createTR",[&](std::ostream &out, const std::string from_clause, const std::string &file, const std::string split_row_by_clause, const int32_t row_split, const std::string split_col_by_clause, const int32_t col_split, const std::string tensor_format_clause, const std::string &tensor_type, const std::string arrow_clause, const std::string &tr_name) {
    const std::string &db = "tra.db";

    create_tensors(out, node, file, row_split, col_split, tensor_type, tr_name, db);
  
  },"(1) Create tensors based on the binary data file. (2)Generate data file format for data loader (3) Save relation into sqlite. Usage : generate tensors from <file> split_row_by <row_split> split_col_by <col_split> tensor_format <tensor_type> -> <Tensor Relation Name>\n");

  
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
    
  
  },"Generate commands for generating matrix. Usage : generate_matrix_commands <num_rows> <num_cols> <row_split> <col_split> <kernel_func> <file_path.sbbts> <db_name>\n");




  /*************************************   aggregate  **********************************************/
  rootMenu->Insert("aggregate",[&](std::ostream &out, const std::string &tr_name, std::string on_clause, const std::string &dimensions, 
                                  std::string using_clause, const std::string &kernel_name, std::string arrow_clause, const std::string &stored_tr_name) {
    

    const std::string db = "tra.db";
    const std::string file_path = stored_tr_name + ".sbbts";
    // the functions

    std::map<std::string, bbts::tid_t> output_mapping;
    std::map<std::string, std::vector<bbts::tid_t>> input_mapping;
    std::vector<std::string> dimension_list = split(dimensions, ",");
    bool relation_exist_flag = true;

    generate_aggregation_tra(out, node, output_mapping, input_mapping, db, dimension_list, tr_name, stored_tr_name, relation_exist_flag);
    generate_tra_op_commands(out, node, kernel_name, output_mapping, input_mapping, file_path, db, relation_exist_flag);


    



    std::map<std::string, bbts::tid_t>::iterator output_it;
    for (output_it = output_mapping.begin(); output_it != output_mapping.end(); output_it++){
      
      auto [success, message] = node.print_tensor_info(static_cast<bbts::tid_t>(output_it->second));
      if(!success) {
        out << bbts::red << "[ERROR]\n";
      }
      out << message << '\n';
      
    }
  
  },"Generate and run commands for aggregation. Usage : aggregate <TR Name> on <dimensions> using <kernel_name> -> <stored TR name>\n");

  rootMenu->Insert("materialize",[&](std::ostream &out, const std::string &tr_name) {

    const std::string &file_path = tr_name + ".sbbts";
    const std::string &db = "tra.db";
    const std::string &delete_file_path = "delete" + tr_name + ".sbbts";
    bool relation_exist_flag = true;

    materialize_commands(out, node, file_path, tr_name, db);
    // delete_tensor_from_cluster(out, node, std::get<1>(stored_R.find(tr_name)->second), delete_file_path, relation_exist_flag);

    // std::vector<std::string> input_tr_name_list = std::get<2>(stored_R.find(tr_name)->second);
    // for(auto input_tr_name: input_tr_name_list){
    //   bool relation_exist_flag = true;
    //   stored_R.erase(input_tr_name);
    //   drop_table(out, node, input_tr_name, db, relation_exist_flag);
    // }
    
  },"Materialize commands. Usage : materialize <TR name>\n");

  /*************************************   join  **********************************************/
  rootMenu->Insert("join",[&](std::ostream &out, const std::string &tr_name_l, const std::string &tr_name_r, std::string on_clause, 
                              std::string joinKeysL, std::string equal_clause, std::string joinKeysR, 
                              std::string using_clause, const std::string &kernel_name, std::string arrow_clause, 
                              const std::string &stored_tr_name) {
    
    const std::string &file_path = stored_tr_name + ".sbbts";
    const std::string &db = "tra.db";
  

    std::map<std::string, bbts::tid_t> output_mapping;
    std::map<std::string, std::vector<bbts::tid_t>> input_mapping;
    std::vector<std::string> dimension_list_l = split(joinKeysL, ",");
    std::vector<std::string> dimension_list_r = split(joinKeysR, ",");
    bool relation_exist_flag = true;

    generate_join_tra(out, node, output_mapping, input_mapping, db, dimension_list_l, dimension_list_r, tr_name_l, tr_name_r, stored_tr_name, relation_exist_flag);

    

    generate_tra_op_commands(out, node, kernel_name, output_mapping, input_mapping, file_path, db, relation_exist_flag);

    // std::map<std::string, bbts::tid_t>::iterator output_it;
    // for (output_it = output_mapping.begin(); output_it != output_mapping.end(); output_it++){
      
    //   auto [success, message] = node.print_tensor_info(static_cast<bbts::tid_t>(output_it->second));
    //   if(!success) {
    //     out << bbts::red << "[ERROR]\n";
    //   }
    //   out << message << '\n';
      
    // }

  
  },"Generate and run commands for join. Usage : join <left_tr_name> <right_tr_name> on <left dimension list> = <right dimension list> using <kernel_name> -> <stored tr name>\n");

  /*********************************************** reKey ******************************************/
  auto rekeySubMenu = std::make_unique<Menu>("rekey");

  rekeySubMenu->Insert("math",[&](std::ostream &out, const std::string &tr_name, const std::string &what, 
                              const std::string &num_or_dim_1, std::string by_or_from_or_to, const std::string &num_or_dim_2) {
   const std::string &db = "tra.db";

   if(math_op.find(what) != math_op.end()){
     if(by_or_from_or_to == "by"){
       handle_math_op_for_rekey(out, node, tr_name, what, num_or_dim_1, num_or_dim_2, db);
     }
     else if (by_or_from_or_to == "from"){
       handle_math_op_for_rekey(out, node, tr_name, what, num_or_dim_2, num_or_dim_1, db);
     }
     else{
       out << "Please use math_op #/<dim>/all by/from #/<dim>/all for mathematical operations.\n";
     }
     
   } 
   else if(collective_op.find(what) != collective_op.end()){
     if(by_or_from_or_to == "to"){
       handle_collective_op_for_rekey(out, node, tr_name, what, num_or_dim_1, num_or_dim_2, db);
     }
     else{
       out<< "Please use collective_op <dim>/all to <dim> for collective operations.\n";
     }
   }

  //  reKey(out, node, db.c_str(), &keyFunc, tr_name, stored_tr_name, kernel_name);
  
  },"Perform reKey. Usage : (1) rekey <TR name> <add(+)/subtract(-)/multiply(*)/divide(/)> <dimensions/#> from/by <dimensions/#>.(2) rekey <TR name> <sum> <dimensions> to <output dimension>\n");

  rekeySubMenu->Insert("drop",[&](std::ostream &out, const std::string &tr_name, const std::string &drop_dimensions){

    const std::string &db = "tra.db";
    drop_dimensions_from_relation(out, node, tr_name, drop_dimensions, db);
  },"Drop dimensions to form new key. Usage: rekey <TR name> drop <input dimensions> to <output dimension>.");

  rekeySubMenu->Insert("append",[&](std::ostream &out, const std::string &tr_name, const std::string &append_dimension, 
                                    std::string from_clause, const std::string &num_or_dim){
                                      
    const std::string &db = "tra.db";
    const std::string &op = "add";
    const std::string &num_or_dim_2 = "0";
    append_dimension_to_relation(out, node, tr_name, append_dimension, num_or_dim, op, num_or_dim_2, db);

  },"Usage: append <TR name> <dimensions> from <#/dim>.");

  rekeySubMenu->Insert("append",[&](std::ostream &out, const std::string &tr_name, const std::string &append_dimension, 
                                    std::string from_clause, const std::string &num_or_dim_1, const std::string &op, const std::string &num_or_dim_2){
    
    const std::string &db = "tra.db";
    append_dimension_to_relation(out, node, tr_name, append_dimension, num_or_dim_1, op, num_or_dim_2, db);
  },"Usage: append <TR name> <dimension> from <#/dim> <add(+)/subtract(-)/multiply(*)/divide(/)> <dimensions/#>");

  rootMenu->Insert(std::move(rekeySubMenu));

  /*********************************************** filter ******************************************/
  auto filterSubMenu = std::make_unique<Menu>("filter");
  filterSubMenu->Insert("by",[&](std::ostream &out, const std::string &tr_name, const std::string &num_or_dim_1, std::string equal_clause, 
                                const std::string &num_or_dim_2) {
   
    const std::string &db = "tra.db";
    filter_relations(out, node, tr_name, num_or_dim_1, "+", "0", num_or_dim_2, db);
  
  },"Perform filter. Usage : filter by <TR name> <#/dim> = <#/dim>\n");

  filterSubMenu->Insert("by",[&](std::ostream &out, const std::string &tr_name, const std::string &num_or_dim_1, const std::string &op, 
                                const std::string &num_or_dim_2, std::string equal_clause, const std::string &num_or_dim_3) {
   
    const std::string &db = "tra.db";
    filter_relations(out, node, tr_name, num_or_dim_1, op, num_or_dim_2, num_or_dim_3, db);
  
  },"Perform filter. Usage : filter by <TR name> <#/dim> <add(+)/subtract(-)/multiply(*)/divide(/)> <#/dim> = <#/dim>\n");

  filterSubMenu->Insert("by",[&](std::ostream &out, const std::string &tr_name, const std::string &num_or_dim_1, std::string equal_clause, 
                                const std::string &num_or_dim_2, const std::string &op, const std::string &num_or_dim_3) {
   
    const std::string &db = "tra.db";
    filter_relations(out, node, tr_name, num_or_dim_2, op, num_or_dim_3, num_or_dim_1, db);
   
  
  },"Perform filter. Usage : filter by <TR name> <#/dim> = <#/dim> <add(+)/subtract(-)/multiply(*)/divide(/)> <#/dim>\n");

  rootMenu->Insert(std::move(filterSubMenu));

  /*********************************************** transform ******************************************/
  rootMenu->Insert("transform",[&](std::ostream &out, const std::string &tr_name, std::string using_clause, 
                                  const std::string &kernel_name, std::string arrow_clause, const std::string &stored_tr_name) {
    

    const std::string &file_path = stored_tr_name + ".sbbts";
    const std::string &db = "tra.db";


    std::map<std::string, bbts::tid_t> output_mapping;
    std::map<std::string, std::vector<bbts::tid_t>> input_mapping;
    std::vector<bbts::tid_t> removed_tid_list;
    bool relation_exist_flag = true;

    generate_transform_tra(out, node, output_mapping, input_mapping, db, stored_tr_name, removed_tid_list, tr_name, relation_exist_flag);
    generate_tra_op_commands(out, node, kernel_name, output_mapping, input_mapping, file_path, db, relation_exist_flag);




    // auto out_it = output_mapping.begin();
    // auto in_it = input_mapping.begin();

    // for(int i = 0; i < output_mapping.size(); i++){
    //   out << in_it->first << " | ";
    //   for(int j = 0; j < in_it->second.size(); j++){
    //     out << in_it->second[j] << " ";
    //   }
    //   out << "| " << out_it->second << "\n";
    //   in_it++;
    //   out_it++;
    // }

    

    std::map<std::string, bbts::tid_t>::iterator output_it;
    for (output_it = output_mapping.begin(); output_it != output_mapping.end(); output_it++){
      
      auto [success, message] = node.print_tensor_info(static_cast<bbts::tid_t>(output_it->second));
      if(!success) {
        out << bbts::red << "[ERROR]\n";
      }
      out << message << '\n';
      
    }

  
  },"Generate and run commands for transform. Usage : transform <TR name> using <kernel func> -> <stored TR name>\n");

  /*********************************************** tile ******************************************/


  /*********************************************** concat ******************************************/

  rootMenu->Insert("execute",[&](std::ostream &out, const std::string &file_path) {

    compile_commands(out, node, file_path);
    run_commands(out, node);

  },"Execute command file. Usage: execute <.sbbts>\n");

  rootMenu->Insert("export",[&](std::ostream &out, const std::string &tr_name, std::string into_clause, const std::string &file_path) {
    const std::string &db = "tra.db";
    export_data_to_file(out, node, db, tr_name, file_path);

  },"Export Tensor Relation into npy file. Usage: export <TR name> into <.npy>\n");

  rootMenu->Insert("clear",[&](std::ostream &out) {

    clear(out, node);

  },"Clears the tensor operating system.\n");

  

  rootMenu->Insert("read_table",[&](std::ostream &out, const std::string &table) {
  const std::string &db = "tra.db";
  read_table(out, db, table);
  
  }, "Read table in sqlite; Usage read_table <db_name> <table_name>");

  rootMenu->Insert("display",[&](std::ostream &out, const std::string &tr_name) {
    
    read_table(out, "tra.db", tr_name);

  }, "Display id pairs of Tensor Relation; Usage display <TR name>");

  rootMenu->Insert("delete",[&](std::ostream &out, const std::string &tr_name){
    const std::string &db = "tra.db";
    std::vector<bbts::tid_t> id_list;
    const std::string &file_path = "delete" + tr_name + ".sbbts";
    bool relation_exist_flag = true;
    std::map<std::string, bbts::tid_t> id_map = get_id_map_from_sqlite(out, node, db.c_str(), 0, tr_name, relation_exist_flag);
    std::map<std::string, bbts::tid_t>::iterator map_it;
   

    for(map_it = id_map.begin(); map_it != id_map.end(); map_it++){
      id_list.push_back(map_it->second);
    }

    drop_table(out, node, tr_name, db, relation_exist_flag);
    delete_tensor_from_cluster(out, node, id_list, file_path, relation_exist_flag);
  }, "Display id pairs of Tensor Relation; Usage display <TR name>");

  rootMenu->Insert("tables",[&](std::ostream &out){
    const std::string &db = "tra.db";
    display_all_table(out, node, db);
  },"Display all tables in tra.db.");

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
