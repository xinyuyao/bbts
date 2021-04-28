#pragma once

#include "../server/coordinator.h"
#include "../../third_party/http-server/httplib.h"
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>

namespace bbts {



class web_server_t {
public:

// creates the server
web_server_t(coordinator_ptr_t _coordinator) : _coordinator(std::move(_coordinator)) {
  init_handlers();
}


struct profile_log_response_t {

  struct log_entry_t {

    command_id_t cmd_id;
    long start;
    long end;
  };  

  std::vector<log_entry_t> commands;
  std::vector<log_entry_t> storage;
  std::vector<log_entry_t> kernel;
  std::vector<log_entry_t> send;
  std::vector<log_entry_t> recv;
};

// run the server
void run() {
  svr.listen("localhost", 8081);
}

// shut it down
void shutdown() {
  svr.stop();
}

private:

void init_handlers() {

if (!svr.set_mount_point("/", "./www")) {
  throw std::runtime_error("");
}

svr.Get("/api/profiling", [&](const httplib::Request &, httplib::Response &res) {

  auto profiling_nfo = _coordinator->get_all_profiling_nfo();

  std::stringstream ss;
  ss << "[";
  for(auto &nfo : profiling_nfo) {
    
    // make the json
    ss << "{";
    ss << "\"id\" : " << nfo.id << ", ";
    ss << "\"start\" : " << nfo.start << ", ";
    ss << "\"end\" : " << nfo.end;
    ss << "},";
  }
  if(!profiling_nfo.empty()) {
    ss.seekp(-1, std::ios_base::end);
  }
  ss << "]";

  res.set_content(ss.str(), "text/json");
});

svr.Get(R"(/api/profile/(\d+))", [&](const httplib::Request &req, httplib::Response &res) {

  try {

    // try to parse the index from here
    char *end;
    auto id = std::strtol(req.matches[1].str().c_str(), &end, 10);

    // try to parse it
    auto profiling_nfo = _coordinator->get_profiling_nfo_for(id);
    if(profiling_nfo.empty()) {
      throw std::runtime_error("Could not find the profiling info for " + std::to_string(id) + ".");
    }

    std::map<std::tuple<int32_t, int32_t>, profile_log_response_t> processed;
    for(node_id_t node = 0; node < profiling_nfo.size(); ++node) {

      // form the json object
      for(auto &nfo : profiling_nfo[node]) {
        
        // get the 
        auto &s = processed[{node, nfo.thread_id}];

        switch (nfo.event) {
          
          // log the start event
          case command_profiler_t::event_t::START : s.commands.push_back(profile_log_response_t::log_entry_t {
                                                                           .cmd_id = nfo.cmd_id, 
                                                                           .start = nfo.ts, 
                                                                           .end = -1
                                                                         }); break;
          // log the end event
          case command_profiler_t::event_t::END : s.commands.back().end = nfo.ts; break;

          // log the storage start
          case command_profiler_t::event_t::STORAGE_OP_START : s.storage.push_back(profile_log_response_t::log_entry_t {
                                                                           .cmd_id = nfo.cmd_id, 
                                                                           .start = nfo.ts, 
                                                                           .end = -1
                                                                         }); break;
          // log the end event
          case command_profiler_t::event_t::STORAGE_OP_END : s.storage.back().end = nfo.ts; break;

          // log the start of a kernel op
          case command_profiler_t::event_t::KERNEL_START : s.kernel.push_back(profile_log_response_t::log_entry_t {
                                                                           .cmd_id = nfo.cmd_id, 
                                                                           .start = nfo.ts, 
                                                                           .end = -1
                                                                         }); break;
          // log the end event
          case command_profiler_t::event_t::KERNEL_END : s.kernel.back().end = nfo.ts; break;

          // log the start of send
          case command_profiler_t::event_t::SEND : s.send.push_back(profile_log_response_t::log_entry_t {
                                                                           .cmd_id = nfo.cmd_id, 
                                                                           .start = nfo.ts, 
                                                                           .end = -1
                                                                         }); break;
          // log the end event
          case command_profiler_t::event_t::SEND_END : s.send.back().end = nfo.ts; break;

          // log the recieve
          case command_profiler_t::event_t::RECV : s.recv.push_back(profile_log_response_t::log_entry_t {
                                                                           .cmd_id = nfo.cmd_id, 
                                                                           .start = nfo.ts, 
                                                                           .end = -1
                                                                         }); break;
          // log the end of recieve
          case command_profiler_t::event_t::RECV_END : s.recv.back().end = nfo.ts; break;
        }
      }
    }

    // form the json
    std::stringstream ss;
    ss << "[";

    // write out the commands 
    for(auto &entry : processed) {

      ss << "{ \"name\": " << "\"Commands\",";
      ss << "\"data\": [";

      for(auto &s : entry.second.kernel) {

        ss << "{";
        ss << "\"x\": \"" << std::get<0>(entry.first) << " thread : " << std::get<1>(entry.first) << "\", ",
        ss << "\"y\": [";
        ss << s.start / 1000000 << ", ";
        ss << s.end / 1000000;
        ss << "]";
        ss << "},";
      }

      if(!entry.second.kernel.empty()) {
        ss.seekp(-1, std::ios_base::end);
      }

      ss << "]";
      ss << "},";
    }

    // storage 
    for(auto &entry : processed) {

      ss << "{ \"name\": " << "\"Storage\",";
      ss << "\"data\": [";

      for(auto &s : entry.second.storage) {

        ss << "{";
        ss << "\"x\": \"" << std::get<0>(entry.first) << " thread : " << std::get<1>(entry.first) << "\", ",
        ss << "\"y\": [";
        ss << s.start / 1000000 << ", ";
        ss << s.end / 1000000;
        ss << "]";
        ss << "},";
      }

      if(!entry.second.storage.empty()) {
        ss.seekp(-1, std::ios_base::end);
      }

      ss << "]";
      ss << "},";
    }

    // kernel
    for(auto &entry : processed) {

      ss << "{ \"name\": " << "\"Kernel\",";
      ss << "\"data\": [";

      for(auto &s : entry.second.kernel) {

        ss << "{";
        ss << "\"x\": \"" << std::get<0>(entry.first) << " thread : " << std::get<1>(entry.first) << "\", ",
        ss << "\"y\": [";
        ss << s.start / 1000000 << ", ";
        ss << s.end / 1000000;
        ss << "]";
        ss << "},";
      }

      if(!entry.second.kernel.empty()) {
        ss.seekp(-1, std::ios_base::end);
      }

      ss << "]";
      ss << "},";
    }

    for(auto &entry : processed) {

      ss << "{ \"name\": " << "\"Send\",";
      ss << "\"data\": [";

      for(auto &s : entry.second.send) {

        ss << "{";
        ss << "\"x\": \"" << std::get<0>(entry.first) << " thread : " << std::get<1>(entry.first) << "\", ",
        ss << "\"y\": [";
        ss << s.start / 1000000 << ", ";
        ss << s.end / 1000000;
        ss << "]";
        ss << "},";
      }

      if(!entry.second.send.empty()) {
        ss.seekp(-1, std::ios_base::end);
      }

      ss << "]";
      ss << "},";
    }

    // recieve
    for(auto &entry : processed) {

      ss << "{ \"name\": " << "\"Recv\",";
      ss << "\"data\": [";

      for(auto &s : entry.second.recv) {

        ss << "{";
        ss << "\"x\": \"" << std::get<0>(entry.first) << " thread : " << std::get<1>(entry.first) << "\", ",
        ss << "\"y\": [";
        ss << s.start / 1000000 << ", ";
        ss << s.end / 1000000;
        ss << "]";
        ss << "},";
      }

      if(!entry.second.recv.empty()) {
        ss.seekp(-1, std::ios_base::end);
      }

      ss << "]";
      ss << "},";
    }

    ss.seekp(-1, std::ios_base::end);
    ss << "]";

    // return the json
    res.set_content(ss.str(), "text/json");
  }
  catch(const std::runtime_error &e) {

    std::stringstream ss;
    ss << "{";
    ss << "\"error\" : " << "\"" << e.what() << "\"";
    ss << "}"; 

    res.set_content(ss.str(), "text/json");
    res.status = 404;
  }
});

}

// the command profiler
coordinator_ptr_t _coordinator;

// the server
httplib::Server svr;


};

using web_server_ptr_t = std::shared_ptr<web_server_t>; 

}