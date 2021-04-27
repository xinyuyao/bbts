#pragma once

#include "../server/coordinator.h"
#include "../../third_party/http-server/httplib.h"
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

    // form the json
    std::stringstream ss;
    ss << "[";
    for(node_id_t node = 0; node < profiling_nfo.size(); ++node) {

      // form the json object
      for(auto &nfo : profiling_nfo[node]) {
        ss << "{";
        ss << "\"id\" : " << nfo.cmd_id << ", ";
        ss << "\"event\" : " << "\"" << nfo.event_to_string() << "\", ";
        ss << "\"timestamp\" : " << nfo.ts << ", ";
        ss << "\"node_id\" : " << node << ", ";
        ss << "\"thread_id\" : " << nfo.thread_id;
        ss << "},";
      }
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

    res.set_content(ss.str(), "text/plain");
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