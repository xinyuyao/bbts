#pragma once

#include <memory>

namespace bbts {

// this structure information on how the node is configured
struct node_config_t {
    
    // the number of arguments passed to the node
    int argc;

    // the arguments as string pointers
    char **argv;
};

// a nice way to reference the configuration ptr
using node_config_ptr_t = std::shared_ptr<node_config_t>;

class node_t {
public:


};


}