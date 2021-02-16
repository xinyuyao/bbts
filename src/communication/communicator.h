#pragma once

#include <mpi.h>
#include <cstdint>
#include <unordered_map>
#include <iostream>

#include "../tensor/tensor_factory.h"
#include "../server/node_config.h"
#include "../ud_functions/ud_function.h"
#include "../tensor/tensor.h"
#include "../commands/command.h"
#include "../server/coordinator_ops.h"

namespace bbts {

// identifier for any node
const node_id_t ANY_NODE = MPI_ANY_SOURCE;
 
// defines some of the most common message request
using com_tags = int32_t;

// matches to any tag
const com_tags ANY_TAG = MPI_ANY_TAG;

// the request to a node to send the sending node the tensors it wants
const com_tags SEND_CMD_TAG = 1;
const com_tags SHUTDOWN_TAG = 2;
const com_tags COORDINATOR_TAG = 3;
const com_tags COORDINATOR_BCAST_CMD_TAG = 4;
const com_tags NOTIFY_TENSOR_TAG = 5;

// this is a special tag that is the first free tag
// it is appended to every and receive send call
const com_tags FREE_TAG = 6;

// the mpi communicator
class mpi_communicator_t {
public:

  explicit mpi_communicator_t(const node_config_ptr_t &_cfg);

  ~mpi_communicator_t();

  // this is the request to fetch a certain number of tensors
  struct sync_request_t {

    // the type of the message
    com_tags message_tag{};
    
    // the number of bytes
    int32_t num_bytes{};

    // the status of the message
    MPI_Status status{};

    // the message identifier
    MPI_Message message{};

    // the success 
    bool success = true;
  };

  struct async_request_t {

    // the message request identifier
    MPI_Request request{};

    // the success 
    bool success = true;
  };

  // recives a blob with the matching tag from a given node, method blocks
  bool recv_sync(void *_bytes, size_t num_bytes, node_id_t _node, com_tags _tag);

  // does the send, method is blocking
  bool send_sync(const void *_bytes, size_t num_bytes, node_id_t _node, com_tags _tag);

  // wait for async request
  bool wait_async(async_request_t &_request);

  // notify a node that tensors were created
  bool tensors_created_notification(node_id_t out_node, const std::vector<bbts::tid_t> &tensor);

  // wait to receive a notification
  std::tuple<node_id_t, std::vector<bbts::tid_t>> receive_tensor_created_notification();

  // shutdown the notification handler
  bool shutdown_notification_handler();

  // send async
  async_request_t send_async(const void *_bytes, size_t num_bytes, node_id_t _node, com_tags _tag);

  // waits for a request to be available from a particular node
  sync_request_t expect_request_sync(node_id_t _node, com_tags _tag);

  // recieves the request that we got from expect_request_sync
  bool receive_request_sync(void *_bytes, sync_request_t &_req);

  // initiates the operation on all the specified nodes
  bool op_request(const command_ptr_t &_cmd);

  // waits to recieve an operation
  command_ptr_t expect_op_request();

  // this sends a shutdown command to the thread that is calling @see expect_op_request
  bool shutdown_op_request();

  // send the coord op to all nodes
  bool send_coord_op(const bbts::coordinator_op_t &op);

  // expect the a coord op
  bbts::coordinator_op_t expect_coord_op();

  // send the cmds to all nodes
  bool send_coord_cmds(const std::vector<command_ptr_t> &cmds);

  // expect the a coord op
  bool expect_coord_cmds(size_t num_cmds, std::vector<command_ptr_t> &out);

  // waits for all the nodes to hit this, should only be used for initialization
  void barrier();

  // sync the resource aquisition
  bool sync_resource_aquisition(command_id_t cmd,
                                const std::vector<node_id_t> &nodes, 
                                bool my_val);

  // sync the resource aquisition between two nodes
  bool sync_resource_aquisition_p2p(command_id_t cmd, node_id_t &node, bool my_val);

  // return the rank
  [[nodiscard]] int32_t get_rank() const;

  // return the number of nodes
  [[nodiscard]] int32_t get_num_nodes() const;

 private:

  // operations communicator
  MPI_Comm _op_com;

  // the world commmunicator, we use this one for initiating requests
  MPI_Comm _world_com;

  // the rank of my node
  int32_t _rank;

  // the number of nodes in the cluster
  int32_t _num_nodes;

  // we need this to get hte tensor size
  tensor_factory_ptr_t _tensor_factory;
};

// the default communicator is the mpi communicator
using communicator_t = mpi_communicator_t;
using communicator_ptr_t = std::shared_ptr<mpi_communicator_t>;

}