#include "communicator.h"

namespace bbts {

mpi_communicator_t::mpi_communicator_t(const node_config_ptr_t &_cfg) {

  // initialize the mpi
  int32_t provided;
  MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
  if (provided != MPI_THREAD_MULTIPLE) {
    throw std::runtime_error("MPI_THREAD_MULTIPLE not provided");
  }

  // get the number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &_num_nodes);

  // get the rank of the process
  MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
}

mpi_communicator_t::~mpi_communicator_t() {

  // shutdown the communicator
  MPI_Finalize();
}

bool mpi_communicator_t::recv_sync(void *_bytes, size_t num_bytes, node_id_t _node, com_tags _tag) {

  // recive the stuff
  return MPI_Recv(_bytes, num_bytes, MPI_CHAR, _node, _tag + FREE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE) == MPI_SUCCESS;
}

// does the send, method is blocking
bool mpi_communicator_t::send_sync(const void *_bytes, size_t num_bytes, node_id_t _node, com_tags _tag) {

  // get the number of byte to send and send the request
  return MPI_Ssend(_bytes, num_bytes, MPI_CHAR, _node, _tag + FREE_TAG, MPI_COMM_WORLD) == MPI_SUCCESS;
}

bool mpi_communicator_t::wait_async(mpi_communicator_t::async_request_t &_request) {
  return MPI_Wait(&_request.request, MPI_STATUSES_IGNORE) == MPI_SUCCESS;
}

mpi_communicator_t::async_request_t mpi_communicator_t::send_async(const void *_bytes, size_t num_bytes, node_id_t _node, com_tags _tag) {

  // initiate an asynchronous send request
  async_request_t _req;
  _req.success = MPI_Isend(_bytes, num_bytes, MPI_CHAR, _node, _tag + FREE_TAG, MPI_COMM_WORLD, &_req.request) == MPI_SUCCESS;

  // return the request handle
  return _req;
}

mpi_communicator_t::sync_request_t mpi_communicator_t::expect_request_sync(node_id_t _node, com_tags _tag) {

  // wait for a request
  sync_request_t _req;
  auto mpi_errno = MPI_Mprobe(_node, _tag  + FREE_TAG, MPI_COMM_WORLD, &_req.message, &_req.status);

  // check for errors
  if (mpi_errno != MPI_SUCCESS) {        
      _req.success = false;
      return _req;
  }

  // get the size
  MPI_Get_count(&_req.status, MPI_CHAR, &_req.num_bytes);

  // set the message type
  _req.message_tag = (com_tags) _req.status.MPI_TAG;

  // return the request
  return _req;
}

// recieves the request that we got from expect_request_sync
bool mpi_communicator_t::recieve_request_sync(void *_bytes, sync_request_t &_req) {

  // recieve the stuff
  return MPI_Mrecv (_bytes, _req.num_bytes, MPI_CHAR, &_req.message, &_req.status) == MPI_SUCCESS;
}

bool mpi_communicator_t::op_request(const command_ptr_t &cmd, node_id_t _node) {

  // send the command
  return MPI_Ssend(cmd.get(), cmd->num_bytes(), MPI_CHAR, _node, SEND_CMD_TAG, MPI_COMM_WORLD) == MPI_SUCCESS;
}

command_ptr_t mpi_communicator_t::listen_for_op_request() {

  // wait for a request
  sync_request_t _req;
  auto mpi_errno = MPI_Mprobe(ANY_NODE, SEND_CMD_TAG, MPI_COMM_WORLD, &_req.message, &_req.status);

  // check for errors
  if(mpi_errno != MPI_SUCCESS) {
    return nullptr;
  }

  // get the size  and set the tag for the request
  MPI_Get_count(&_req.status, MPI_CHAR, &_req.num_bytes);
  _req.message_tag = (com_tags) _req.status.MPI_TAG;

  // allocate the memory and receive the command
  std::unique_ptr<char[]> p(new char[_req.num_bytes]);
  if(MPI_Mrecv (p.get(), _req.num_bytes, MPI_CHAR, &_req.message, &_req.status) != MPI_SUCCESS) {
    return nullptr;
  }

  // cast it to the command
  auto p_rel = p.release();
  auto p_cmd = (bbts::command_t *)(p_rel);
  auto d = std::unique_ptr<bbts::command_t, command_deleter_t>(p_cmd);

  // move the command
  return std::move(d);
}

// waits for all the nodes to hit this, should only be used for initialization
void mpi_communicator_t::barrier() {

  // wait for every node
  MPI_Barrier(MPI_COMM_WORLD);
}

// return the rank
int32_t mpi_communicator_t::get_rank() const {
  return _rank;
}

// return the number of nodes
int32_t mpi_communicator_t::get_num_nodes() const {
  return _num_nodes;
}

}