#include <cstdint>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#include <tuple>
#include <memory>
#include "../tensor/tensor.h"
#include "../commands/commands.h"
#include "../storage/storage.h"

namespace bbts {

// here we queue all the commands this node needs to execute that we need 
// to execute in conjunction with other nodes, are kept in the external_commands_queue_t
class reservation_station_t {
 public:

  reservation_station_t(node_id_t _node_id, storage_ptr_t _storage) : _my_rank(_node_id), _storage(std::move(_storage)) {}

  // queue a command, this command has to be executed in the same thread and the commands
  // have to be queued in the exact order they are coming in
  bool queue_command(command_ptr_t _command) {

    // lock the reservation station
    std::unique_lock<std::mutex> lk(_m);

    if(_command->get_root_node() == _my_rank) {
      return _queue_local(std::move(_command));
    }
    else {
      return _queue_remote(std::move(_command));
    }
  }

  // mark that a command is processed
  bool retire_command(command_ptr_t _command) {

    // lock the reservation station
    std::unique_lock<std::mutex> lk(_m);

    // check if this is a remote node
    if(_command->get_root_node() == _my_rank) {
      return _retire_command(std::move(_command));
    }
    else {
      return _retire_remote_command(std::move(_command));
    }
  }

  // a list of node tensor for each node that needs to be notified that a tensor is available.
  // We get that information by looking at the input tensors of remote commands scheduled here
  [[nodiscard]] std::vector<std::vector<tid_t>> tensors_available_notification() {

    // lock the tensor
    std::unique_lock<std::mutex> lk(_m);

    return {};
  }

  // notify the reservation station that the tensor on an another node became available
  void notify_available_tensors(node_id_t node, const std::vector<tid_t> &tensors) {

    // lock the tensor
    std::unique_lock<std::mutex> lk(_m);

  }

  // get the next command, you must use the result of this command as it is a unique ptr
  [[nodiscard]] command_ptr_t get_next_command() {

    // wait until we have something here
    std::unique_lock<std::mutex> lk(_m);
    _cv.wait(lk, [&]{return !_execute.empty();});

    // pop the unique pointer of the vector
    auto tmp = std::move(_execute.back());
    _execute.pop_back();

    // return it
    return std::move(tmp);
  }

  // register the tensor that was added externally,
  // that is it was not created through the execution of a command
  void register_tensor(tid_t _tid) {

    // lock the tensor
    std::unique_lock<std::mutex> lk(_m);

  }

 private:

  bool _queue_remote(command_ptr_t _command) {

    // handle delete
    if(_command->is_delete()) {

    }

    // count the number of inputs that are not present
    int32_t num_not_present = 0;
    for(int32_t i = 0; i < _command->get_num_inputs(); i++) {

      // get the tensor required in the input
      auto _ts = _command->get_input(i);

      // check if this is a remote tensor or a local tensor
      if(_ts.node == _my_rank) {

        // if it is a local we need to check if it was created already
        auto &s = _tensors[_ts.tid];

        // you can not use a tensors scheduled for deletion
        assert(!s.scheduled_for_delition);

        // if it was not created we need to keep track of that
        if(!s.is_created) {

          // tensor is not present
          num_not_present++;

          // mark that this command is waiting
          _commands_waiting_for[_ts.tid] = _command->id;
        }
      }
      else {

        // ok this is a remote tensor, we need to check if it is present
        auto &rts = _remote_tensors[_ts.node];
        auto it = rts.find(_ts.tid);

        // if the tensor is not present or we don't have the most recent info about it
        // (the command that requested the status is smaller than the id of this command)
        if(it == rts.end() && it->second < _command->id) {


        }
      }
    }

    for(int32_t i = 0; i < _command->get_num_outputs(); i++) {

    }

    return false;
  }

  bool _queue_local(command_ptr_t _command) {

    // we are done here
    return true;
  }

  // retire the local command
  bool _retire_command(command_ptr_t _command) {

    return true;
  }

  // retire remote the command
  bool _retire_remote_command(command_ptr_t _command) {

    return true;
  }

  void _schedule_for_excution(command_ptr_t _cmd) {

    // schedule the command for execution
    _execute.emplace_back(std::move(_cmd));
    _cv.notify_all();
  }

  // remove the tensor
  void _remove_tensor(tid_t _tid) {

  }

  // the state of the tensor
  struct internal_tensor_state_t {

    // the number of commands to read this, includes both the remote and local commands
    int32_t num_to_read = 0;

    // the number of commands to write this
    int32_t writing_tensor = false;

    // is the tensor created
    bool is_created = false;

    // is this tensor scheduled for delition
    bool scheduled_for_delition = false;
  };

  // the mutex
  std::mutex _m;

  // we use this to wait for commands
  std::condition_variable _cv;

  // the node for which this reservation station is for
  node_id_t _my_rank;

  // commands ready to execute
  std::vector<command_ptr_t> _execute;

  // the local commands and the number of tensors they are waiting for
  std::unordered_map<command_id_t, std::pair<command_ptr_t, int32_t>> _local_commands;

  // the remote commands
  std::unordered_map<command_id_t, command_ptr_t> _remote_commands;

  // the local tensors commands are waiting for
  std::unordered_multimap<tid_t, command_id_t> _commands_waiting_for;

  // keeps all the local tensors
  std::unordered_map<tid_t, internal_tensor_state_t> _tensors;

  // keeps the status of all the remote tensors. If the tensor is present we store the command, that requested
  // the status for it. All the commands with a lower or equal command id can be executed safely as the tensor can not
  // be removed unless they are executed.
  std::vector<std::unordered_map<tid_t, command_t>> _remote_tensors;


};

using reservation_station_ptr_t = std::shared_ptr<reservation_station_t>;

}