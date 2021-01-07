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

  reservation_station_t(node_id_t _node_id, int32_t num_nodes) : _my_rank(_node_id),
                                                                 _num_nodes(num_nodes),
                                                                 _send_status_cv(num_nodes),
                                                                 _commands_waiting_for(num_nodes) {

    // make one of these for each node
    _remote_tensors.resize(num_nodes);
    _send_status_queue.resize(num_nodes);
  }

  // queue a command, this command has to be executed in the same thread and the commands
  // have to be queued in the exact order they are coming in
  bool queue_command(command_ptr_t _command) {

    // lock the reservation station
    std::unique_lock<std::mutex> lk(_m);

    // make sure the commands are scheduled in order
    auto cmd_id = _command->id;
    if(_last_cmd >= cmd_id) {
      return false;
    }

    // figure out whether this is a local command or a remote command (started by this machine or a remote machine)
    bool success;
    if(_command->get_root_node() == _my_rank) {
      success = _queue_local(std::move(_command));
    }
    else {
      success = _queue_remote(std::move(_command));
    }

    // if we succeeded we need to update the last command
    if(success) {
      _last_cmd = cmd_id;
    }

    // we are done get out of here
    return success;
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

  // a list of node tensors for a node that needs to be notified that the tensors are available.
  // We get that information by looking at the input tensors of remote commands scheduled here
  [[nodiscard]] std::vector<tid_t> tensors_to_notify_node(node_id_t node, bool &is_done) {

    // lock the tensor
    std::unique_lock<std::mutex> lk(_m);

    // wait until we are shutdown or we have something to notify for this node
    _send_status_cv[node].wait(lk, [&]{ return !_send_status_queue[node].empty() || _shutdown; });

    // if we have shutdown return
    if(_shutdown) {
      is_done = true;
      return {};
    }

    // get the tensors we need to send
    std::vector<tid_t> out;
    std::swap(out, _send_status_queue[node]);

    // return them
    is_done = false;
    return std::move(out);
  }

  // notify the reservation station that the tensor on an another node became available
  void notify_available_tensors(node_id_t node, const std::vector<tid_t> &tensors) {

    // lock the tensor
    std::unique_lock<std::mutex> lk(_m);

    // go through tensors
    for(auto t : tensors) {

      // go through all the commands that are waiting for this tensor
      auto cw = _commands_waiting_for[node].equal_range(t);
      for (auto it = cw.first; it != cw.second;) {

        // try to find the command
        auto jt = _local_commands.find(it->second);
        assert(jt != _local_commands.end());

        // check if we have all the inputs
        if(0 == (--jt->second.second)) {

          // schedule the command for execution
          _schedule_for_excution(std::move(jt->second.first));

          // remove the command
          _local_commands.erase(jt);
        }

        // remove the command from the waiting list
        it = _commands_waiting_for[node].erase(it);
      }

      // mark that the tensor is here
      _remote_tensors[node].insert(t);
    }
  }

  // get the next command, you must use the result of this command as it is a unique ptr
  [[nodiscard]] command_ptr_t get_next_command() {

    // wait until we have something here
    std::unique_lock<std::mutex> lk(_m);
    _cv.wait(lk, [&]{ return !_execute.empty()  || _shutdown; });

    // if we have shutdown return null as we have no command left...
    if(_shutdown) {
      return nullptr;
    }

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

    // get the tensor state if any
    auto &s = _tensors[_tid];

    // make sure that it was not created before
    assert(!s.is_created);

    // we are done writing to the tensor and
    s.is_created = true;
    s.writing_tensor = false;

    // go through the commands that are waiting
    auto cw = _commands_waiting_for[_my_rank].equal_range(_tid);
    for (auto it = cw.first; it != cw.second;) {

      // try to find the command
      auto jt = _local_commands.find(it->second);
      assert(jt != _local_commands.end());

      // check if we have all the inputs
      if(0 == (--jt->second.second)) {

        // schedule the command for execution
        _schedule_for_excution(std::move(jt->second.first));

        // remove the command
        _local_commands.erase(jt);
      }

      // remove the command from the waiting list
      it = _commands_waiting_for[_my_rank].erase(it);
    }

    // go through all nodes we need to notify once this tensor is created
    auto nd = _notify_on_creation.equal_range(_tid);
    while (nd.first != nd.second) {

      // add it to the send status queue so that we can notify the node
      auto it = nd.first;
      _send_status_queue[it->second].push_back(_tid);
      _send_status_cv[it->second].notify_all();

      // erase the ones we don't need
      _notify_on_creation.erase(++nd.first, nd.second);
    }
  }

  // returns tensors that are scheduled to be remove from the storage
  tid_t get_to_remove() {

    // wait until we have something to delete
    std::unique_lock<std::mutex> lk(_m);
    _cv.wait(lk, [&]{ return !_to_delete.empty() || _shutdown; });

    // if we have nothing to delete
    if(_to_delete.empty()) {
      return -1;
    }

    // return the tensor we want to delete
    auto tmp = _to_delete.back(); _to_delete.pop_back();
    return tmp;
  }

  void shutdown() {

    // set the flag
    std::unique_lock<std::mutex> lk(_m);
    _shutdown = true;

    // notify that we are done
    _cv.notify_all();
    _deletion_cv.notify_all();
    for(auto &cv : _send_status_cv) { cv.notify_all(); }
  }

 private:

  bool _queue_remote(command_ptr_t _command) {

    // handle delete
    if(_command->is_delete()) {

      // this is bad a remote delete should never exist... // TODO handle this gracefully
      return false;
    }

    // get the node that is going to initiate the execution of this command
    auto rootNode = _command->get_root_node();

    // count the number of inputs that are not present
    for(int32_t i = 0; i < _command->get_num_inputs(); i++) {

      // get the tensor required in the input
      auto _in = _command->get_input(i);

      // check if this input is supposed to be located on this machine
      if(_in.node == _my_rank) {

        // if it is we need to mark that it is being read and
        auto &s = _tensors[_in.tid];

        // mark that we are reading this tensor
        s.num_to_read++;

        // check if it was not crated we need to mark that
        // we need to notify the remote node once the tensor is created
        if(!s.is_created) {

          // mark that we need to notify once this tid is created _in.node
          _notify_on_creation.insert({_in.tid, rootNode});
        }
        else {

          // add it to the send status queue so that we can notify the node
          _send_status_queue[rootNode].push_back(_in.tid);
          _send_status_cv[rootNode].notify_all();
        }
      }
    }

    // go through the output tensors
    for(auto idx = 0; idx < _command->get_num_outputs(); ++idx) {

      // grab the output tensor
      auto &_out = _command->get_output(idx);

      // check if the node
      if(_out.node == _my_rank) {

        // get the tid
        auto &s = _tensors[_out.tid];

        // make sure everything is fine TODO I need to recover from this somehow...
        if(s.scheduled_for_delition && s.is_created) { return false; }

        // we are writing to this tensor
        s.writing_tensor = true;
      }
    }

    return true;
  }

  bool _queue_local(command_ptr_t _command) {

    // handle delete
    if(_command->is_delete()) {

      // go through the inputs and eiter remove the tensors directly,
      // or mark them for deletion if they are going to be used soon
      for(auto idx = 0; idx < _command->get_num_inputs(); ++idx) {

        // grab the input tensor
        auto &in = _command->get_input(idx);

        // mark the tensor as scheduled for deletion
        auto &s = _tensors[in.tid];

        // if we created the tensor, if not just delete it!
        if(s.is_created && s.num_to_read == 0 && !s.writing_tensor) {

          // remove the tensor immediately
          _remove_tensor(in.tid);
        }
        else {

          // ok the tensor is not ready for deletion schedule it
          s.scheduled_for_delition = true;
        }
      }

      // finish delete processed
      return true;
    }

    // count the number of inputs that are not present
    int32_t num_not_present = 0;
    for(int32_t i = 0; i < _command->get_num_inputs(); i++) {

      // get the tensor required in the input
      auto _in = _command->get_input(i);

      // check if this is a remote tensor or a local tensor
      if(_in.node == _my_rank) {

        // if it is a local we need to check if it was created already
        auto &s = _tensors[_in.tid];

        // make sure that this tensor was not deleted before this TODO I need to recover from this somehow...
        if(s.scheduled_for_delition) { return false; }

        // we are reading this tensor
        s.num_to_read++;

        // if it was not created we need to keep track of that
        if(!s.is_created) {

          // tensor is not present
          num_not_present++;

          // mark that this command is waiting
          _commands_waiting_for[_my_rank].insert({_in.tid, _command->id});
        }
      }
      else {

        // ok this is a remote tensor, we need to check if is present
        auto &_rts = _remote_tensors[_in.node];
        auto _ts = _rts.find(_in.tid);

        // if the tensor is not present we have to mark that we are waiting for
        // a notification from a remote node
        if(_ts == _rts.end()) {

          // tensor is not present
          num_not_present++;

          // mark the this command is waiting for a remote tensor
          _commands_waiting_for[_in.node].insert({_in.tid, _command->id});
        }
      }
    }

    // go through the output tensors
    for(auto idx = 0; idx < _command->get_num_outputs(); ++idx) {

      // grab the output tensor
      auto &_out = _command->get_output(idx);

      // check if the node
      if(_out.node == _my_rank) {

        // get the tid
        auto &s = _tensors[_out.tid];

        // make sure everything is fine TODO I need to recover from this somehow...
        if(s.scheduled_for_delition && s.is_created) { return false; }

        // we are writing to this tensor
        s.writing_tensor = true;
      }
    }

    // if we have all the required tensors we can kick off the command
    if(num_not_present == 0) {

      _execute.emplace_back(std::move(_command));
      _cv.notify_all();
    }
    else {

      // store the number of tensors this command is waiting for
      auto cmd_id = _command->id;
      _local_commands[cmd_id] = { std::move(_command),  num_not_present };
    }

    // we are done here
    return true;
  }

  // retire the local command
  bool _retire_command(command_ptr_t _command) {

    // if this is a delete we remove the tensor
    if (_command->type == command_t::op_type_t::DELETE) {

      // remove the tensors
      for(int32_t i = 0; i < _command->get_num_inputs(); i++) {

        // get the tensor required in the input
        auto t = _command->get_input(i);

        // remove the tensor immediately
        _remove_tensor(t.tid);

        // remove the command
        _local_commands.erase(_command->id);
      }

      return true;
    }

    // make sure to go through the created tensors
    for(int32_t i = 0; i < _command->get_num_outputs(); i++) {

      // get the tensor required in the output
      auto out = _command->get_output(i);

      // if this tensor is not on our node we don't do anything
      if(out.node != _my_rank) {
        continue;
      }

      // get the tid
      auto tid = out.tid;
      auto &s = _tensors[tid];

      // make sure that it was not created before
      assert(!s.is_created);

      // we are done writing to the tensor and
      s.is_created = true;
      s.writing_tensor = false;

      // remove the tensor if it is not needed
      if (s.num_to_read == 0 && s.scheduled_for_delition) {

        // remove the tensor immediately
        _remove_tensor(out.tid);
      }

      // go through the commands that are waiting
      auto cw = _commands_waiting_for[_my_rank].equal_range(tid);
      for (auto it = cw.first; it != cw.second;) {

        // try to find the command
        auto jt = _local_commands.find(it->second);
        assert(jt != _local_commands.end());

        // check if we have all the inputs
        if (0 == (--jt->second.second)) {

          // schedule the command for execution
          _schedule_for_excution(std::move(jt->second.first));

          // remove the command
          _local_commands.erase(jt);
        }

        // remove the command from the waiting list
        it = _commands_waiting_for[_my_rank].erase(it);
      }

      // go through all nodes we need to notify once this tensor is created
      auto nd = _notify_on_creation.equal_range(tid);
      while(nd.first != nd.second) {

        // add it to the send status queue so that we can notify the node
        auto it = nd.first;
        _send_status_queue[it->second].push_back(tid);
        _send_status_cv[it->second].notify_all();

        // erase the ones we don't need
        _notify_on_creation.erase(++nd.first, nd.second);
      }
    }

    for(int32_t i = 0; i < _command->get_num_inputs(); i++) {

      // get the tensor required in the input
      auto in = _command->get_input(i);

      // if this tensor is not on our node we don't do anything
      if(in.node != _my_rank) {
        continue;
      }

      // get the tid
      auto tid = in.tid;
      auto &s = _tensors[tid];

      // decrement the number of readers
      s.num_to_read--;
      assert(s.num_to_read >= 0);

      // if there are no command that is writing to this tensor
      // reading this tensor and the tensor is scheduled for deletion, delete it here
      if (s.num_to_read == 0 && !s.writing_tensor && s.scheduled_for_delition) {

        // remove the tensor immediately
        _remove_tensor(in.tid);
      }
    }

    // remove the command
    _local_commands.erase(_command->id);

    return true;
  }

  // retire remote the command
  bool _retire_remote_command(command_ptr_t _command) {

    // if this is a delete we remove the tensor
    if (_command->type == command_t::op_type_t::DELETE) {
      throw std::runtime_error("There are no remote delete instructions...");
    }

    // make sure to go through the created tensors
    for(int32_t i = 0; i < _command->get_num_outputs(); i++) {

      // get the tensor required in the output
      auto out = _command->get_output(i);

      // if this tensor is not on our node we don't do anything
      if(out.node != _my_rank) {
        continue;
      }

      // get the tid
      auto tid = out.tid;
      auto &s = _tensors[tid];

      // make sure that it was not created before
      assert(!s.is_created);

      // we are done writing to the tensor and
      s.is_created = true;
      s.writing_tensor = false;

      // remove the tensor if it is not needed
      if (s.num_to_read == 0 && s.scheduled_for_delition) {

        // remove the tensor immediately
        _remove_tensor(out.tid);
      }

      // go through the commands that are waiting
      auto cw = _commands_waiting_for[_my_rank].equal_range(tid);
      for (auto it = cw.first; it != cw.second;) {

        // try to find the command
        auto jt = _local_commands.find(it->second);
        assert(jt != _local_commands.end());

        // check if we have all the inputs
        if (0 == (--jt->second.second)) {

          // schedule the command for execution
          _schedule_for_excution(std::move(jt->second.first));

          // remove the command
          _local_commands.erase(jt);
        }

        // remove the command from the waiting list
        it = _commands_waiting_for[_my_rank].erase(it);
      }

      // go through all nodes we need to notify once this tensor is created
      auto nd = _notify_on_creation.equal_range(tid);
      while(nd.first != nd.second) {

        // add it to the send status queue so that we can notify the node
        auto it = nd.first;
        _send_status_queue[it->second].push_back(tid);
        _send_status_cv[it->second].notify_all();

        // erase the ones we don't need
        _notify_on_creation.erase(++nd.first, nd.second);
      }
    }

    for(int32_t i = 0; i < _command->get_num_inputs(); i++) {

      // get the tensor required in the input
      auto in = _command->get_input(i);

      // if this tensor is not on our node we don't do anything
      if(in.node != _my_rank) {
        continue;
      }

      // get the tid
      auto tid = in.tid;
      auto &s = _tensors[tid];

      // decrement the number of readers
      s.num_to_read--;
      assert(s.num_to_read >= 0);

      // if there are no command that is writing to this tensor
      // reading this tensor and the tensor is scheduled for deletion, delete it here
      if (s.num_to_read == 0 && !s.writing_tensor && s.scheduled_for_delition) {

        // remove the tensor immediately
        _remove_tensor(in.tid);
      }
    }

    return true;
  }

  void _schedule_for_excution(command_ptr_t _cmd) {

    // schedule the command for execution
    _execute.emplace_back(std::move(_cmd));
    _cv.notify_all();
  }

  // remove the tensor
  void _remove_tensor(tid_t _tid) {

    // remove the tensor from the storage
    _to_delete.push_back(_tid);

    // remove the tensor
    _tensors.erase(_tid);

    // make sure there are not commands waiting for the delete
    assert(_commands_waiting_for[_my_rank].find(_tid) == _commands_waiting_for[_my_rank].end());
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

  // is it still running
  bool _shutdown = false;

  // the node for which this reservation station is for
  node_id_t _my_rank;

  // the number of nodes in the cluster
  int32_t _num_nodes;

  // the id of the last command we have executed
  command_id_t _last_cmd = -1;

  // commands ready to execute
  std::vector<command_ptr_t> _execute;

  // the local commands and the number of tensors they are waiting for
  std::unordered_map<command_id_t, std::pair<command_ptr_t, int32_t>> _local_commands;

  // the local tensors commands are waiting for
  std::vector<std::unordered_multimap<tid_t, command_id_t>> _commands_waiting_for;

  // keeps all the local tensors and information about them
  std::unordered_map<tid_t, internal_tensor_state_t> _tensors;

  // keeps the status of all the remote tensors. If the tensor is present we store the command, that requested
  // the status for it. All the commands with a lower or equal command id can be executed safely as the tensor can not
  // be removed unless they are executed.
  std::vector<std::unordered_set<tid_t>> _remote_tensors;

  // tensors for which we need to send the status for, once they are created...
  std::unordered_multimap<tid_t, node_id_t> _notify_on_creation;

  // the status commands we need to send
  std::vector<std::vector<tid_t>> _send_status_queue;

  // we use this to wait for commands
  std::vector<std::condition_variable> _send_status_cv;

  // deletion cv
  std::condition_variable _deletion_cv;

  // the tensors we want to delete from storage
  std::vector<tid_t> _to_delete;
};

using reservation_station_ptr_t = std::shared_ptr<reservation_station_t>;

}