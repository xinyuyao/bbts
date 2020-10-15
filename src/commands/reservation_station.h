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

  reservation_station_t(node_id_t _node_id, storage_ptr_t _storage) : _node_id(_node_id), _storage(std::move(_storage)) {}

  // queue a command, this command has to be executed in the same thread and the commands
  // have to be queued in the exact order they are coming in
  bool queue_command(command_ptr_t _command) {

    // lock the reservation station
    std::unique_lock<std::mutex> lk(_m);

    if(_command->get_root_node() == _node_id) {
      return _queue_local(std::move(_command));
    }
    else {
      return _queue_remote(std::move(_command));
    }
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

  // mark that a command is processed
  bool retire_command(command_ptr_t _command) {
    
    // if this is a delete we remove the tensor 
    if(_command->_type == command_t::op_type_t::DELETE) {
      
      // remove the tensors
      for(auto &t : _command->_input_tensors) {

        // remove the tensor immediately
        _remove_tensor(t.tid);

        // remove the command
        _local_commands.erase(_command->_id);
      }

      return true;
    }

    // make sure to go through the created tensors
    for(auto &out : _command->_output_tensors) {
      
      // get the tid
      auto tid = out.tid;
      auto &s = _tensors[tid];

      // make sure that it was not created before
      assert(!s.is_created);

      // we are done writing to the tensor and
      s.is_created = true;
      s.writing_tensor = false;

      // go through the commands that are waiting
      auto cw = _commands_waiting_for.equal_range(tid);
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
        it = _commands_waiting_for.erase(it);
      }
    }

    for(auto &in : _command->_input_tensors) {

      // get the tid
      auto tid = in.tid;
      auto &s = _tensors[tid];

      // decrement the number of readers
      s.num_to_read--;
      assert(s.num_to_read >= 0);

      // if there are no command that is writing to this tensor
      // reading this tensor and the tensor is scheduled for deletion, delete it here
      if(s.num_to_read == 0 && !s.writing_tensor && s.scheduled_for_delition) {

        // remove the tensor immediately
        _remove_tensor(in.tid);
      }
    }

    return true;
  }

  // 
  void retire_remote_command(command_ptr_t _command) {
    
  }

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
    auto cw = _commands_waiting_for.equal_range(_tid);
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
      it = _commands_waiting_for.erase(it);
    }
  }

private:

  bool _queue_remote(command_ptr_t _command) {

    // you can not schedule remote delete
    if(_command->is_delete()) {
      return false;
    } 

    // counts all the tensors not present
    int32_t not_present = 0;

    // the input tensors
    for(auto &in : _command->_input_tensors) {

      // see if we already have this tensor, if we don't we need to wait for it
      auto &s = _tensors[in.tid];

      // make sure that this tensor was not deleted before this
      if(s.scheduled_for_delition) { return false; }

      // we are reading this tensor
      s.num_to_read++;
      not_present++;
    }

    // go through the ouput tensors
    for(auto &out : _command->_output_tensors) {
      
      // get the tid
      auto &s = _tensors[out.tid];

      // make sure everything is fine
      if(s.scheduled_for_delition && s.is_created) { return false; }

      // we are writing to this tensor
      s.writing_tensor = true;
      not_present++;
    }

    // check if we actually had something
    if(not_present != 0) {
      
      // store the remote command
      _remote_commands[_command->_id] = std::move(_command);

      // we scheduled it
      return true;
    }
    
    return false;
  }

  bool _queue_local(command_ptr_t _command) {

    // if the command is a delete, schedule all the tensors for deletion
    if(_command->is_delete()) {

      for(auto &in : _command->_input_tensors) {

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

    // counts all the tensors not present
    int32_t not_present = 0;

    // the input tensors
    for(auto &in : _command->_input_tensors) {

      // see if we already have this tensor, if we don't we need to wait for it
      auto it = _tensors.find(in.tid);

      // make sure that this tensor was not deleted before this
      if(it->second.scheduled_for_delition) { return false; }

      // we are reading this tensor
      it->second.num_to_read++;
    
      // if the tensor is not yet created wait for it
      if(!it->second.is_created) {
        
        // add the entry
        _commands_waiting_for.insert({in.tid, _command->_id});

        // we have more
        not_present++;
      }
    }

    // go through the ouput tensors
    for(auto &out : _command->_output_tensors) {
      
      // get the tid
      auto &s = _tensors[out.tid];

      // make sure everything is fine
      if(s.scheduled_for_delition && s.is_created) { return false; }

      // we are writing to this tensor
      s.writing_tensor = true;
    }

    // check if there are some tensors that are not present
    if(not_present != 0) {

      // store the command
      auto cmd_id = _command->_id;
      _local_commands[cmd_id] = {std::move(_command),  not_present}; 
    }
    else {

      // add the command
      _schedule_for_excution(std::move(_command));
    }

    // we are done here
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
    _storage->remove_by_tid(_tid);
    
    // remove the tensor
    _tensors.erase(_tid);

    // make sure there are not commands waiting for the delete
    assert(_commands_waiting_for.find(_tid) == _commands_waiting_for.end());
  }

  // the state of the tensor
  struct internal_tensor_state_t {

    // the number of commands to read this
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
  node_id_t _node_id;

  // the storage
  storage_ptr_t _storage;

  // commands ready to execute
  std::vector<command_ptr_t> _execute;

  // the local commands and the number of tensors they are waiting for
  std::unordered_map<command_id_t, std::pair<command_ptr_t, int32_t>> _local_commands;

  // the remote commands
  std::unordered_map<command_id_t, command_ptr_t> _remote_commands;

  // what do these entries require
  std::unordered_multimap<tid_t, command_id_t> _commands_waiting_for;

  // keeps all the tensors 
  std::unordered_map<tid_t, internal_tensor_state_t> _tensors;
};

}