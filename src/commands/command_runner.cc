#include "command_runner.h"
#include "../operations/move_op.h"
#include "../operations/reduce_op.h"
#include "../operations/broadcast_op.h"
#include "../operations/local_reduce_op.h"
#include <thread>

bbts::command_runner_t::command_runner_t(bbts::storage_ptr_t ts,
                                         bbts::tensor_factory_ptr_t tf,
                                         bbts::udf_manager_ptr udm,
                                         bbts::reservation_station_ptr_t rs,
                                         bbts::communicator_ptr_t comm,
                                         bbts::logger_ptr_t logger,
                                         tensor_stats_ptr_t stats)  
    : _ts(std::move(ts)), _tf(std::move(tf)), _udm(std::move(udm)),
      _rs(std::move(rs)), _comm(std::move(comm)), _logger(std::move(logger)), _stats(std::move(stats)) {}

void bbts::command_runner_t::local_command_runner() {

  while (true) {

    // get the command
    auto cmd = _rs->get_next_command();
    if (cmd == nullptr) {
      break;
    }

    // if we have a move
    if (cmd->type == command_t::MOVE) {

      // move the
      _logger->message("MOVE " + std::to_string(cmd->id) + " on my_node : " + std::to_string(_comm->get_rank()) + " Executed...\n");

      // it is a point to point move
      if(cmd->is_move()) {

        // get the tensor we want to sent
        auto t = _ts->get_by_tid(cmd->get_input(0).tid);

        // get the size of the tensor
        auto num_bytes = _tf->get_tensor_size(t->_meta);

        // set it as a command extra info
        cmd->nfo.num_bytes = num_bytes;

        // forward the command to the right nodes
        if(!_comm->op_request(cmd)) {
          throw std::runtime_error("Failed to forward the command.");
        }

        // create the move operation
        move_op_t op(*_comm, cmd->id, t, num_bytes, *_stats, cmd->get_input(0).tid, true, *_ts, cmd->get_output(0).node);

        // do the apply
        op.apply();

        // retire the command so it knows that we have processed the tensors
        _rs->retire_command(std::move(cmd));
      }
      // it is a broadcast
      else {

        _logger->message("BROADCAST\n");

        // get the tensor we want to sent
        auto t = _ts->get_by_tid(cmd->get_input(0).tid);

        // get the size of the tensor
        auto num_bytes = _tf->get_tensor_size(t->_meta);

        // set it as a command extra info
        cmd->nfo.num_bytes = num_bytes;

        // forward the command to the right nodes
        if(!_comm->op_request(cmd)) {
          throw std::runtime_error("Failed to forward reduce command.");
        }

        // get the nodes involved
        auto nodes = cmd->get_nodes();

        // create the move operation
        broadcast_op_t op(*_comm, *_ts, *_stats, nodes, cmd->id, t, num_bytes, cmd->get_input(0).tid);

        // do the apply
        op.apply();

        // retire the command so it knows that we have processed the tensors
        _rs->retire_command(std::move(cmd));
      }

    } else if (cmd->type == command_t::APPLY) {

      _logger->message("APPLY " + std::to_string(cmd->id) + " on my_node : " + std::to_string(_comm->get_rank()) + " Executed...\n");

      // return me that matcher for matrix addition
      auto ud = _udm->get_fn_impl(cmd->fun_id);

      /// 1. Figure out the meta data for the output

      // make the input meta arguments
      ud_impl_t::meta_args_t input_meta_args(cmd->get_num_inputs());
      for (size_t idx = 0; idx < cmd->get_num_inputs(); ++idx) {

        // store it
        auto t = _ts->get_by_tid(cmd->get_input(idx).tid);
        assert(t != nullptr);
        input_meta_args.set(idx, t->_meta);
      }

      // figure out the output arguments
      std::vector<tensor_meta_t> arguments(cmd->get_num_outputs());
      ud_impl_t::meta_args_t output_meta_args(arguments);

      // get the output meta
      ud->get_out_meta({ ._params = cmd->get_parameters() }, input_meta_args, output_meta_args);

      /// 2. Prepare the output arguments

      // make the input arguments
      ud_impl_t::tensor_args_t input_args(cmd->get_num_inputs());
      for (size_t idx = 0; idx < cmd->get_num_inputs(); ++idx) {

        // store it
        auto t = _ts->get_by_tid(cmd->get_input(idx).tid);
        input_args.set(idx, *t);
      }

      // setup the output arguments
      ud_impl_t::tensor_args_t output_args(cmd->get_num_outputs());
      for (size_t idx = 0; idx < cmd->get_num_outputs(); ++idx) {

        // the size of the tensor
        auto ts_size = _tf->get_tensor_size(output_meta_args.get_by_idx(idx));

        // create it
        auto tid = cmd->get_output(idx).tid;
        auto is_gpu = _stats->is_gpu(tid);
        auto t = _ts->create_tensor(tid, ts_size, is_gpu);

        // get the type of the output
        auto &type = ud->outputTypes[idx];
        t->_meta.fmt_id = _tf->get_tensor_ftm(type);

        // set the output arg
        output_args.set(idx, *t);
      }

      /// 3. Run the actual UD Function

      // apply the ud function
      ud->call_ud(bbts::ud_impl_t::tensor_params_t{._params = cmd->get_parameters() }, input_args, output_args);

      // retire the command so it knows that we have processed the tensors
      _rs->retire_command(std::move(cmd));

    } else if (cmd->type == command_t::DELETE) {

      // this should never happen
      throw std::runtime_error("We should never get a delete to execute, delete is implicit...");

    } else if (cmd->type == command_t::REDUCE) {

      // check if the reduce is remote or local
      if (cmd->is_local_reduce(_comm->get_rank())) {

        // return me that matcher for matrix addition
        auto ud = _udm->get_fn_impl(cmd->fun_id);

        // preallocate the input pointers
        auto cmd_inputs = cmd->get_inputs();
        std::vector<bbts::tensor_t*> inputs;
        inputs.reserve(cmd_inputs.size());

        // get all the tensors we need
        for(const auto& in : cmd_inputs) {

          // get the source tensor
          auto t = _ts->get_by_tid(in.tid);
          inputs.push_back(t);
        }

        // create the reduce op
        local_reduce_op_t op(*_tf, *_ts, *_stats, inputs, { ._params = cmd->get_parameters() },
                             cmd->get_output(0).tid, *ud);

        // do the apply
        op.apply();

        _logger->message("LOCAL_REDUCE " + std::to_string(cmd->id) + " on node " + std::to_string(_comm->get_rank()) + '\n');

        // retire the command so it knows that we have processed the tensors
        _rs->retire_command(std::move(cmd));


      } else {

        _logger->message("REMOTE_REDUCE_SCHEDULED");

        // forward the command to the right nodes
        if(!_comm->op_request(cmd)) {
          throw std::runtime_error("Failed to forward reduce command.");
        }

        // get the nodes involved
        auto nodes = cmd->get_nodes();

        // return me that matcher for matrix addition
        auto ud = _udm->get_fn_impl(cmd->fun_id);

        // preallocate the input pointers
        auto cmd_inputs = cmd->get_inputs();
        std::vector<bbts::tensor_t*> inputs;
        inputs.reserve(cmd_inputs.size());

        // get all the tensors we need
        for(const auto& in : cmd_inputs) {

          // check if the node
          if(in.node == _comm->get_rank()) {

            // get the source tensor
            auto t = _ts->get_by_tid(in.tid);
            inputs.push_back(t);
          }
        }

        // create the move operation
        reduce_op_t op(*_comm, *_tf, *_ts, *_stats, nodes, cmd->id, inputs, { ._params = cmd->get_parameters() }, cmd->get_output(0).tid, *ud);

        // do the apply
        op.apply();

        // retire the command so it knows that we have processed the tensors
        _rs->retire_command(std::move(cmd));

        //_logger->message("REMOTE_REDUCE PROCESSED on node " + std::to_string(_comm->get_rank()) + '\n');
      }
    }
  }
}

void bbts::command_runner_t::remote_command_handler() {

  // while we
  while (true) {

    // get the remote command
    auto cmd = _comm->expect_op_request();

    // if this is a shutdown just finish immediately
    if(cmd->type == bbts::command_t::MOVE) {

      // check if this is the move
      if(cmd->is_move()) {

        // kick off a thread to process the request
        std::thread child = std::thread([this, c = std::move(cmd)]() mutable {

          if(c->nfo.num_bytes == 0) {
            std::cout << "Empty tensor " << '\n' << std::flush;
          }

          // create the move operation
          move_op_t op(*_comm, c->id, nullptr, c->nfo.num_bytes, *_stats, c->get_input(0).tid, false, *_ts, c->get_input(0).node);

          // do the apply
          op.apply();

          // retire the command
          _rs->retire_command(std::move(c));
        });

        // detach the thread
        child.detach();
      }
      // check if this is a broadcast
      else {

        // kick off a thread to process the request
        std::thread child = std::thread([this, c = std::move(cmd)]() mutable {

          // get the nodes involved
          auto nodes = c->get_nodes();

          // create the move operation
          broadcast_op_t op(*_comm, *_ts, *_stats, nodes, c->id, nullptr, c->nfo.num_bytes, c->get_input(0).tid);

          // do the apply
          op.apply();

          // retire the command so it knows that we have processed the tensors
          _rs->retire_command(std::move(c));
        });

        // detach the thread
        child.detach();
      }
    }
    else if(cmd->type == bbts::command_t::REDUCE) {

      // kick off a thread to process the request
      std::thread child = std::thread([this, c = std::move(cmd)]() mutable {

        // get the nodes involved
        auto nodes = c->get_nodes();

        // return me that matcher for matrix addition
        auto ud = _udm->get_fn_impl(c->fun_id);

        // preallocate the input pointers
        auto cmd_inputs = c->get_inputs();
        std::vector<bbts::tensor_t*> inputs;
        inputs.reserve(cmd_inputs.size());

        // get all the tensors we need
        for(const auto& in : cmd_inputs) {

          // check if the node
          if(in.node == _comm->get_rank()) {

            // get the source tensor
            auto t = _ts->get_by_tid(in.tid);
            inputs.push_back(t);
          }
        }

        // create the move operation
        reduce_op_t op(*_comm, *_tf, *_ts, *_stats, nodes, c->id, inputs, { ._params = c->get_parameters() }, c->get_output(0).tid, *ud);

        // do the apply
        op.apply();

        // retire the command
        _rs->retire_command(std::move(c));
      });

      // detach the thread
      child.detach();
    }
    else if(cmd->type == bbts::command_t::SHUTDOWN) {
      break;
    }
    else {

      // throw the runtime error
      throw std::runtime_error("This is bad can not process a command of type " + std::to_string(cmd->type));
    }
  }
}

void bbts::command_runner_t::run_deleter() {

  // while we have something remove
  tid_t id;
  while (true) {

    // get the next tensor to remove
    id = _rs->get_to_remove();
    if (id == -1) {
      break;
    }

    // deleted
    _ts->remove_by_tid(id);
    _logger->message("Remove tensor : " + std::to_string(id) + '\n');

    // remove it from the reservation station
    _rs->retire_remove(id);
  }
}

void bbts::command_runner_t::shutdown() {
  _comm->shutdown_op_request();
}