#include "scheduler.h"

bbts::scheduler_t::scheduler_t(bbts::communicator_ptr_t _comm) : _comm(std::move(_comm)) {
    _shutdown = false;
}

void bbts::scheduler_t::schedule(bbts::command_ptr_t _cmd) {

    std::unique_lock<std::mutex> lk(_m);

    std::cout << "Added\n";
    // store the command
    _cmds_to_bcst.push(std::move(_cmd));
    _cv.notify_one();
}

void bbts::scheduler_t::accept() {

    // receive the commands
    while(true) {

        // get the next command
        auto _cmd = _comm->expect_cmd();

        std::cout << "Got\n";

        // check if we are done that is if the command is null
        if(_cmd == nullptr) {
            break;
        }

        // add the command to the reservation station
        _rs->queue_command(std::move(_cmd));
    }
}

void bbts::scheduler_t::forward() {

    // process it
    while(true) {

        // the command we want to forward
        command_ptr_t _cmd;

        // get the lock
        std::unique_lock<std::mutex> lk(_m);
        _cv.wait(lk, [&]{ return _shutdown || !_cmds_to_bcst.empty(); });

        // check if we are shutdown
        if(_shutdown) {
            break;
        }

        // grab the command
        _cmd = std::move(_cmds_to_bcst.front());
        _cmds_to_bcst.pop();

        // unlock here
        lk.unlock();

        // forward the command to the right nodes
        _comm->forward_cmd(_cmd);

        // move it to the reservation station
        if(_cmd->uses_node(_comm->get_rank())) {
            _rs->queue_command(std::move(_cmd));
        }
    }
}

void bbts::scheduler_t::shutdown() {

    std::unique_lock<std::mutex> lk(_m);

    // mark the we are done
    _shutdown = true;
    _cv.notify_one();
}
