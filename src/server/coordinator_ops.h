#pragma once

namespace bbts {

enum class coordinator_op_types_t : int32_t {

  FAIL, // indicates a coordinator failure
  SCHEDULE, // batch schedules commands
  RUN, // runs all the commands
  CLEAR, // clear the storage
  SHUTDOWN, // clear the storage
  VERBOSE, // turn on or off the debug messages
};

struct coordinator_op_t {

  // the type of the op
  coordinator_op_types_t _type;

  // used by schedule
  size_t _val;
};



}