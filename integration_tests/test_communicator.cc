#include <mpich/mpi.h>
#include <stdio.h>
#include <iostream>

struct matrix_t {
  int num_rows;
  int num_cols;

  float data[0];
};

matrix_t *allocate_matrix(int32_t num_rows, int32_t num_cols) {
  auto tmp = (matrix_t*) malloc(sizeof(matrix_t) + sizeof(float) * num_rows * num_cols);
  tmp->num_rows = num_rows;
  tmp->num_cols = num_cols;
  return tmp;
}

size_t get_size(const matrix_t &_matrix) {
  return sizeof(matrix_t) + sizeof(float) * _matrix.num_rows * _matrix.num_cols;
}

void add(const matrix_t &_in1, const matrix_t &_in2, matrix_t &_out) {
  for (int i = 0; i < _in1.num_cols * _in1.num_rows; ++i) {
    _out.data[i] = _in1.data[i] + _in2.data[i];
  }
  _out.num_rows = _in1.num_rows;
  _out.num_cols = _in1.num_cols;
}

size_t get_add_size(const matrix_t &_in1, const matrix_t &_in2) {
  return sizeof(matrix_t) + sizeof(float) * _in1.num_rows * _in1.num_cols;
}


// This code is based on the implementation from MPICH-1.
// Here's the algorithm.  Relative to the root, look at the bit pattern in
// my rank.  Starting from the right (lsb), if the bit is 1, send to
// the node with that bit zero and exit; if the bit is 0, receive from the
// node with that bit set and combine (as long as that node is within the
// group)

// Note that by receiving with source selection, we guarantee that we get
// the same bits with the same input.  If we allowed the parent to receive
// the children in any order, then timing differences could cause different
// results (roundoff error, over/underflows in some cases, etc).

// Because of the way these are ordered, if root is 0, then this is correct
// for both commutative and non-commutitive operations.  If root is not
// 0, then for non-commutitive, we use a root of zero and then send
// the result to the root.  To see this, note that the ordering is
// mask = 1: (ab)(cd)(ef)(gh)            (odds send to evens)
// mask = 2: ((ab)(cd))((ef)(gh))        (3,6 send to 0,4)
// mask = 4: (((ab)(cd))((ef)(gh)))      (4 sends to 0)

// Comments on buffering.
// If the datatype is not contiguous, we still need to pass contiguous
// data to the user routine.
// In this case, we should make a copy of the data in some format,
// and send/operate on that.

// In general, we can't use MPI_PACK, because the alignment of that
// is rather vague, and the data may not be re-usable.  What we actually
// need is a "squeeze" operation that removes the skips.
matrix_t *reduce(int32_t num_nodes, int32_t my_rank, int32_t root, int32_t tag, matrix_t &_in) {

  MPI_Status status;
  MPI_Message message;

  int32_t mask = 0x1;
  int32_t lroot = 0;

  // relative rank
  int32_t relrank = (my_rank - lroot + num_nodes) % num_nodes;

  // get the lhs address
  matrix_t *lhs = &_in;

  // do stuff
  int32_t source;
  while (mask < num_nodes) {

    // receive 
    if ((mask & relrank) == 0) {
      
      source = (relrank | mask);
      if (source < num_nodes) {

        // wait till we get a message from the right node
        source = (source + lroot) % num_nodes;
        auto mpi_errno = MPI_Mprobe(source, tag, MPI_COMM_WORLD, &message, &status);
        
        // check if there is an error
        if (mpi_errno) {
          std::cout << "Error \n";
        }

        // allocate a buffer for the tensor
        int32_t count; 
        MPI_Get_count(&status, MPI_CHAR, &count);
        matrix_t *rhs = (matrix_t*) malloc(count);

        // recieve the stuff
        mpi_errno = MPI_Mrecv (rhs, count, MPI_CHAR, &message, &status);

        // check if there is an error
        if (mpi_errno) {
          std::cout << "Error \n";
        }

        // how much do we need to allocated
        auto output_size = get_add_size(*lhs, *rhs);

        // allocate the output
        auto out = (matrix_t*) malloc(output_size);

        // add them together
        add(*lhs, *rhs, *out);

        // manage the memory
        if(lhs != &_in) {
            free(lhs);
        }
        free(rhs);
        
        // set the lhs
        lhs = out;
      }

    } else {

      // I've received all that I'm going to.  Send my result to my parent
      source = ((relrank & (~mask)) + lroot) % num_nodes;

      // do the sending
      size_t num_bytes = get_size(*lhs);
      auto mpi_errno = MPI_Ssend(lhs, num_bytes, MPI_CHAR, source, tag, MPI_COMM_WORLD);

      // log the error if there was any
      if (mpi_errno) {        
          std::cout << "Error \n";
      }

      break;
    }
    mask <<= 1;
  }

  // the result is at the node with rank 0, we need to move it
  if (root != 0) {

    // the node with rank 0 sends the node with the root rank recieves
    int mpi_errno;
    if (my_rank == 0) {

      // send it to the root
      size_t num_bytes = get_size(*lhs);
      mpi_errno = MPI_Ssend(lhs, num_bytes, MPI_CHAR, root, tag, MPI_COMM_WORLD);

    } else if (my_rank == root) {
      
        // wait for the message
        auto mpi_errno = MPI_Mprobe(0, tag, MPI_COMM_WORLD, &message, &status);
        
        // check if there is an error
        if (mpi_errno) {
          std::cout << "Error \n";
        }

        // allocate a buffer for the tensor
        int32_t count; 
        MPI_Get_count(&status, MPI_CHAR, &count);

        // manage the memory
        if(lhs != &_in) {
            std::cout << "Got here\n" << std::flush;
            free(lhs);
        }
        lhs = (matrix_t*) malloc(count);

        // recieve the stuff
        mpi_errno = MPI_Mrecv (lhs, count, MPI_CHAR, &message, &status);

        // check if there is an error
        if (mpi_errno) {
          std::cout << "Error \n";
        }
    }
    
    if (mpi_errno) {
          std::cout << "Error \n";
    }
  }

  // free the lhs
  if(my_rank != root) {
    if(lhs != &_in) {
      free(lhs);
    }
    lhs = nullptr;
  }

  // return the reduced tensor
  return lhs;
}

int main(int argc, char **argv) {
  // Initialize the MPI environment
  MPI_Init(NULL, NULL);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  // Print off a hello world message
  printf("Hello world from processor %s, rank %d out of %d processors\n", processor_name, world_rank, world_size);

  auto a = allocate_matrix(100, 200);

  for(int i = 0; i < a->num_cols * a->num_rows; ++i) {
    a->data[i] = 1 + world_rank + i;
  }

  auto b = reduce(world_size, world_rank, 0, 111, *a);

  if(b != nullptr) {
    for(int i = 0; i < a->num_cols * a->num_rows; ++i) {
        
        int32_t val = 0;
        for(int rank = 0; rank < world_size; ++rank) {
          val += 1 + rank + i;
        }
        if(b->data[i] != val) {
          std::cout << "not ok" << '\n';
        }
    }
  }
  
  free(a);
  free(b);
    
  // Finalize the MPI environment.
  MPI_Finalize();
}