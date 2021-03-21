#!/bin/sh

# run the integration test as an mpi program
rm /tmp/dump.*
mpirun -np $2 sudo nvprof -o /tmp/dump.'printenv PMI_RANK'.nvvp ./bin/$1
echo "profiling in /tmp/dump.[n]"