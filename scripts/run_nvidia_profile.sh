#!/bin/sh

# run the integration test as an mpi program
sudo rm /tmp/dump.*
sudo mpirun -np $2 nvprof --unified-memory-profiling off -o /tmp/dump.%q{PMI_RANK}.nvvp ./bin/$1
echo "profiling in /tmp/dump.[n]"