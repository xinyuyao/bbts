#!/bin/sh

# run the integration test as an mpi program
tmp=$((PMI_RANK + 1))

export CUDA_VISIBLE_DEVICES=$tmp
export LC_ALL=C

# run the thing
nvprof --unified-memory-profiling off -f -o /tmp/dump.%q{CUDA_VISIBLE_DEVICES}.nvvp ./bin/$1