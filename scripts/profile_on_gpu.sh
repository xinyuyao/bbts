#!/bin/sh

# run the integration test as an mpi program
tmp=$((PMI_RANK))

export CUDA_VISIBLE_DEVICES=$tmp
export LC_ALL=C

# run the thing
nvprof -f -o ./dump.%q{CUDA_VISIBLE_DEVICES}.nvvp ./bin/$1