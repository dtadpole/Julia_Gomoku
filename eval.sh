#!/bin/bash

while true ; do
    julia ./eval.jl --mcts_temperature_mean 0.3 "$@"
    sleep 30
done
