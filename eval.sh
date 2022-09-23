#!/bin/bash

while true ; do
    julia ./eval.jl --mcts_temperature_mean 0.25 "$@"
    sleep 30
done
