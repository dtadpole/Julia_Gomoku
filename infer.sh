#!/bin/bash

while true ; do
    julia ./infer.jl "$@"
    sleep 3
done
