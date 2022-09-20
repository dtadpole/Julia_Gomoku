#!/bin/bash

while true ; do
    julia ./infer.jl "$@"
    sleep 5
done
