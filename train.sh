#!/bin/bash

while true ; do
    julia ./train.jl "$@"
    sleep 1
done
