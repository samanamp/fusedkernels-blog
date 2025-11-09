#!/bin/bash
if [ -f ./image_change ]; then
    echo "Removing old binary..."
    rm ./image_change
fi
set -e
# g++ -std=c++20 image_change.cpp -o image_change
nvcc -std=c++20 image_change.cpp -o image_change
./image_change