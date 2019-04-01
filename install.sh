#!/bin/bash
#rm -rf build
mkdir -p build
cd build
cmake  -DBUILD_SHARED_LIBS=OFF ..
make 
