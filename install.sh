#!/bin/bash
#rm -rf build
mkdir -p build
cd build
cmake  -- /verbosity:detailed -DBUILD_SHARED_LIBS=OFF ..
make 
