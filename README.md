# mLEARn
An Implementation of Mutli-layer Perceptron in C++. 

## Features
### General
* Single-threaded CPU only
* Reasonable fast
* Simple C++ constructs
* Linux platform currently supported
* Feed-forward networks

### activation functions
* tanh
* sigmoid
* softmax
* rectified linear unit (relu)
* leaky relu
* identity
* exponential linear units(elu)

### cost functions
* cross-entropy
* mean squared error (MSE)
* mean absolute error (MAE)

### optimization algorithms
* mini-batch stochastic gradient descent (SGD)
* adagrad
* rmsprop

### Miscellenous
* L1/L2 regularization
* Momentum

## Building from Source
### dependencies
* C++11 compiler
* Boost (program_options, unit_test_framework, serialization, ublas)
* CMake (tested with version 3.5.1 )

### binaries and static libraries
The following will build static libraries and binaries. The binaries and libraries are located in the bin and lib directories respectively.
```bash
$ mkdir -p build
$ cd build
$ cmake  ..
$ make 
```
## Running mlearn Program
The executables (mlearn and unit_test) will reside in bin/. You can run from the project directory or put the binaries in /usr/local/bin/ or execution path. Note that the programs look for the MNIST dataset in the data/ directory. So the data/ directory should be in your run directory, if you want to run unit test or train/test with MNIST.

### data format
Each row of sample should be concatenation of features and labels using comma as the delimiter. The file may contain a header. Sample dataset for xor and MNIST can be found in the data/ directory. The following is a header-less xor data file, of feature and label dimensions 2 and 1 respectively. The delimiter is a comma (the last column represents the labels).
```
1,0,1
0,1,1
0,0,0
1,1,0
1,0,1
0,1,1
0,0,0
1,1,0
1,0,1
0,1,1
0,0,0
1,1,0
```
In order to know  the parameters and usage:
```bash
$ mlearn -h 
$ mlearn --help
```
