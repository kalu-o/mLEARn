/*
MIT License

Copyright (c) 2019 Kalu U. Ogbureke

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Authors: Kalu U. Ogbureke
Change Log: 01.04.2019 - Version 1.0.0
*/
#ifndef DATA_READERS_H
#define DATA_READERS_H
#include <fstream>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <vector>
#include <set>
#include "utils.h"

namespace mlearn {
/**
    The DataReader class is the base class responsible for
    reading train/test dataset into features/labels. 3 different
    readers are implemented, namely, MNIST_CIFARReader, GenericReader
    and IrisReader. They extend the base class DataReader.
*/
class DataReader
{
    protected:
        /** Name of input file */
        const std::string file_name;
        /** Vector of Node pointers for Features */
        vec_vec_ptr_double features;
        /** Vector of Node pointers for labels */
        vec_vec_ptr_double labels;
        /** Dimension of Features */
        uint64_t feature_dim;
        /** Dimension of labels */
        uint64_t label_dim;
        /** Number of instances */
        uint64_t row_dim;
        /** Encoding of labels as one-hot */
        bool one_hot{true};
        /** Character delimiter */
        char sep{','};
        /** Dataset can contain header information as the first line */
        bool header{false};

    public:
        /** The default constructor */
        DataReader(): feature_dim{0}, label_dim{0}, row_dim{0}{}
        /** Overloaded constructor with 1 argument */
        DataReader(const std::string file_name): file_name{file_name}{}
        /** Overloaded constructor with 3 arguments */
        DataReader(const std::string file_name, uint64_t feature_dim, uint64_t label_dim):
                  file_name{file_name},
                  feature_dim{feature_dim},
                  label_dim{label_dim}{}
        /** Overloaded constructor with 4 arguments */
        DataReader(const std::string file_name, uint64_t feature_dim, uint64_t label_dim, bool one_hot):
                  file_name{file_name},
                  feature_dim{feature_dim},
                  label_dim{label_dim},
                  one_hot{one_hot}{}
        /** Overloaded constructor with 5 arguments */
        DataReader(const std::string file_name, uint64_t feature_dim, uint64_t label_dim, char sep, bool header):
                  file_name{file_name},
                  feature_dim{feature_dim},
                  label_dim{label_dim},
                  sep{sep},
                  header{header}{}
        /** Overloaded constructor with 4 argument */
        DataReader(const std::string file_name, uint64_t feature_dim, uint64_t label_dim, char sep):
                  file_name{file_name},
                  feature_dim{feature_dim},
                  label_dim{label_dim},
                  sep{sep}{}
        /** Copy constructor */
        DataReader(const DataReader& arg):
                  file_name{arg.file_name},
                  features{arg.features},
                  labels{arg.labels},
                  feature_dim{arg.feature_dim},
                  label_dim{arg.label_dim},
                  row_dim{arg.row_dim}{}
        /** Returns features */
        const vec_vec_ptr_double& getFeatures()const{return features;}
        /** Returns labels */
        const vec_vec_ptr_double& getLabels()const{return labels;}
        /** Returns feature dimension */
        uint64_t getFeatureDim()const{return feature_dim;}
        /** Returns label dimension */
        uint64_t getLabelDim()const{return label_dim;}
        /** Returns number of instances */
        uint64_t getRowDim()const{return row_dim;}
        /** Reads dataset file into features and labels */
        virtual DataReader& read() = 0;
        /** Assignment operator */
        DataReader& operator=(const DataReader&);
        /**
            This is for shuffling train dataset for use with stochastic
            gradient descent optimizers and variants.

            @param indices Indices of training data
            @return A reference to the shuffled indices
        */
        std::vector<int>& shuffleIndex(std::vector<int>& indices)const;
        /**
            This splits train dataset into train/validation set.

            @param test An empty DataReader object that will hold validation set
            @return test_size Percentage of train set used for validation
        */
        DataReader& trainTestSplit(DataReader& test, double test_size = 0.1);
        /**
            Responsible for releasing/deleting dynamically allocated memory.

            @return Reference to empty DataReader object
        */
        DataReader& destroy();
        /** Virtual destructor */
        virtual ~DataReader(){destroy();}
};

/**
    The MNIST_CIFARReader class extends the DataReader class.
    It is responsible for reading the MNIST_CIFAR dataset into feature/label.
    The MNIST_CIFAR database of handwritten digits  by Yann Lecun, Corinna Cortes
    http://yann.lecun.com/exdb/mnist/. https://pjreddie.com/projects/mnist-in-csv/.
    \n
    @code
        //  Creates an MNIST_CIFARReader object mnist and read a header-less file
        //  "mnist_sample.csv". Entries are delimited by a comma
        MNIST_CIFARReader<double> mnist("data/mnist_sample.csv", ',',  false);
        //  Calls the read method
        mnist.read();
        //  Creates a vector of integers and shuffles it.
        std::vector<int> indices;
        mnist.shuffleIndex(indices);
        //  Creates an MNIST_CIFARReader object test
        MNIST_CIFARReader<double> test
        //  Splits "mnist" into train/validation set
        //  10% of the original data is copied to test
        mnist.trainTestSplit(test, 0.1);
    @endcode
*/
class MNIST_CIFARReader: public DataReader
{
    public:
        /** Default constructor */
        MNIST_CIFARReader(): DataReader(){}
        /** Overloaded constructor with 3 arguments. The
            number of features is 784(28 * 28) and the
            label dimension is 10 ("one_hot" set to true).
        */
        MNIST_CIFARReader(const std::string file_name, uint64_t feature_dim, uint64_t label_dim, char sep, bool header): DataReader(file_name, feature_dim, label_dim, sep, header){}
        /** Overloaded constructor with 2 arguments */
        MNIST_CIFARReader(const std::string file_name, uint64_t feature_dim, uint64_t label_dim, bool one_hot): DataReader(file_name, feature_dim, label_dim, one_hot){}
        /** Copy constructor.

            @param argv MNIST_CIFARReader object to be copied.
        */
        MNIST_CIFARReader(const MNIST_CIFARReader& argv);
        /**
            Overloaded assignment operator.

            @param argv Reference to the second operand
            @return Reference to self
        */
        MNIST_CIFARReader& operator=(const MNIST_CIFARReader& argv);
        /**
            Reads a text file containing features and labels.

            @return Reference to self.
        */
        MNIST_CIFARReader& read();
        /** Virtual destructor */
        virtual ~MNIST_CIFARReader(){}
};

/**
    The IrisReader class extends the DataReader class.
    It is responsible for reading the Iris dataset into feature/label
    (https://archive.ics.uci.edu/ml/datasets/iris)
    The class implements the "read" method specific to the Iris dataset.
*/
class IrisReader: public DataReader
{
    public:
        IrisReader(): DataReader(){}
        IrisReader(const std::string file_name): DataReader(file_name){}
        IrisReader(const std::string file_name, char sep, bool header): DataReader(file_name, sep, header){}
        IrisReader(const std::string file_name, uint64_t feature_dim, uint64_t label_dim, char sep):
                  DataReader(file_name, feature_dim, label_dim, sep){}
        IrisReader(const IrisReader& arg): DataReader(arg){}
        IrisReader& operator=(const IrisReader&);
        IrisReader& read();
        virtual ~IrisReader(){}
};

/**
    The GenericReader class extends the DataReader class and implements
    the "read" method. It is responsible for reading any text dataset into
    features/labels. The file can contain header information. Each row of
    sample should be concatenation of features and labels, where "sep" is
    the delimiter.

    The following is an example for the "xor" dataset. The line "x1,x2,y1"
    is the header. x1 and x2 are the features and y1 the label. "0,0,0" is an
    instance containing features and labels. The delimiter is a comma.
    \n
    @code
    x1,x2,y1
    0,0,0
    0,1,1
    1,0,1
    1,1,0
    @endcode

*/
class GenericReader: public DataReader
{
    public:
        GenericReader(): DataReader(){}
        GenericReader(const std::string file_name, char sep, bool header):
                     DataReader(file_name, sep, header){}
        GenericReader(const std::string file_name, uint64_t feature_dim, uint64_t label_dim, char sep):
                     DataReader(file_name, feature_dim, label_dim, sep){}
        GenericReader(const std::string file_name, uint64_t feature_dim, uint64_t label_dim, char sep, bool header):
                     DataReader(file_name, feature_dim, label_dim, sep, header){}
        GenericReader(const std::string file_name, uint64_t feature_dim, uint64_t label_dim, bool one_hot):
                     DataReader(file_name, feature_dim, label_dim, one_hot){}
        GenericReader(const GenericReader& arg): DataReader(arg){}
        GenericReader& operator=(const GenericReader&);
        GenericReader& read();
        virtual ~GenericReader(){}
};
} // namespace mlearn
#endif
