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
#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include <vector>
#include "network.h"
#include "data_reader.h"

namespace mlearn {
template <class T>
/**
    The Optimizer class is the base class responsible for
    training algorithms. 3 optimizers are currently implemented,
    namely, SGD, Adagrad and RMSProp. The SGD optimizer is mini-
    batch stochastic gradient descent. The difference between Adagrad/
    RMSProp and SGD is that the latter uses variable/per parameter learning rate.
*/
class Optimizer
{
    public:
        /**
            Virtual function. This is responsible for model training.
            .
            @param model The Model object to train
            @param model_file The name of file to save trained model
            @param train Pointer to train dataset
            @param validation Pointer to validation dataset. Default is nullptr
            @param id The type of optimizer
            @return A reference to the trained model
        */
        virtual Network<T>& train(
                Network<T>& model,
                std::string model_file,
                const DataReader<T>* train,
                const DataReader<T>* validation = nullptr,
                std::string id = "sgd");

        /**
            Virtual function. This is responsible for prediction/test.
            .
            @param model Trained model object loaded from file
            @param test Pointer to test dataset
            @param model_file The name of file to save trained model
            @return A test metric (accuracy, mse, mae)
        */
        virtual double predict(Network<T>& model, const DataReader<T>* test, std::string model_file);
        /**
            Virtual function. This is responsible for model update during training.
            .
            @param model The model object to update
            @return A reference to updated model
        */
        virtual Network<T>& update(Network<T>& model);
        /** Virtual destructor */
        virtual ~Optimizer(){}
};
template <class T>
/**
    The base optimizer function that implements vanilla SGD.
    Other optimizers call the SGDHelper function.
    .
    @param model Model object to train
    @param batch_size Train batch size
    @param num_epochs Number of train epochs
    @param opt Pointer to optimizer
    @param train Pointer to train dataset
    @param validation Pointer to validation dataset. Default is nullptr
    @param id Type of optimizer
    @return A reference to the trained model
*/
Network<T>& SGDHelper(
        Network<T>& model,
        uint32_t batch_size,
        uint32_t num_epochs,
        Optimizer<T>* opt,
        const DataReader<T>* train,
        const DataReader<T>* validation = nullptr,
        std::string id = "sgd");
template <class T>
/**
    The SGD class extends the Optimizer class. It implements
    the classical mini-batch stochastic gradient descent.
*/
class SGD : public Optimizer<T>
{
    private:
        /** The id/type of optimizer */
        std::string id{"sgd"};
    protected:
        /** Train learning rate */
        double learning_rate{0.1};
        /** Train batch size */
        uint32_t batch_size{10};
        /** Number of train epochs */
        uint32_t num_epochs{20};
        /** Regularization parameter */
        double lambda{0.0};
        /** Type of Regularization (L1 or L2) */
        std::string reg{"None"};
        /** Momentum term/parameter */
        double beta{0.0};

    public:
        /** Default constructor */
        SGD(){}
        /** Overloaded constructor with 6 arguments */
        SGD(double learning_rate, uint32_t batch_size, uint32_t n_epochs, double lambda, std::string reg,  double beta):
             learning_rate{learning_rate}, batch_size{batch_size}, num_epochs{n_epochs}, lambda{lambda}, reg{reg}, beta{beta}{}
        /** Overloaded constructor with 3 arguments */
        SGD(double learning_rate, uint32_t batch_size, uint32_t n_epochs): learning_rate{learning_rate}, batch_size{batch_size}, num_epochs{n_epochs}{}
        /** Overloaded constructor with 5 arguments */
        SGD(double learning_rate, uint32_t batch_size, uint32_t n_epochs, double lambda, std::string reg):
             learning_rate{learning_rate}, batch_size{batch_size}, num_epochs{n_epochs}, lambda{lambda}, reg{reg}{}
        /** Implements train function */
        Network<T>& train(Network<T>&, std::string, const DataReader<T>*, const DataReader<T>* = nullptr, std::string = "sgd");
        /** Implements update function */
        Network<T>& update(Network<T>&);
        /** Implements predict function */
        double predict(Network<T>&, const DataReader<T>*, std::string);
        /** Virtual destructor */
        virtual ~SGD(){}

};
template <class T>
/**
    The Adagrad class extends the SGD class. It implements the
    adaptive rate SGD. The only difference as compared
    to SGD is how the parameters are updated. Adagrad adapts
    the learning rate to each parameter.

    Adaptive gradient method
    J Duchi, E Hazan and Y Singer,
    Adaptive subgradient methods for online learning and stochastic optimization
    The Journal of Machine Learning Research, pages 2121-2159, 2011.


*/
class Adagrad : public SGD<T>
{
    private:
        /** The id/type of optimizer */
        std::string id{"adagrad"};

    public:
        /** Default constructor */
        Adagrad(){}
        /** Overloaded constructor with 6 arguments */
        Adagrad(double learning_rate, uint32_t batch_size, uint32_t n_epochs, double lambda, std::string reg,  double beta):
            SGD<T>(learning_rate, batch_size, n_epochs, lambda, reg, beta){}
        /** Overloaded constructor with 3 arguments */
        Adagrad(double learning_rate, uint32_t batch_size, uint32_t n_epochs): SGD<T>(learning_rate, batch_size, n_epochs){}
        /** Overloaded constructor with 5 arguments */
        Adagrad(double learning_rate, uint32_t batch_size, uint32_t n_epochs, double lambda, std::string reg):
            SGD<T>(learning_rate, batch_size, n_epochs, lambda, reg){}
        /** Implement train function */
        Network<T>& train(Network<T>&, std::string, const DataReader<T>*, const DataReader<T>* = nullptr, std::string = "adagrad");
        /** Implement update function */
        virtual Network<T>& update(Network<T>&);
        /** Implement predict function */
        double predict(Network<T>&, const DataReader<T>*, std::string);
        /** Virtual destructor */
        virtual ~Adagrad(){}
};

template <class T>
/**
    The RMSProp class extends the SGD class. It implements
    the root means square SGD. The only difference as compared
    to SGD is how the parameters are updated. RMSProp adapts
    the learning rate to each parameter.

    Root mean square propagation (RMSProp)
    T Tieleman, and G E Hinton,
    Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine Learning (2012)
*/
class RMSProp : public SGD<T>
{
    private:
        /** The id/type of optimizer */
        std::string id{"rmsprop"};
    public:
        /** Default constructor */
        RMSProp(){}
        /** Overloaded constructor with 6 arguments */
        RMSProp(double learning_rate, uint32_t batch_size, uint32_t n_epochs, double lambda, std::string reg,  double beta):
            SGD<T>(learning_rate, batch_size, n_epochs, lambda, reg, beta){}
        /** Overloaded constructor with 3 arguments */
        RMSProp(double learning_rate, uint32_t batch_size, uint32_t n_epochs): SGD<T>(learning_rate, batch_size, n_epochs){}
        /** Overloaded constructor with 5 arguments */
        RMSProp(double learning_rate, uint32_t batch_size, uint32_t n_epochs, double lambda, std::string reg):
            SGD<T>(learning_rate, batch_size, n_epochs, lambda, reg){}
        /** Implements train function */
        Network<T>& train(Network<T>&, std::string, const DataReader<T>*, const DataReader<T>* = nullptr, std::string = "rmsprop");
        /** Implements update function */
        virtual Network<T>& update(Network<T>&);
        /** Implements predict function */
        double predict(Network<T>&, const DataReader<T>*, std::string);
        /** Virtual destructor */
        virtual ~RMSProp(){}
};
template <class T>
class PSDSquare : public SGD<T>
{
    private:
        /** The id/type of optimizer */
        std::string id{"psdsquare"};
    public:
        /** Default constructor */
        PSDSquare(){}
        /** Overloaded constructor with 6 arguments */
        PSDSquare(double learning_rate, uint32_t batch_size, uint32_t n_epochs, double lambda, std::string reg,  double beta):
            SGD<T>(learning_rate, batch_size, n_epochs, lambda, reg, beta){}
        /** Overloaded constructor with 3 arguments */
        PSDSquare(double learning_rate, uint32_t batch_size, uint32_t n_epochs): SGD<T>(learning_rate, batch_size, n_epochs){}
        /** Overloaded constructor with 5 arguments */
        PSDSquare(double learning_rate, uint32_t batch_size, uint32_t n_epochs, double lambda, std::string reg):
            SGD<T>(learning_rate, batch_size, n_epochs, lambda, reg){}
        /** Implements train function */
        Network<T>& train(Network<T>&, std::string, const DataReader<T>*, const DataReader<T>* = nullptr, std::string = "psdsquare");
        /** Implements update function */
        virtual Network<T>& update(Network<T>&);
        /** Implements predict function */
        double predict(Network<T>&, const DataReader<T>*, std::string);
        /** Virtual destructor */
        virtual ~PSDSquare(){}
};
} // namespace mlearn

#endif
