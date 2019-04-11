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

#ifndef LIBUTIL_H
#define LIBUTIL_H
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <stdlib.h>
#include <exception>
#include <iomanip>
#include <stdexcept>
#include <chrono>
#include <ctime>
#include <random>
#define EPSILON 1e-2
#define ADAGRAD_EPSILON 1e-8
#define MAX_MNIST_VALUE 255
#define MAX_IRIS_VALUE 7.9

/**
    This contains declarations of various classes in libutil.cpp,
    and resides in the mlearn namespace.
    @file libutil.h
    @author Kalu U. Ogbureke
    @version 1.0.0
    @date 13.02.2019
*/
namespace mublas = boost::numeric::ublas;
namespace mlearn {
template <class T>
/**
    The Node class is the fundamental data structure used.
    It contains 2 protected members: data and data_size.
    data is of type boost ublas vector and may hold features
    or labels. Different machine learning methods can extend
    the Node class, e.g. for multi-layer perceptron, NetNode
    extends the Node class.
*/
class Node
{
    protected:
        /** This can hold features/labels */
        mublas::vector<T> data;
        /** This is the size/dimension of data */
        uint64_t data_size;

        /** The is responsible for saving/serialization of members */
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive& ar, const uint64_t version)
        {
            if (version >= 0)
            {
                ar & data;
                ar & data_size;
            }
        }

    public:
        /** The default constructor */
        Node(): data_size{0}{}
        /**
            Overloaded constructor that takes one argument: in_data.

            @param in_data A vector of data used to initialize the "data" member
        */
        Node(const mublas::vector<T>& in_data): data{in_data}, data_size{in_data.size()}{}
        /**
            Overloaded constructor that takes one argument: data_size.
            This initializes "data" member with a zero vector of size "data_size".

            @param data_size The size/dimension of "data"
        */
        Node(uint64_t data_size): data_size{data_size}
        {
            mublas::zero_vector<T> in_data(data_size);
            data = in_data;
        }
        /**
            The copy constructor.

            @param in_node The Node object to be copied
        */
        Node(const Node<T>& in_node): data{in_node.data}, data_size{in_node.data_size}{}
        /**
            Accessor for the Node "data" member.

			@pre None
			@post Does not change the object
            @return data A vector of type T
        */
        const mublas::vector<T>& getData()const{return data;}
        /**
            A Mutator for the Node data member.

            @pre None
            @post The "data" member variable of Node will be changed to the input value.
            @param in_data A vector of input data of type T, copied to the "data" member
            @return *this A reference to the calling object
        */
        Node<T>& setData(const mublas::vector<T>& in_data)
        {
            data = in_data;
            data_size = data.size();
            return *this;
        }
        /**
            Accessor for the Node "data_size" member.

			@pre None
			@post Does not change the object.
            @return data_size The size/dimension of "data"
        */
        uint64_t getDataSize()const{return data_size;}
        /**
            Overloaded addition operator.

			@param argv A reference to the second operand.
            @return A new Node object.
        */
        Node<T> operator+(const Node<T>& argv);
        /**
            Overloaded subtraction operator.

			@param argv A reference to the second operand
            @return A new Node object
        */
        Node<T> operator-(const Node<T>& argv);
        /**
            Overloaded multiplication operator.

			@param argv A reference to the second operand
            @return A new Node object
        */
        Node<T> operator*(const Node<T>& argv);
        /**
            Overloaded assignment operator.

			@param argv A reference to the second operand
            @return A reference to self.
        */
        Node<T>& operator=(const Node<T>& argv);
        /**
            Implements scalar multiplication (constant * data).

            @param rate A scalar of type "double"
            @return A new Node object with scaled "data"
        */
        Node<T> scalarMultiply(double rate);
        /** Returns the sum of "data" of the calling object (this). */
        double sum()const;
        /** Returns the sum of abs of "data". */
        double fabsSum()const;
        /** Prints values of "data" */
        const Node<T>& describeNode()const
        {
            std::cout <<getData() << std::endl;
            return *this;
        }
        /**
            Generates "data" randomly between a range. Useful for
            constructing a Node object with randomly generated "data"
            value. The data follows a distribution with lower and upper
            bounds computed from the passed parameters.

            @param input_dim Input size
            @param output_dim Output size
            @return A reference to self
        */
        Node<T>& generateRandomData(uint64_t input_dim, uint64_t output_dim);

        /**
            Compares the values of "data" between self and argv.
            If it's less, returns 1; if it's greater, returns -1
            and otherwise returns 0.

            @param argv Node to be compared
            @param result Result Node
            @return A reference to result Node
        */
        Node<T>& compare(const Node<T>& argv, Node<T>& result)const;
        /**
           A virtual destructor. The Node class is a base class.
           Base pointers and references will often be used for
           derived classes. Making this virtual ensures the correct
           destructor is called.
        */
        virtual ~Node(){}
};

template <class T>
/**
    The NetNode class is a class derived from the Node class.
    The NetNode class extends the Node class and inherits "data"
    and 'data_size' members of the Node class. It adds functions
    specific to neural networks.
*/
class NetNode : public Node<T>
{
    public:
        /** The default constructor */
        NetNode(){}
        /**
            Overloaded constructor that takes one argument: in_data.

            @param in_data A vector of data used to initialize the "data" member.
        */
        NetNode(const mublas::vector<T>& in_data): Node<T>(in_data){}
        /** Calls the Node(uint64_t data_size) constructor */
        NetNode(uint64_t data_size): Node<T>(data_size){}
        /** The copy constructor */
        NetNode(const NetNode<T>& in_node): Node<T>(in_node){}
        /** Calls the Node copy constructor */
        NetNode(const Node<T>& in_node): Node<T>(in_node){}
        /** Overloaded assignment operator */
        NetNode& operator=(const NetNode&);
        /**
            Computes the sigmoid (activation) of values of "data".

            @pre None
			@post Modifies "data" values.
            @return A reference to self (with sigmoid of "data" values)
        */
        NetNode<T>& sigmoid();
        /**
            Computes the derivative of sigmoid. The derivative of sigmoid
            is \f{equation}{f(x) * (1 - f(x)) \f}.

            @return A reference to self (with sigmoid derivative of "data" values)
        */
        NetNode<T> sigmoidPrime();
        /**
            Computes the tanh (activation) of values of "data".

            @pre None
			@post Modifies "data" values.
            @return A reference to self (with tanh of "data" values)
        */
        NetNode<T>& hyperTan();
        /**
            Computes the derivative of tanh. The derivative of tanh
            is \f{equation}{1 - (f(x) * f(x)) \f}.

            @return A reference to self (with tanh derivative of "data" values)
        */
        NetNode<T>& hyperTanPrime();
        /**
            Computes the Rectified linear Unit (ReLU) activation: max(0, x).

            @pre None
			@post Modifies "data" values.
            @return A reference to self (with ReLU of "data" values)
        */
        NetNode<T>& ReLU();
        /**
            The derivative of ReLU. This becomes a leaky ReLU if "alpha"
            is greater than 0.

            @param alpha A parameter used for leaky ReLU
            @return A reference to self (with the derivative of ReLU of "data" values)
        */
        NetNode<T>& ReLUPrime(double alpha);
        /**
            Computes softmax activation function; forces sum of probabilities to 1.

            @pre None
			@post Modifies "data" values.
            @return A reference to self (with softmax of "data" values)
        */
        NetNode<T>& softmax();
        /**
            Computes derivative of softmax.

            @param argv An empty 2D Jacobian matrix
            @return A reference to argv
        */
        mublas::matrix<T>& softmaxPrime(mublas::matrix<T>& argv);
        /**
            Computes exponential linear unit (ELU) activation.

            @param alpha A parameter used in ELU
            @return A reference to self (with ELU of "data" values)
        */
        NetNode<T>& ELU(double alpha);
        /**
            Computes the derivative of ELU.

            @param alpha A parameter used in ELU
            @return A reference to self
        */
        NetNode<T>& ELUPrime(double alpha);
        /** The identity function. Does not modify "data" */
        NetNode<T>& identity();
        /** The derivative of identity function */
        NetNode<T>& identityPrime();
        /** Computes log to base e of "data" */
        NetNode<T>& log_e();
        /** A virtual destructor */
        virtual ~ NetNode(){}
};

template <class T>
/**
    The Activation class handles activations in the network.
    Currently, functions implemented are sigmoid, tanh, ReLU,
    leaky ReLU, identity, softmax and ELU.
*/
class Activation
{
    protected:
        /** The type of function, default sigmoid */
        std::string type{"sigmoid"};
        /** A parameter used in some functions such as ELU, leaky ReLU */
        double alpha{0.0};
        /** The is responsible for saving/serialization of members */
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const uint64_t version)
        {
            if (version >= 0)
            {
                ar & type;
                ar & alpha;
            }
        }

    public:
        /** The default constructor */
        Activation(){}
        /** Overloaded constructor.
            @param type The type of activation function.
        */
        Activation(std::string type): type{type}{}
        /** Overloaded constructor.

            @param type Type of activation function
            @param alpha Parameter for some functions
        */
        Activation(std::string type, double alpha): type{type}, alpha{alpha}{}
        /** Computes the activation function.

            @param input Input (type NetNode) to the function
            @return Computed value of the function
        */
        NetNode<T>& compute(NetNode<T>& input);
        /** Overloaded function that computes the derivative of
            activation function.

            @param input Input (type NetNode) to the function
            @return Computed derivative of the function
        */
        NetNode<T>& computeDerivative(NetNode<T>& input);
        /** Overloaded function that computes the derivative of
            activation function. This returns a 2D Jacobian matrix.

            @param input Input (type NetNode) to the function
            @param jacobian_m An empty 2D Jacobian matrix
            @return Computed derivative of the function.
        */
        mublas::matrix<T>& computeDerivative(NetNode<T>&, mublas::matrix<T>& jacobian_m);
        /** Accessor function that returns the type of activation function */
        std::string getType(){return type;}
};

template <class T>
/**
    The CostFunction class is responsible for objective functions.
    Cost functions implemented are mean squared error (MSE),
    mean absolute error (MAE) and cross entropy.
*/
class CostFunction
{
    protected:
        /** Type of cost function */
        std::string id;
    public:
        /** Default constructor */
        CostFunction(){}
        /** Constructor that takes a single parameter: id */
        CostFunction(std::string id): id{id}{}
        /** Accessor function that returns the id of the cost function */
        std::string getId()const{return id;}
        /** Pure virtual function */
        virtual double cost(const NetNode<T>& prediction, const NetNode<T>& label) = 0;
        /** Pure virtual function */
        virtual double accuracy(const NetNode<T>& prediction, const NetNode<T>& label) = 0;
        /** Pure virtual function */
        virtual NetNode<T>& costDerivative(const NetNode<T>&, const NetNode<T>&, NetNode<T>&) = 0;
        /** Virtual destructor */
        virtual ~CostFunction(){}
};

template <class T>
/**
    The MSE (mean squared error) derives from the class CostFunction. It implements
    "cost" "accuracy" and "costDerivative" functions. Used for regression problems.
*/
class MSE : public CostFunction<T>
{
    public:
        /** Default constructor */
        MSE(): CostFunction<T>("mse"){}
        /** Computes the error between the prediction and label.
            This is the square of the difference between the prediction
            and label.

            @param prediction Hypotheses
            @param label Reference
            @return The square of the difference between the prediction and label
        */
        double cost(const NetNode<T>& prediction, const NetNode<T>& label);
        /** Computes the error between the prediction and label.
            Calls the cost function.

            @param prediction Hypotheses
            @param label Reference
            @return The square of the difference between the prediction and label
        */
        double accuracy(const NetNode<T>& prediction, const NetNode<T>& label) {return cost(prediction, label);}
        /** Computes the derivative of MSE. This is the difference
            between the prediction and label.

            @param prediction Hypotheses
            @param label Reference
            @param result Result
            @return The difference between the prediction and label
        */
        NetNode<T>& costDerivative(const NetNode<T>& prediction, const NetNode<T>& label, NetNode<T>& result);

};

template <class T>
/**
    CrossEntropy derives from the class CostFunction.
    It implements the "cost", "accuracy" and "costDerivative" functions.
    Used for classification problems.
*/
class CrossEntropy : public CostFunction<T>
{
    public:
        /** Default constructor */
        CrossEntropy(): CostFunction<T>("crossentropy"){}
        /** Computes the loss between prediction and label.
            This is the negative of product of label and log of prediction.

            @param prediction Hypotheses
            @param label Reference
            @return Product of negative log of prediction and label
        */
        double cost(const NetNode<T>& prediction, const NetNode<T>& label);
        /** Computes the accuracy between prediction and label.
            This is +1 if prediction and label are same and 0 otherwise.

            @param prediction Hypotheses
            @param label Reference
            @return +1 if prediction and label are same and 0 otherwise
        */
        double accuracy(const NetNode<T>& prediction, const NetNode<T>& label);
        /** Computes the derivative. This is implemented as the difference
            between the prediction and label.

            @param prediction Hypotheses
            @param label Reference
            @param result Result
            @return The difference between the prediction and label
        */
        NetNode<T>& costDerivative(const NetNode<T> &prediction, const NetNode<T> &label, NetNode<T>& result);

};

template <class T>
/**
    The MAE(mean absolute error) derives from the class CostFunction.
    It implements the "cost", "accuracy" and "costDerivative" functions.
    Used for regression problems.
*/
class MAE : public CostFunction<T>
{
    public:
        /** Default constructor */
        MAE(): CostFunction<T>("mae"){}
        /** Computes the absolute error between the prediction and label.

            @param prediction Hypotheses
            @param label Reference
            @return The absolute difference between the prediction and label
        */
        double cost(const NetNode<T>& prediction, const NetNode<T>& label);
        /** Computes the absolute error between prediction and label.
            Calls the cost function.

            @param prediction Hypotheses
            @param label Reference
            @return The absolute difference between the prediction and label
        */
        double accuracy(const NetNode<T>& prediction, const NetNode<T>& label) {return cost(prediction, label);}
        /** Computes the derivative of MAE.

            @param prediction Hypotheses
            @param label Reference
            @param result Result
            @return Derivative of MAE
        */
        NetNode<T>& costDerivative(const NetNode<T>& prediction, const NetNode<T>& label, NetNode<T>& result);

};

template <class T>
/**
    Releases dynamically allocated vector of pointers
    to Node objects and clears the vector.

    @param argv Vector of pointers to Node
    @return Empty vector
*/
std::vector<Node<T>*>& destroy(std::vector<Node<T>*>& argv)
{
    try{
        for (auto ptr: argv)delete ptr;
        argv.clear();
    }catch (std::logic_error const& e){
        std::cerr << e.what() << std::endl;
    }

    return argv;
}

template <class T>
/**
    Computes the mean of values of a matrix.

    @param argv The matrix
    @return The mean
*/
double mean(mublas::matrix<T> argv)
{
    double total_sum = 0.0;
    uint64_t num_items = argv.size1() * argv.size2();
    if(num_items == 0)return total_sum;
    for (uint64_t i = 0; i < argv.size1(); ++ i)
        for (uint64_t j = 0; j < argv.size2(); ++ j)
            total_sum += argv(i, j);
    return total_sum/num_items;
}
} // namespace mlearn
#endif
