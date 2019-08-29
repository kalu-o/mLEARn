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
#ifndef LAYER_H
#define LAYER_H
#include <iostream>
#include <vector>
#include "libutil.h"

namespace mlearn {
template <class T>
/**
    The Layer class represents a layer in a neural network.
    The pointers "previous" and "next" point to the previous
    and next layers respectively. Each layer has a forward
    function(forwardProp) that produces output data from input
    data and a backward function(backwardProp) that produces an
    output delta (gradient) from an input delta. A layer can be
    a hidden or an output layer.
    \n
    @code
        //  Creates 2 Activation objects: hidden and output
        Activation<double> hidden("sigmoid"), output("softmax");
        //  A hidden layer with input and output dimensions 2
        //  Activation function used is sigmoid
        Layer<double> hidden_layer(2, 2, "hidden", hidden);
        //  An output layer with input and output dimensions 2 and 1 respectively.
        //  Activation function used is softmax
        Layer<double> output_layer(2, 1, "output", output);
    @endcode

    @note
    The output dimension of the last hidden layer must be same as the input dimension
    of the output layer.
*/
class Layer
{
    protected:
        /** Input dimension of layer */
        uint64_t input_dim;
        /** Output dimension of layer */
        uint64_t output_dim;
        /** Pointer to previous layer */
        Layer* previous;
        /** Pointer to next layer */
        Layer* next;
        /** Output data of layer */
        NetNode<T> output_data;
        /** Output delta of layer */
        NetNode<T> output_delta;
        /** Input data of layer */
        NetNode<T> input_data;
        /** Input delta of layer */
        NetNode<T> input_delta;
        /** Bias of layer */
        NetNode<T> bias;
        /** Delta of bias */
        NetNode<T> delta_b;
        /** Contains activations */
        NetNode<T> activation;
        /** Weight matrix of layer */
        mublas::matrix<T> weight;
        /** Delta of weight matrix */
        mublas::matrix<T> delta_w;
        /** Weight matrix used for regularization */
        mublas::matrix<T> regularize_w;
        /** Weight matrix used for momentum */
        mublas::matrix<T> momentum_w;
        /** Accumulates previous deltas. Used in Adagrad and RMSProp */
        mublas::matrix<T> sq_delta_w;
        /** Layer type (hidden_layer or output_layer) */
        std::string type;
        /** Activation function of layer */
        Activation<T> act_function;
        mublas::vector<T> psdsquare;
        /** Responsible for saving/serialization of members */
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const uint64_t version)
        {
            if (version >= 0)
            {
                ar & input_dim;
                ar & output_dim;
                ar & output_data;
                ar & output_delta;
                ar & input_data;
                ar & input_delta;
                ar & bias;
                ar & delta_b;
                ar & activation;
                ar & weight;
                ar & delta_w;
                ar & regularize_w;
                ar & momentum_w;
                ar & sq_delta_w;
                ar & type;
                ar & act_function;
                ar & psdsquare;
            }
        }

    public:
        /** Default constructor */
        Layer():  input_dim{0}, output_dim{0}, previous{nullptr}, next{nullptr}, type{"hidden"}{}
        /** Overloaded constructor with 4 arguments */
        Layer(uint64_t input_dim, uint64_t output_dim, std::string type, Activation<T> act_function):
             input_dim{input_dim},
             output_dim{output_dim},
             previous{nullptr},
             next{nullptr},
             bias{output_dim},
             delta_b{output_dim},
             weight(output_dim, input_dim),
             delta_w(output_dim, input_dim, 0),
             regularize_w(output_dim, input_dim, 0),
             momentum_w(output_dim, input_dim, 0),
             sq_delta_w(output_dim, input_dim, 0.1),
             type{type},
             act_function{act_function},
             psdsquare(output_dim)
        {initialize(); }
        /**
            Used to initialize weight in layers.
            Generates random values between a range.
            The values are  uniformly sampled between
            sqrt(-6./(input_dim + output_dim)) * 4 and
            sqrt(6./(input_dim + output_dim)) * 4.
            (Y. Bengio, X. Glorot, Understanding the difficulty
             of training deep feedforward neuralnetworks, AISTATS 2010).

        */
        Layer<T>& initialize();
        /**
            Uses vector to initialize rows of weight matrix.

            @param row_id Matrix row index
            @param in_data Vector of data
            @return Reference to self
        */
        Layer<T>& push_row(uint64_t row_id, const std::vector<T>& in_data)
        {
            for (uint64_t i = 0; i < in_data.size(); ++i) weight(row_id, i) = in_data[i];
            return *this;
        }
        /**
            Connects a layer to its neighbors.

            @param layer Layer to be connected
            @param in_data Vector of data
            @return Returns void
        */
        void connect(Layer<T>& layer);
        /**
            The first layer gets input data from the training data,
            while other layers get their inputs from the output of
            previous layer.

            @param out An empty NetNode
            @return out Input data
        */
        NetNode<T>& getForwardInput(NetNode<T>& out);
        /**
            An input data is propagated forward through a layer
            and produces an output data.

            @return output_data Output data
        */
        NetNode<T>& forwardProp();
        /**
            The last layer gets input delta from the derivative of
            the cost function, while other layers get their input deltas
            from their successors.

            @return out Input delta
        */
        NetNode<T>& getBackwardInput(NetNode<T>& out);
        /**
            Propagates input error (delta) backward and produces an output delta.

            @return output_delta Output delta
        */
        NetNode<T>& backwardProp();
        /** Returns the weight matrix of a layer */
        const mublas::matrix<T>& getWeight()const{return weight;}
        /** Returns the bias vector of a layer in a NetNode object */
        const NetNode<T>& getBias()const{return bias;}
        /** Returns the delta weight matrix of a layer */
        const mublas::matrix<T>& getDeltaWeight()const{return delta_w;}
        /** Returns the delta bias vector of a layer in a NetNode object */
        const NetNode<T>& getDeltaBias()const{return delta_b;}
        /** Returns the input data to a layer in a NetNode object */
        const NetNode<T>& getInputData()const{return input_data;}
        /** Returns the output data from a layer in a NetNode object */
        const NetNode<T>& getOutputData()const{return output_data;}
        /** Returns the input delta to a layer in a NetNode object */
        const NetNode<T>& getInputDelta()const{return input_delta;}
        /** Returns the output delta from a layer in a NetNode object */
        const NetNode<T>& getOutputDelta()const{return output_delta;}
        /** Sets the weight matrix of a layer */
        void setWeight(const mublas::matrix<T>& in_weight){weight = in_weight;}
        /** Sets the bias vector of a layer */
        void setBias(const NetNode<T>& in_bias){bias = in_bias;}
        /** Sets the delta weight matrix of a layer */
        void setDeltaWeight(const mublas::matrix<T>& in_delta_w){delta_w = in_delta_w;}
        /** Sets the bias vector of a layer */
        void setDeltaBias(const NetNode<T>& in_delta_b){delta_b = in_delta_b;}
        /** Sets the input data to a layer */
        void setInputData(const NetNode<T>& in_data){input_data = in_data;}
        /** Sets the output data from a layer */
        void setOutputData(const NetNode<T>& out_data){output_data = out_data;}
        /** Sets the input delta to a layer */
        void setInputDelta(const NetNode<T>& in_delta){input_delta = in_delta;}
        /** Sets the output delta from a layer */
        void setOutputDelta(const NetNode<T>& out_delta){output_delta = out_delta;}
        /**
            After a single forward and backward pass, the parameters of
            each layer is updated, and the deltas reset to zero. Beta is
            a momentum to use to decide if a part of a previous weight
            update should be used. default is 0 (no momentum).

            @param beta Momentum term/parameter
            @return Reference to self
        */
        Layer<T>& clearDeltas(double beta = 0.0);
        /**
            Overloaded function. Handles regularization of layer
            parameters (weight). Currently, only L1 and L2 are
            implemented. The default is no regularization(None).

            @param rate Learning rate
            @param lambda Regularization parameter
            @param reg Type of regularization (L1, L2 or None)
            @return Reference to self
        */
        Layer<T>& regularize(double rate, double lambda, std::string reg = "None");
        /**
            Overloaded function. Handles regularization for Adagrad/RMSProp.

            @param rate Learning rate
            @param lambda Regularization parameter
            @param reg Type of regularization (L1, L2 or None)
            @return Reference to self
        */
        Layer<T>& regularize(const mublas::matrix<T>& rate, double lambda, std::string reg = "None");
        /**
            Overloaded function. After a single forward and backward pass,
            the parameters (weight and bias) of each layer are updated.

            @param rate Learning rate
            @param batch_size Batch size used in training
            @param lambda Regularization parameter
            @param reg Type of regularization (L1, L2 or None)
            @param beta Momentum term/parameter
            @return Reference to self
        */
        Layer<T>& updateParams(double rate, uint32_t batch_size, double lambda, std::string reg, double beta);
        /**
            Overloaded function used for parameter updates of Adagrad/RMSProp.

            @param rate Learning rate
            @param batch_size Batch size used in training
            @param lambda Regularization parameter
            @param reg Type of regularization (L1, L2 or None)
            @param beta Momentum term/parameter
            @param change_rate Determines if learning rates should be changed
            @return Reference to self
        */
        Layer<T>& updateParams(double rate, uint32_t batch_size, double lambda, std::string reg, double beta, bool change_rate, std::string = "adagrad");
        /** Virtual destructor */
        virtual ~Layer(){}
};
} // namespace mlearn
#endif
