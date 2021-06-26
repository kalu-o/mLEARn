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
#ifndef NETWORKS_H
#define NETWORKS_H
#include <vector>
#include <fstream>
#include "layers.h"

namespace mlearn {
/**
    The Network class is a classic multi-layer perceptron(MLP)
    consisting of sequences of layers: one or more hidden layers
    and an output layer. The network is trained using mini-batch
    SGD, Adagrad or RMSProp.
    \n
    @code
        //  Creates 2 Activation objects: hidden and output.
        Activation<double> hidden("sigmoid"), output("softmax");
        //  A hidden layer with input and output dimensions 2.
        //  Activation function used is sigmoid
        Layer<double> hidden_layer(2, 2, "hidden", hidden);
        //  An output layer with input and output dimensions 2 and 1 respectively.
        //  Activation function used is softmax
        Layer<double> output_layer(2, 1, "output", output);
        //  Creates a Network object, cost function is cross entropy
        Network<double> model(new CrossEntropy<double>);
        //  Adds the hidden and output layers to the network
        model.addLayer(hidden_layer);
        model.addLayer(output_layer);
        //  Connects the layers together
        model.connectLayers();
    @endcode

*/
class Network
{
    protected:
        /** A vector of Layer objects */
        std::vector <Layer*> layers;
        /** A pointer to CostFunction */
        Cost* loss;
        /** Used to decide if training rate
            should be updated/changed for
            Adagrad/RMSProp.
        */
        bool update_rate{false};
        /** Responsible for saving/serialization of members */
         friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const uint64_t version)
        {
            if (version >= 0)
            {
				ar.register_type(static_cast<DenseLayer *>(NULL));
				ar.register_type(static_cast<SigmoidLayer *>(NULL));
				ar.register_type(static_cast<TanhLayer *>(NULL));
				ar.register_type(static_cast<ReLULayer *>(NULL));
				ar.register_type(static_cast<ELULayer *>(NULL));
				ar.register_type(static_cast<SoftmaxLayer *>(NULL));
				
				//ar.template register_type<Layer >();
				/* ar.template register_type<DenseLayer >();
				ar.template register_type<SigmoidLayer >();
				ar.template register_type<TanhLayer >();
				ar.template register_type<ReLULayer >();
				ar.template register_type<ELULayer >();
				ar.template register_type<SigmoidLayer >(); */
                ar & layers;
                //ar & update_rate;
            }
        } 
    public:
        /** Default constructor, default cost function MSE */
        Network(): loss{new MSE}{}
        /** Overloaded constructor with 1 argument */
        Network(Cost* objective): loss{objective}{}
        /** Adds a layer to the network.

            @param layer The layer to be added
            @return A reference to self
        */
        Network& addLayer(Layer* layer);
        /** Connects all layers in the network and
            returns a reference to self.
        */
        Network& connectLayers();
        /** Returns the layers in a network */
        std::vector <Layer*>& getLayers(){return layers;}
        /** Inputs a single data through the network and propagates forward.

            @param in_data The input data; a NetNode object
            @return A reference to output data
        */
        mublas::vector<double>& singleForward(const mublas::vector<double>& in_data);
        /** Inputs delta through the network and propagates backward.

            @param in_delta The input delta; a NetNode object
            @return A reference to output delta
        */
        mublas::vector<double>& singleBackward(const mublas::vector<double>& in_delta);
        /** Overloaded function. Updates network parameters.

            @param learning_rate Train learning rate
            @param batch_size Batch size used in training
            @param lambda Regularization parameter (between 0 and 1)
            @param reg Type of regularization (L1, L2 or None)
            @param beta A momentum term/parameter (between 0 and 1)
            @return A reference to self
        */
        Network& updateNetwork(double learning_rate,  uint32_t batch_size, double lambda, std::string reg, double beta, bool change_rate, std::string id);
        /** Returns the cost function */
        Cost* getCost(){return loss;}
        /** Sets the update_rate */
        void setUpdateRate(bool value){update_rate = value;}
        /** Gets the update_rate */
        bool getUpdateRate(){return update_rate;}
        /** Saves parameters of network/model.

            @param model_file Name of the file to save model
        */
        void saveModel(std::string model_file);
        /** Loads network/model from an archive file.

            @param model_file Name of the model file
            @return A reference to self
        */
        Network& loadModel(std::string model_file);
        /** Virtual destructor */
        virtual ~Network(){delete loss;}
};
struct Model
{
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
		ar.register_type(static_cast<DenseLayer *>(NULL));
		ar.register_type(static_cast<SigmoidLayer *>(NULL));
		ar.register_type(static_cast<TanhLayer *>(NULL));
		ar.register_type(static_cast<ReLULayer *>(NULL));
		ar.register_type(static_cast<ELULayer *>(NULL));
		ar.register_type(static_cast<SoftmaxLayer *>(NULL));
		/*ar.template register_type<DenseLayer >();
		ar.template register_type<SigmoidLayer >();
		ar.template register_type<TanhLayer >();
		ar.template register_type<ReLULayer >();
		ar.template register_type<ELULayer >();
		ar.template register_type<SigmoidLayer >();*/
		//ar.template register_type< Layer >();
		//ar & boost::serialization::base_object<Layer>(*this);
        ar & layers;
    }
	public:
		std::vector<Layer*> layers;
		Model() = default;
		Model(std::vector<Layer*> &layers): layers(layers){}
};
} // namespace mlearn
#endif
