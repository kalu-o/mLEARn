#ifndef LAYERS_H
#define LAYERS_H
#include <iostream>
#include <vector>
#include "utils.h"

namespace mlearn {
class Layer
{
     protected:
		/** Input dimension of layer */
		uint64_t input_dim;
		/** Output dimension of layer */
		uint64_t output_dim;
		 /** Weight matrix of layer */
        /** Pointer to previous layer */
        Layer* previous;
        /** Pointer to next layer */
        Layer* next;
        /** Output data of layer */
        mublas::vector<double> output_data;
        /** Output delta of layer */
        mublas::vector<double> output_delta;
        /** Input data of layer */
        mublas::vector<double> input_data;
        /** Input delta of layer */
        mublas::vector<double> input_delta;
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
            }
        }
	public:
        /** Default constructor */
        Layer(): previous{nullptr}, next{nullptr}{}
		Layer(uint64_t input_dim, uint64_t output_dim): input_dim{input_dim}, output_dim{output_dim}, previous{nullptr}, next{nullptr}{}
        /**
            Connects a layer to its neighbors.

            @param layer Layer to be connected
            @param in_data Vector of data
            @return Returns void
        */
        void connect(Layer* layer);

		 /**
            An input data is propagated forward through a layer
            and produces an output data.

            @return output_data Output data
        */
         //virtual mublas::vector<double>& forwardProp() = 0;
		 virtual mublas::vector<double>& forwardProp() {return input_data;}
		/**
            The first layer gets input data from the training data,
            while other layers get their inputs from the output of
            previous layer.

            @param out An empty NetNode
            @return out Input data
        */
		 mublas::vector<double>& getForwardInput(mublas::vector<double>& out);
		 /**
            Propagates input error (delta) backward and produces an output delta.

            @return output_delta Output delta
        */
         //virtual mublas::vector<double>& backwardProp() = 0;
		 virtual mublas::vector<double>& backwardProp() {return output_delta;}
        /**
            The last layer gets input delta from the derivative of
            the cost function, while other layers get their input deltas
            from their successors.

            @return out Input delta
        */
         mublas::vector<double>& getBackwardInput(mublas::vector<double>& out);
		  /** Sets the input data to a layer */
        void setInputData(const mublas::vector<double>& in_data){input_data = in_data;}
        /** Sets the output data from a layer */
        void setOutputData(const mublas::vector<double>& out_data){output_data = out_data;}
        /** Sets the input delta to a layer */
        void setInputDelta(const mublas::vector<double>& in_delta){input_delta = in_delta;}
        /** Sets the output delta from a layer */
        void setOutputDelta(const mublas::vector<double>& out_delta){output_delta = out_delta;}
		 /** Returns the input data to a layer in a NetNode object */
        mublas::vector<double>& getInputData(){return input_data;}
        /** Returns the output data from a layer in a NetNode object */
        mublas::vector<double>& getOutputData(){return output_data;}
        /** Returns the input delta to a layer in a NetNode object */
        mublas::vector<double>& getInputDelta(){return input_delta;}
        /** Returns the output delta from a layer in a NetNode object */
        mublas::vector<double>& getOutputDelta(){return output_delta;}
        /** Virtual destructor */
        virtual ~Layer(){}
};
BOOST_SERIALIZATION_ASSUME_ABSTRACT(Layer)
class DenseLayer : public Layer
{
     protected:
        mublas::matrix<double> weight;
		/** Bias of layer */
        mublas::vector<double> bias;
		/** Delta of weight matrix */
        mublas::matrix<double> delta_w;
        /** Delta of bias */
        mublas::vector<double> delta_b;
		/** Weight matrix used for regularization */
        mublas::matrix<double> regularize_w;
		/** Weight matrix used for momentum */
        mublas::matrix<double> momentum_w;
		  /** Accumulates previous deltas. Used in Adagrad and RMSProp */
        mublas::matrix<double> learning_rate_buffer;
		mublas::matrix<double> adapt_rates;
		mublas::vector<double> tied_rates;
		/** Responsible for saving/serialization of members */
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const uint64_t version)
        {
            if (version >= 0)
            {
                 // save/load base class information
				ar.template register_type< Layer >();
				ar & boost::serialization::base_object<Layer>(*this);
				ar & weight;
				ar & bias;
				ar & delta_w;
                ar & delta_b;                
                //ar & regularize_w;
                //ar & momentum_w;
                //ar & learning_rate_buffer;
                //ar & type;
                //ar & act_function;
                //ar & psdsquare;
            }
        }

	public:
		//zero_matrix<double> m (3, 3);
		/** Default constructor */
        DenseLayer(): Layer(0, 0){}
        /** Overloaded constructor with 4 arguments */
        DenseLayer(uint64_t input_dim, uint64_t output_dim):
			Layer(input_dim, output_dim),
			 weight(output_dim, input_dim),
             bias{output_dim},
			 delta_w(output_dim, input_dim, 0),
             delta_b{output_dim},
             regularize_w(output_dim, input_dim, 0),
			 momentum_w(output_dim, input_dim, 0),
			 learning_rate_buffer(output_dim, input_dim, 0.1),
			 adapt_rates(output_dim, input_dim, 0.1),
			 tied_rates{output_dim}
        {initialize(); }
		mublas::vector<double>& forwardProp();
		mublas::vector<double>& backwardProp();
		DenseLayer& initialize();
		/** Returns the weight matrix of a layer */
        const mublas::matrix<double>& getWeight()const{return weight;}
        /** Returns the bias vector of a layer in a NetNode object */
        const mublas::vector<double>& getBias()const{return bias;}
        /** Returns the delta weight matrix of a layer */
        const mublas::matrix<double>& getDeltaWeight()const{return delta_w;}
        /** Returns the delta bias vector of a layer in a NetNode object */
        const mublas::vector<double>& getDeltaBias()const{return delta_b;}
        /** Sets the weight matrix of a layer */
        void setWeight(const mublas::matrix<double>& in_weight){weight = in_weight;}
        /** Sets the bias vector of a layer */
        void setBias(const mublas::vector<double>& in_bias){bias = in_bias;}
        /** Sets the delta weight matrix of a layer */
        void setDeltaWeight(const mublas::matrix<double>& in_delta_w){delta_w = in_delta_w;}
        /** Sets the bias vector of a layer */
        void setDeltaBias(const mublas::vector<double>& in_delta_b){delta_b = in_delta_b;}

        /**
            After a single forward and backward pass, the parameters of
            each layer is updated, and the deltas reset to zero. Beta is
            a momentum to use to decide if a part of a previous weight
            update should be used. default is 0 (no momentum).

            @param beta Momentum term/parameter
            @return Reference to self
        */
        DenseLayer& clearDeltas(double beta);
        /**
            Overloaded function. Handles regularization of layer
            parameters (weight). Currently, only L1 and L2 are
            implemented. The default is no regularization(None).

            @param rate Learning rate
            @param lambda Regularization parameter
            @param reg Type of regularization (L1, L2 or None)
            @return Reference to self
        */
		DenseLayer& regularize(double rate, double lambda, std::string type);
        DenseLayer& updateParams(double rate, uint32_t batch_size, double lambda, std::string reg, double beta, bool change_rate, std::string id);
        /** Virtual destructor */
        virtual ~DenseLayer(){}
};
class SigmoidLayer : public Layer
{
     protected:
		std::string type;
		/** Responsible for saving/serialization of members */
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const uint64_t version)
        {
            if (version >= 0)
            {
                 // save/load base class information
				ar.template register_type< Layer >();
				ar & boost::serialization::base_object<Layer>(*this);
				ar & type;
            }
        }
		
	public:
		/** Default constructor */
        SigmoidLayer(): Layer(0, 0), type{"sigmoid"}{}
		SigmoidLayer(uint64_t input_dim, uint64_t output_dim): Layer(input_dim, output_dim), type{"sigmoid"}{}
		mublas::vector<double>& forwardProp();
		mublas::vector<double>& backwardProp();
		virtual ~SigmoidLayer(){}
};
class TanhLayer : public Layer
{
     protected:
		std::string type;
		/** Responsible for saving/serialization of members */
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const uint64_t version)
        {
            if (version >= 0)
            {
                 // save/load base class information
				ar.template register_type< Layer >();
				ar & boost::serialization::base_object<Layer>(*this);
				ar & type;
            }
        }

	public:
		/** Default constructor */
        TanhLayer(): Layer(0, 0), type{"tanh"}{}
		TanhLayer(uint64_t input_dim, uint64_t output_dim): Layer(input_dim, output_dim), type{"tanh"}{}
		mublas::vector<double>& forwardProp();
		mublas::vector<double>& backwardProp();
		virtual ~TanhLayer(){}
};
class ReLULayer : public Layer
{
     protected:
		std::string type;
		double alpha;
	/** Responsible for saving/serialization of members */
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const uint64_t version)
        {
            if (version >= 0)
            {
                 // save/load base class information
				ar.template register_type< Layer >();
				ar & boost::serialization::base_object<Layer>(*this);
				ar & type;
				ar & alpha;
            }
        }

	public:
		/** Default constructor */
        ReLULayer(): Layer(0, 0), type{"relu"}, alpha{0.0}{}
		ReLULayer(uint64_t input_dim, uint64_t output_dim): Layer(input_dim, output_dim), type{"relu"}, alpha{0.0}{}
		mublas::vector<double>& forwardProp();
		mublas::vector<double>& backwardProp();
		virtual ~ReLULayer(){}
};
class ELULayer : public Layer
{
     protected:
		std::string type;
		double alpha;
	/** Responsible for saving/serialization of members */
        friend class boost::serialization::access;
        template<class Archive>
	void serialize(Archive & ar, const uint64_t version)
        {
            if (version >= 0)
            {
                 // save/load base class information
				ar.template register_type< Layer >();
				ar & boost::serialization::base_object<Layer>(*this);
				ar & type;
				ar & alpha;
            }
        }

	public:
		/** Default constructor */
        ELULayer(): Layer(0, 0), type{"elu"}, alpha{0.0}{}
		ELULayer(uint64_t input_dim, uint64_t output_dim): Layer(input_dim, output_dim), type{"elu"}, alpha{0.0}{}
		mublas::vector<double>& forwardProp();
		mublas::vector<double>& backwardProp();
		virtual ~ELULayer(){}
};
class SoftmaxLayer : public Layer
{
     protected:
		std::string type;
	/** Responsible for saving/serialization of members */
        friend class boost::serialization::access;
        template<class Archive>
	void serialize(Archive & ar, const uint64_t version)
        {
            if (version >= 0)
            {
                 // save/load base class information
				ar.template register_type< Layer >();
				ar & boost::serialization::base_object<Layer>(*this);
				ar & type;
            }
        }

	public:
		/** Default constructor */
        SoftmaxLayer(): Layer(0, 0), type{"softmax"}{}
		SoftmaxLayer(uint64_t input_dim, uint64_t output_dim): Layer(input_dim, output_dim), type{"softmax"}{}
		mublas::vector<double>& forwardProp();
		mublas::vector<double>& backwardProp();
		virtual ~SoftmaxLayer(){}
};
//BOOST_SERIALIZATION_ASSUME_ABSTRACT(Layer)
//BOOST_CLASS_EXPORT_GUID(DerivedOne, “DOne”)
//BOOST_CLASS_EXPORT_GUID(DenseLayer, "denselayer");
//BOOST_CLASS_EXPORT(SigmoidLayer)
//BOOST_CLASS_EXPORT(TanhLayer)
//BOOST_CLASS_EXPORT(ELULayer)
//BOOST_CLASS_EXPORT(ReLULayer)

} // namespace mlearn
#endif
