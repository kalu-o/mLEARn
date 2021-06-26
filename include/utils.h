#ifndef UTILS_H
#define UTILS_H
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>

#include <boost/serialization/base_object.hpp>
//#include <boost/serialization/utility.hpp>
#include <boost/serialization/assume_abstract.hpp>

#include <boost/serialization/export.hpp>  
#include <boost/serialization/access.hpp>
#include <boost/serialization/serialization.hpp>

#include <stdlib.h>
#include <exception>
#include <iomanip>
#include <stdexcept>
#include <chrono>
#include <ctime>
#include <random>
#include <cmath>
#define EPSILON 1e-2

namespace mublas = boost::numeric::ublas;
//typedef  mublas::vector<mublas::vector<double>> vec_vec_double;
typedef std::vector<mublas::vector<double>*> vec_vec_ptr_double;
typedef mublas::vector<double> b_vec_double;
typedef mublas::matrix<double> b_mat_double;
typedef b_mat_double* ptr_matrix;
typedef std::vector<ptr_matrix> vec_ptr_matrix;
typedef std::vector<vec_ptr_matrix> vec_vec_ptr_matrix;
namespace mlearn {
/**
    The CostFunction class is responsible for objective functions.
    Cost functions implemented are mean squared error (MSE),
    mean absolute error (MAE) and cross entropy.
*/
class Cost
{
    protected:
        /** Type of cost function */
        std::string id;
    public:
        /** Default constructor */
        Cost(){}
        /** Constructor that takes a single parameter: id */
        Cost(std::string id): id{id}{}
        /** Accessor function that returns the id of the cost function */
        std::string getId()const{return id;}
        /** Pure virtual function */
        virtual double costFunction(const mublas::vector<double>& prediction, const mublas::vector<double>& label) = 0;
        /** Pure virtual function */
        virtual double accuracy(const mublas::vector<double>& prediction, const mublas::vector<double>& label) = 0;
		/** Pure virtual function */
        virtual mublas::vector<double>& costDerivative(const mublas::vector<double>&, const mublas::vector<double>&, mublas::vector<double>&) = 0;
        /** Virtual destructor */
        virtual ~Cost(){}
};
/**
    The MSE (mean squared error) derives from the class CostFunction. It implements
    "cost" "accuracy" and "costDerivative" functions. Used for regression problems.
*/
class MSE : public Cost
{
    public:
        /** Default constructor */
        MSE(): Cost("mse"){}
        /** Computes the error between the prediction and label.
            This is the square of the difference between the prediction
            and label.

            @param prediction Hypotheses
            @param label Reference
            @return The square of the difference between the prediction and label
        */
        double costFunction(const mublas::vector<double>& prediction, const mublas::vector<double>& label);
        /** Computes the error between the prediction and label.
            Calls the cost function.

            @param prediction Hypotheses
            @param label Reference
            @return The square of the difference between the prediction and label
        */
        double accuracy(const mublas::vector<double>& prediction, const mublas::vector<double>& label) {return costFunction(prediction, label);}
        /** Computes the derivative of MSE. This is the difference
            between the prediction and label.

            @param prediction Hypotheses
            @param label Reference
            @param result Result
            @return The difference between the prediction and label
        */
        mublas::vector<double>& costDerivative(const mublas::vector<double>& prediction, const mublas::vector<double>& label, mublas::vector<double>& result);
};
/**
    CrossEntropy derives from the class CostFunction.
    It implements the "cost", "accuracy" and "costDerivative" functions.
    Used for classification problems.
*/
class CrossEntropy : public Cost
{
    public:
        /** Default constructor */
        CrossEntropy(): Cost("crossentropy"){}
        /** Computes the loss between prediction and label.
            This is the negative of product of label and log of prediction.

            @param prediction Hypotheses
            @param label Reference
            @return Product of negative log of prediction and label
        */
        double costFunction(const mublas::vector<double>& prediction, const mublas::vector<double>& label);
        /** Computes the accuracy between prediction and label.
            This is +1 if prediction and label are same and 0 otherwise.

            @param prediction Hypotheses
            @param label Reference
            @return +1 if prediction and label are same and 0 otherwise
        */
        double accuracy(const mublas::vector<double>& prediction, const mublas::vector<double>& label);
        /** Computes the derivative. This is implemented as the difference
            between the prediction and label.

            @param prediction Hypotheses
            @param label Reference
            @param result Result
            @return The difference between the prediction and label
        */
        mublas::vector<double>& costDerivative(const mublas::vector<double> &prediction, const mublas::vector<double> &label, mublas::vector<double>& result);

};
/**
    The MAE(mean absolute error) derives from the class CostFunction.
    It implements the "cost", "accuracy" and "costDerivative" functions.
    Used for regression problems.
*/
class MAE : public Cost
{
    public:
        /** Default constructor */
        MAE(): Cost("mae"){}
        /** Computes the absolute error between the prediction and label.

            @param prediction Hypotheses
            @param label Reference
            @return The absolute difference between the prediction and label
        */
        double costFunction(const mublas::vector<double>& prediction, const mublas::vector<double>& label);
        /** Computes the absolute error between prediction and label.
            Calls the cost function.

            @param prediction Hypotheses
            @param label Reference
            @return The absolute difference between the prediction and label
        */
        double accuracy(const mublas::vector<double>& prediction, const mublas::vector<double>& label) {return costFunction(prediction, label);}
        /** Computes the derivative of MAE.

            @param prediction Hypotheses
            @param label Reference
            @param result Result
            @return Derivative of MAE
        */
        mublas::vector<double>& costDerivative(const mublas::vector<double>& prediction, const mublas::vector<double>& label, mublas::vector<double>& result);

};

/**
	Computes the sigmoid (activation) of values of "data".

	@pre None
	@post Modifies "data" values.
	@return A reference to self (with sigmoid of "data" values)
*/
mublas::vector<double>& sigmoid(mublas::vector<double>& argv);
/**
	Computes the derivative of sigmoid. The derivative of sigmoid
	is \f{equation}{f(x) * (1 - f(x)) \f}.

	@return A reference to self (with sigmoid derivative of "data" values)
*/
mublas::vector<double> sigmoidPrime(mublas::vector<double>& argv);
/**
	Computes the tanh (activation) of values of "data".

	@pre None
	@post Modifies "data" values.
	@return A reference to self (with tanh of "data" values)
*/
mublas::vector<double>& hyperTan(mublas::vector<double>& argv);
/**
	Computes the derivative of tanh. The derivative of tanh
	is \f{equation}{1 - (f(x) * f(x)) \f}.

	@return A reference to self (with tanh derivative of "data" values)
*/
mublas::vector<double>& hyperTanPrime(mublas::vector<double>& argv);
/**
	Computes the Rectified linear Unit (ReLU) activation: max(0, x).

	@pre None
	@post Modifies "data" values.
	@return A reference to self (with ReLU of "data" values)
*/
mublas::vector<double>& ReLU(mublas::vector<double>& argv);
/**
	The derivative of ReLU. This becomes a leaky ReLU if "alpha"
	is greater than 0.

	@param alpha A parameter used for leaky ReLU
	@return A reference to self (with the derivative of ReLU of "data" values)
*/
mublas::vector<double>& ReLUPrime(mublas::vector<double>& argv, double alpha);
/**
	Computes softmax activation function; forces sum of probabilities to 1.

	@pre None
	@post Modifies "data" values.
	@return A reference to self (with softmax of "data" values)
*/
mublas::vector<double>& softmax(mublas::vector<double>& argv);
/**
	Computes derivative of softmax.

	@param argv An empty 2D Jacobian matrix
	@return A reference to argv
*/
mublas::matrix<double>& softmaxPrime(mublas::vector<double>& argv1, mublas::matrix<double>& argv2);
/**
	Computes exponential linear unit (ELU) activation.

	@param alpha A parameter used in ELU
	@return A reference to self (with ELU of "data" values)
*/
mublas::vector<double>& ELU(mublas::vector<double>& argv, double alpha);
/**
	Computes the derivative of ELU.

	@param alpha A parameter used in ELU
	@return A reference to self
*/
mublas::vector<double>& ELUPrime(mublas::vector<double>& argv, double alpha);
/** The identity function. Does not modify "data" */
mublas::vector<double>& identity(mublas::vector<double>& argv);
/** The derivative of identity function */
mublas::vector<double>& identityPrime(mublas::vector<double>& argv);
/*vec_ptr_matrix& reshape(b_vec_double &input, vec_ptr_matrix &output, uint8_t num_channels, uint8_t image_dim);
b_vec_double& reshape(vec_ptr_matrix &input, b_vec_double &output);
vec_ptr_matrix& convolve(vec_ptr_matrix &input, vec_ptr_matrix &output, vec_ptr_matrix &delta, uint8_t stride);
vec_ptr_matrix& convolve(vec_ptr_matrix &input, vec_ptr_matrix &output, vec_ptr_matrix &filter, uint8_t num_channels, uint8_t stride);
vec_ptr_matrix& convBackward(vec_ptr_matrix &input, vec_ptr_matrix &output, ptr_matrix delta, uint8_t stride);
ptr_matrix convForward(vec_ptr_matrix &input, vec_ptr_matrix &filter, ptr_matrix output, uint8_t stride);
vec_ptr_matrix& destroy(vec_ptr_matrix &argv);
ptr_matrix& addBias(ptr_matrix &argv, double bias);
double findMax(ptr_matrix argv); */
} // namespace mlearn
#endif
