/*
IT License

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
Change Log: 21.05.2019 - Version 1.0.1
Change Log: 01.04.2019 - Version 1.0.0
*/
#include "layers.h"

namespace mlearn {

void Layer::connect(Layer* layer)
{
    previous = layer;
    layer->next = this;
}

mublas::vector<double>& Layer::getForwardInput(mublas::vector<double>& out)
{
    if (previous != nullptr) out = previous->output_data;
    else out = input_data;
    return out;
}
mublas::vector<double>& Layer::getBackwardInput(mublas::vector<double>& out)
{
    if (next != nullptr) out = next->output_delta;
    else out = input_delta;
    return out;
}

mublas::vector<double>& DenseLayer::forwardProp()
{
    mublas::vector<double> data;
	data = getForwardInput(data);
    output_data = mublas::prod(weight, data) + bias;
	return output_data;
}
mublas::vector<double>& DenseLayer::backwardProp()
{
    mublas::vector<double> delta, data;
    delta = getBackwardInput(delta);
    data = getForwardInput(data);
	delta_b += delta;
	delta_w += mublas::outer_prod(delta, data);
	mublas::matrix<double> trans_weight = mublas::trans(weight);
	output_delta = mublas::prod(trans_weight, delta);
    return output_delta;
}

DenseLayer& DenseLayer::initialize()
{
    double lower = - sqrt(6.0/(input_dim + output_dim)) * 4;
    double upper = - (lower);
    using value_type = double;
    std::random_device r;
    std::uniform_real_distribution<value_type> distribution(lower, upper);
    static std::default_random_engine generator {r()};
    if(weight.size1() == 0) throw std::bad_alloc(); //("Matrix size cannot be zero!");
    for (uint64_t i = 0; i < output_dim; ++i)
    {
        for (uint64_t j = 0; j < input_dim; ++j)
            weight(i, j) = distribution(generator);
    }
    return *this;
}

DenseLayer& DenseLayer::regularize(double rate, double lambda, std::string type)
{
    if(type == "L1")
    {
        for(uint64_t i = 0; i < output_dim; ++i)
            for (uint64_t j = 0; j < input_dim; ++j)
            {
               regularize_w(i, j) = weight(i, j) >= 0 ? rate * lambda : -rate * lambda;
            }

    }
    else if (type == "L2")regularize_w = weight * (rate * lambda);
    return *this;
}

DenseLayer& DenseLayer::clearDeltas(double beta)
{
    uint64_t data_size = delta_b.size();
	if (beta > 0.0) momentum_w += delta_w;
    mublas::zero_vector<double> zero_v(data_size);
    mublas::zero_matrix<double> zero_m(output_dim, input_dim);
    delta_b = zero_v;
    delta_w = zero_m;
    return *this;
}

DenseLayer& DenseLayer::updateParams(
                    double rate,
                    uint32_t batch_size,
                    double lambda,
                    std::string reg,
					double beta, 
					bool change_rate,
                    std::string id
					)
{
    double mu = 0.9;
	double epsilon = 1e-9;
	//mublas::matrix<double> adapt_rates(output_dim, input_dim, rate);
	mublas::matrix<double> delta(output_dim, input_dim);
	delta_w /= batch_size;
	delta_b /= batch_size;
	
    
    if (id == "adagrad" && change_rate)
	{
		learning_rate_buffer += element_prod(delta_w, delta_w);
		//std::cout<<"yes"<<std::endl;
		for(uint64_t i = 0; i < output_dim; ++i)
        {
			for (uint64_t j = 0; j < input_dim; ++j) adapt_rates(i, j) = rate * 1/(std::sqrt(learning_rate_buffer(i, j)) + epsilon);
		}
	}
	else if (id == "rmsprop" && change_rate)
	{
		learning_rate_buffer = mu * learning_rate_buffer + (1 - mu) * element_prod(delta_w, delta_w);
		for(uint64_t i = 0; i < output_dim; ++i)
        {
			for (uint64_t j = 0; j < input_dim; ++j) adapt_rates(i, j) = rate * 1/(std::sqrt(learning_rate_buffer(i, j)) + epsilon);
		}
	}
	else if (id == "psdsquare" && change_rate)
    {
		learning_rate_buffer = mu * learning_rate_buffer + (1 - mu) * element_prod(delta_w, delta_w);
		double sum_rates = 0;
		for(uint64_t i = 0; i < output_dim; ++i)
        {
			for (uint64_t j = 0; j < input_dim; ++j) sum_rates += rate * 1/(std::sqrt(learning_rate_buffer(i, j)) + epsilon);
		}
		rate = sum_rates/(input_dim * output_dim);
		
    }
	
	
	
	/*if (reg != "None" && beta > 0.0) weight = weight - element_prod(delta_w, adagrad_rates) - regularize_w -  momentum_w * beta;  // momentum_w * beta * rate;
    else if (reg == "None" && beta > 0.0) weight = weight - element_prod(delta_w, adagrad_rates) - momentum_w * beta;
    else if (reg != "None" && beta == 0.0) weight = weight - element_prod(delta_w, adagrad_rates) - regularize_w;
    else weight = weight - element_prod(delta_w, adagrad_rates);*/
	
    if(reg != "None") regularize(rate, lambda, reg);
	if ((id == "adagrad") || (id == "rmsprop"))delta = element_prod(delta_w, adapt_rates) ;
	else delta = delta_w * rate;
	
    if (reg != "None" && beta > 0.0) weight = weight - delta  - regularize_w - momentum_w * beta;
    else if (reg == "None" && beta > 0.0) weight = weight  - delta  - momentum_w * beta;
    else if (reg != "None" && beta == 0.0) weight = weight - delta - regularize_w;
    else weight -= delta;
	bias -= (delta_b * rate);
    return *this;
}
mublas::vector<double>& SigmoidLayer::forwardProp()
{
    mublas::vector<double> data;
	data = getForwardInput(data);
    output_data = sigmoid(data);
	return output_data;
}
mublas::vector<double>& SigmoidLayer::backwardProp()
{
    mublas::vector<double> delta, data;
    delta = getBackwardInput(delta);
    //data = getForwardInput(data);
	output_delta = element_prod(delta, sigmoidPrime(output_data));
    return output_delta;
}
mublas::vector<double>& TanhLayer::forwardProp()
{
    mublas::vector<double> data;
	data = getForwardInput(data);
    output_data = hyperTan(data);
	return output_data;
}
mublas::vector<double>& TanhLayer::backwardProp()
{
    mublas::vector<double> delta;
    delta = getBackwardInput(delta);
    //data = getForwardInput(data);
	output_delta = element_prod(delta, hyperTanPrime(output_data));
    return output_delta;
}
mublas::vector<double>& ReLULayer::forwardProp()
{
    mublas::vector<double> data;
	data = getForwardInput(data);
    output_data = ReLU(data);
	return output_data;
}
mublas::vector<double>& ReLULayer::backwardProp()
{
    mublas::vector<double> delta, data;
    delta = getBackwardInput(delta);
    data = getForwardInput(data);
	output_delta = element_prod(delta, ReLUPrime(data, alpha));
    return output_delta;
}
mublas::vector<double>& ELULayer::forwardProp()
{
    mublas::vector<double> data;
	data = getForwardInput(data);
    output_data = ELU(data, alpha);
	return output_data;
}
mublas::vector<double>& ELULayer::backwardProp()
{
    mublas::vector<double> delta, data;
    delta = getBackwardInput(delta);
    data = getForwardInput(data);
	output_delta = element_prod(delta, ELUPrime(data, alpha));
    return output_delta;
}
mublas::vector<double>& SoftmaxLayer::forwardProp()
{
    mublas::vector<double> data;
	data = getForwardInput(data);
    output_data = softmax(data);
	return output_data;
}
mublas::vector<double>& SoftmaxLayer::backwardProp()
{
    mublas::vector<double> delta, data;
    delta = getBackwardInput(delta);
    //data = getForwardInput(data);
	uint64_t data_size = output_data.size();
	mublas::matrix<double> temp(data_size, data_size);
	temp = softmaxPrime(output_data, temp);
	output_delta = mublas::prod(temp, delta);
    return output_delta;
}
} // namespace mlearn
