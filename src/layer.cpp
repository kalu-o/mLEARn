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
#include "layer.h"

namespace mlearn {
template <class T>
Layer<T>& Layer<T>::initialize()
{
    double lower = - sqrt(6.0/(input_dim + output_dim)) * 4;
    double upper = - (lower);
    using value_type = double;
    std::random_device r;
    std::uniform_real_distribution<value_type> distribution(
        lower,
        upper);
    static std::default_random_engine generator {r()};
    if(weight.size1() == 0) throw std::bad_alloc(); //("Matrix size cannot be zero!");
    for (uint64_t i = 0; i < output_dim; ++i)
    {
        for (uint64_t j = 0; j < input_dim; ++j)
            this->weight(i, j) = distribution(generator);
    }

    return *this;
}

template <class T>
void Layer<T>::connect(Layer<T>& layer)
{
    previous = &layer;
    layer.next = this;
}

template <class T>
NetNode<T>& Layer<T>::getForwardInput(NetNode<T>& out)
{
    if (previous != nullptr) out = previous->output_data;
    else out = input_data;
    return out;
}

template <class T>
NetNode<T>& Layer<T>::forwardProp()
{
    NetNode<T> data, out;
    getForwardInput(data);
    mublas::vector<T> dot_prod = mublas::prod(weight, data.getData());
    out.setData(dot_prod);
    activation = out + bias;
    out = act_function.compute(activation);
    setOutputData(out);
    return output_data;
}

template <class T>
NetNode<T>& Layer<T>::getBackwardInput(NetNode<T>& out)
{
    mublas::matrix<T> temp ;
    if (next != nullptr)
    {
        temp = mublas::trans(next->weight);
        mublas::vector<T> dot_prod = mublas::prod(temp, next->output_delta.getData());
        out.setData(dot_prod);
    }
    else out = input_delta;
    return out;
}

template <class T>
NetNode<T>& Layer<T>::backwardProp()
{
    NetNode<T> delta, data;
    getBackwardInput(delta);
    getForwardInput(data);
    if (act_function.getType() == "sigmoid") output_delta = delta * (act_function.computeDerivative(output_data));
    else if (act_function.getType() == "tanh") output_delta = delta * (act_function.computeDerivative(output_data));
    else if (act_function.getType() == "softmax")
    {
        mublas::matrix<T> temp(output_dim, output_dim);
        temp = act_function.computeDerivative(output_data, temp);
        mublas::vector<T> dot_prod = mublas::prod(temp, delta.getData());
        output_delta.setData(dot_prod);
    }
    else output_delta = delta * (act_function.computeDerivative(activation));
    delta_b = delta_b + output_delta;
    delta_w = delta_w + mublas::outer_prod(output_delta.getData(), data.getData());
    return output_delta;
}

template <class T>
Layer<T>& Layer<T>::regularize(double rate, double lambda, std::string type)
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

template <class T>
Layer<T>& Layer<T>::regularize(const mublas::matrix<T>& rate, double lambda, std::string type)
{
    if(type == "L1")
    {
        for(uint64_t i = 0; i < output_dim; ++i)
            for (uint64_t j = 0; j < input_dim; ++j)
            {
               regularize_w(i, j) = weight(i, j) >= 0 ? rate(i, j) * lambda : -rate(i, j) * lambda;
            }

    }
    else if (type == "L2")regularize_w = element_prod(weight, (rate * lambda));
    return *this;
}

template <class T>
Layer<T>& Layer<T>::clearDeltas(double beta)
{
    uint64_t data_size = delta_b.getDataSize();
    if (beta > 0.0) momentum_w = delta_w;
    mublas::zero_vector<T> zero_v(data_size);
    mublas::zero_matrix<T> zero_m(output_dim, input_dim);
    delta_b.setData(zero_v);
    delta_w = zero_m;

    return *this;
}

template <class T>
Layer<T>& Layer<T>::updateParams(
                    double rate,
                    uint32_t batch_size,
                    double lambda,
                    std::string reg,
                    double beta)
{
    rate/=batch_size;
    if(reg != "None") regularize(rate, lambda, reg);
    NetNode<T> out = delta_b.scalarMultiply(rate);
    bias = bias - out;
    if (reg != "None" && beta > 0.0) weight = weight - (delta_w * rate) - regularize_w - momentum_w * beta;
    else if (reg == "None" && beta > 0.0) weight = weight - (delta_w * rate) - momentum_w * beta;
    else if (reg != "None" && beta == 0.0) weight = weight - (delta_w * rate) - regularize_w;
    else weight = weight - (delta_w * rate);
    return *this;
}

template <class T>
Layer<T>& Layer<T>::updateParams(
                    double rate,
                    uint32_t batch_size,
                    double lambda,
                    std::string reg,
                    double beta,
                    bool change_rate,
                    std::string id)
{
    rate/=batch_size;
    mublas::matrix<T> adagrad_rates(output_dim, input_dim, rate);
    NetNode<T> out = delta_b.scalarMultiply(rate);
    bias = bias - out;
    if (id == "rmsprop")
    {
        double mu = 0.9;
        sq_delta_w = mu * sq_delta_w + (1 - mu) * element_prod(delta_w, delta_w);
    }
    else if (id == "adagrad")
    {
        sq_delta_w += element_prod(delta_w, delta_w);
    }

    if (change_rate)
    {
        for(uint64_t i = 0; i < output_dim; ++i)
            for (uint64_t j = 0; j < input_dim; ++j) adagrad_rates(i, j) = rate * 1/(std::sqrt(sq_delta_w(i, j) + ADAGRAD_EPSILON));
    }
    if(reg != "None") regularize(adagrad_rates, lambda, reg);
    if (reg != "None" && beta > 0.0) weight = weight - element_prod(delta_w, adagrad_rates) - regularize_w -  momentum_w * beta;  // momentum_w * beta * rate;
    else if (reg == "None" && beta > 0.0) weight = weight - element_prod(delta_w, adagrad_rates) - momentum_w * beta;
    else if (reg != "None" && beta == 0.0) weight = weight - element_prod(delta_w, adagrad_rates) - regularize_w;
    else weight = weight - element_prod(delta_w, adagrad_rates);

    return *this;
}
template class Layer<double>;
} // namespace mlearn
