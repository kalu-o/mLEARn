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
#include <iostream>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <math.h>
#include "libutil.h"

namespace mlearn {
template <class T>
Node<T> Node<T>::operator+(const Node<T>& argv)
{
    if(data_size != argv.data_size) throw std::length_error("The data_size of the Nodes must be equal! (operator+)");
    mublas::vector<T> out = getData() + argv.getData();
    return out;
}

template <class T>
Node<T> Node<T>::operator-(const Node<T>& argv)
{
    //std::cout<<data_size <<" " <<argv.data_size <<std::endl;
    if(data_size != argv.data_size) throw std::length_error("The data_size of the Nodes must be equal! (operator-)");
    mublas::vector<T> out = getData() - argv.getData();
    return out;
}

template <class T>
Node<T> Node<T>::operator*(const Node<T>& argv)
{
    if(data_size != argv.data_size) throw std::length_error("The data_size of the Nodes must be equal! (operator*)");
    mublas::vector<T> out = element_prod(getData(), argv.getData());
    return out;
}

template <class T>
Node<T>& Node<T>::operator=(const Node<T>& argv)
{
    if(this != &argv)
    {
        data = argv.data;
        data_size = argv.data_size;
    }
    return *this;
}

template <class T>
Node<T> Node<T>::scalarMultiply(double rate)
{
    mublas::vector<T> out = getData() * rate;
    return out;
}

template <class T>
double Node<T>::sum()const
{
    auto result = mublas::sum(getData());
    return result;
}

template <class T>
double Node<T>::fabsSum()const
{
    auto result = 0.0;
    mublas::vector<T> temp_data = getData();
    for(uint64_t i = 0; i < data_size; ++i) result += fabs(temp_data[i]);
    return result;
}

template <class T>
Node<T>& Node<T>::compare(const Node<T>& argv, Node<T>& result)const
{
    mublas::vector<T> temp(data_size);
    for(uint64_t i = 0; i < data_size; ++i)
    {
        if (data[i] < argv.data[i])temp[i] = -1.0;
        else if (data[i] > argv.data[i])temp[i] = 1.0;
        else temp[i] = 0.0;
    }
    result.data = temp;
    return result;
}

template <class T>
NetNode<T>& NetNode<T>::operator=(const NetNode& argv)
{
    if(this != &argv)
    {
        this->data = argv.data;
        this->data_size = argv.data_size;
    }
    return *this;
}

template <class T>
NetNode<T>& NetNode<T>::sigmoid()
{

    for(uint64_t i = 0; i < this->data_size; ++i)
    {
        this->data[i] = 1.0 / (1.0 + exp(-this->data[i]));
    }

    return *this;
}

template <class T>
NetNode<T> NetNode<T>::sigmoidPrime()
{

    NetNode<T> temp = *this, out = (temp * temp);
    return temp - out;
}

template <class T>
NetNode<T>& NetNode<T>::hyperTan()
{
    for(uint64_t i = 0; i < this->data_size; ++i) this->data[i] = tanh(this->data[i]);
    return *this;
}

template <class T>
NetNode<T>& NetNode<T>::hyperTanPrime()
{
    for(uint64_t i = 0; i < this->data_size; ++i) this->data[i] = (1 - this->data[i] * this->data[i]);
    return *this;
}

template <class T>
NetNode<T>& NetNode<T>::ReLU()
{
    for(uint64_t i = 0; i < this->data_size; ++i) this->data[i] = std::max(0.0, this->data[i]);
    return *this;
}

template <class T>
NetNode<T>& NetNode<T>::ReLUPrime(double alpha)
{
    for(uint64_t i = 0; i < this->data_size; ++i)this->data[i] = this->data[i] < 0.0 ? alpha : 1.0;
    return *this;
}

template <class T>
NetNode<T>& NetNode<T>::ELU(double alpha)
{
    for(uint64_t i = 0; i < this->data_size; ++i) this->data[i] = this->data[i] > 0 ? this->data[i] : alpha * (exp(this->data[i]) - 1);
    return *this;
}

template <class T>
NetNode<T>& NetNode<T>::ELUPrime(double alpha)
{
    for(uint64_t i = 0; i < this->data_size; ++i)this->data[i] = this->data[i] > 0.0 ? 1.0 : alpha * exp(this->data[i]);
    return *this;
}

template <class T>
NetNode<T>& NetNode<T>::identity()
{
    return *this;
}

template <class T>
NetNode<T>& NetNode<T>::identityPrime()
{
    for(uint64_t i = 0; i < this->data_size; ++i) this->data[i] = 1.0;
    return *this;
}

template <class T>
NetNode<T>& NetNode<T>::softmax()
{
    double t_sum = 0.0, max_e = *max_element(this->data.begin(), this->data.end());
    for(uint64_t i = 0; i < this->data_size; ++i)
    {
        this->data[i] = exp(this->data[i] -= max_e);
        t_sum += this->data[i];
    }
    for(uint64_t i = 0; i < this->data_size; ++i)this->data[i] = this->data[i]/t_sum;
    return *this;
}

template <class T>
mublas::matrix<T>& NetNode<T>::softmaxPrime(mublas::matrix<T>& argv)
{

    for(uint64_t i = 0; i < this->data_size; ++i)
        for(uint64_t j = 0; j < this->data_size; ++j)
        {
            if (i == j) argv(i, j) = this->data[i] * (1 - this->data[i]);
            else argv(i, j) = -this->data[i] * this->data[j];
        }

    return argv;
}

template <class T>
NetNode<T>& NetNode<T>::log_e()
{
    for(uint64_t i = 0; i < this->data_size; ++i) this->data[i] = log(this->data[i]);
    return *this;
}

template <class T>
Node<T>& Node<T>::generateRandomData(uint64_t input_dim, uint64_t output_dim)
{
    double lower = - sqrt(6.0/(input_dim + output_dim)) * 4;
    double upper = - (lower);
    using value_type = double;
    std::random_device r;
    static std::uniform_real_distribution<value_type> distribution(
        lower,
        upper);
    static std::default_random_engine generator {r()};
    std::generate(data.begin(), data.end(), []() { return distribution(generator); });

    return *this;
}

template <class T>
NetNode<T>& Activation<T>::compute(NetNode<T>& input)
{
    if (type == "sigmoid") input = input.sigmoid();
    else if (type == "tanh")input= input.hyperTan();
    else if (type == "relu")input = input.ReLU();
    else if (type == "softmax")input = input.softmax();
    else if (type == "elu")input = input.ELU(alpha);
    else if (type == "identity")input = input.identity();
    else
    {
        std::cerr << "Unknown activation function!" << std::endl;
        exit(EXIT_FAILURE);
    }
    return input;
}

template <class T>
NetNode<T>& Activation<T>::computeDerivative(NetNode<T>& input)
{
    if (type == "sigmoid") input = input.sigmoidPrime();
    else if (type == "tanh")input = input.hyperTanPrime();
    else if (type == "relu")input = input.ReLUPrime(alpha);
    else if (type == "elu")input = input.ELUPrime(alpha);
    else if (type == "identity")input = input.identityPrime();
    return input;
}

template <class T>
mublas::matrix<T>& Activation<T>::computeDerivative(NetNode<T>& input, mublas::matrix<T>& jacobian_m)
{
    if (type == "softmax") jacobian_m = input.softmaxPrime(jacobian_m);
    return jacobian_m;
}

template <class T>
double MSE<T>::cost(const NetNode<T>& prediction, const NetNode<T>& label)
{
    double error = 0.0;
    NetNode<T> out = prediction;
    out = out - label;
    error = mublas::inner_prod(out.getData(), out.getData());
    return error;
}

template <class T>
NetNode<T>& MSE<T>::costDerivative(const NetNode<T>& prediction, const NetNode<T>& label, NetNode<T>& result)
{
    result = prediction;
    result = result - label;
    return result;
}

template <class T>
double MAE<T>::cost(const NetNode<T>& prediction, const NetNode<T>& label)
{
    NetNode<T> out = prediction;
    out = out - label;
    return out.fabsSum();
}

template <class T>
NetNode<T>& MAE<T>::costDerivative(const NetNode<T> &prediction, const NetNode<T> &label, NetNode<T> &result)
{
    result = prediction.compare(label, result);
    return result;
}

template <class T>
double CrossEntropy<T>::cost(const NetNode<T>& prediction, const NetNode<T>& label)
{
    double loss = 0.0;
    uint64_t arg_max_ref;
    mublas::vector<T> pred_data, label_data = label.getData();
    arg_max_ref = std::distance(label_data.begin(), std::max_element(label_data.begin(), label_data.end()));
    if (label_data[arg_max_ref] == 1)
    {
        pred_data = prediction.getData();
        if (pred_data[arg_max_ref] == 0) pred_data[arg_max_ref] = EPSILON;
        else if (pred_data[arg_max_ref] == 1) pred_data[arg_max_ref] = 1 - EPSILON;
        loss = log(pred_data[arg_max_ref]);
    }
    else
    {
        NetNode<T> temp_pred = prediction, temp_label = label;
        temp_pred = temp_label * temp_pred.log_e();
        loss = temp_pred.sum();
    }
    return -loss;
}

template <class T>
double CrossEntropy<T>::accuracy(const NetNode<T>& prediction, const NetNode<T>& label)
{
    int arg_max_hyp, arg_max_ref;
    mublas::vector<T> pred_data, label_data;
    double result = 0.0;
    pred_data = prediction.getData();
    label_data = label.getData();
    arg_max_hyp = std::distance(pred_data.begin(), std::max_element(pred_data.begin(), pred_data.end()));
    arg_max_ref = std::distance(label_data.begin(), std::max_element(label_data.begin(), label_data.end()));
    if (arg_max_hyp == arg_max_ref) result = 1.0;
    return result;
}

template <class T>
NetNode<T>& CrossEntropy<T>::costDerivative(const NetNode<T> &prediction, const NetNode<T> &label, NetNode<T> &result)
{
    result = prediction;
    result = result - label;
    return result;
}
template class Node<double>;
template class NetNode<double>;
template class Activation<double>;
template class CostFunction<double>;
template class MSE<double>;
template class MAE<double>;
template class CrossEntropy<double>;

} // namespace mlearn
