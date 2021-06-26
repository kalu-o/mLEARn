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
#include "utils.h"

namespace mlearn {
double MSE::costFunction(const mublas::vector<double>& prediction, const mublas::vector<double>& label)
{
    double error = 0.0;
    mublas::vector<double> out = prediction - label;
    error = mublas::inner_prod(out, out);
    return error;
}

mublas::vector<double>& MSE::costDerivative(const mublas::vector<double>& prediction, const mublas::vector<double>& label, mublas::vector<double>& result)
{
    result = prediction - label;
    return result;
}

double MAE::costFunction(const mublas::vector<double>& prediction, const mublas::vector<double>& label)
{
    auto result = 0.0;
	mublas::vector<double> out = prediction - label;
    for(uint64_t i = 0; i < out.size(); ++i) result += fabs(out[i]);
    return result;
}

mublas::vector<double>& MAE::costDerivative(const mublas::vector<double>& prediction, const mublas::vector<double>& label, mublas::vector<double> &result)
{
	for(uint64_t i = 0; i < prediction.size(); ++i)
    {
        if (prediction[i] < label[i])result[i] = -1.0;
        else if (prediction[i] > label[i])result[i] = 1.0;
        else result[i] = 0.0;
    }
    return result;
}

double CrossEntropy::costFunction(const mublas::vector<double>& prediction, const mublas::vector<double>& label)
{
    double loss = 0.0;
    uint64_t arg_max_ref;
    mublas::vector<double> temp_prediction = prediction;
    arg_max_ref = std::distance(label.begin(), std::max_element(label.begin(), label.end()));
    if (label[arg_max_ref] == 1)
    {
        if (temp_prediction[arg_max_ref] == 0) temp_prediction[arg_max_ref] = EPSILON;
        else if (temp_prediction[arg_max_ref] == 1) temp_prediction[arg_max_ref] = 1 - EPSILON;
        loss = log(temp_prediction[arg_max_ref]);
		//std::cout << temp_prediction[arg_max_ref]<<loss<<std::endl;
    }
    else
    {
		for(uint64_t i = 0; i < temp_prediction.size(); ++i) temp_prediction[i] = log(prediction[i]);
        mublas::vector<double> result = element_prod(label, temp_prediction);
        loss = mublas::sum(result);
    }
    return -loss;
}

double CrossEntropy::accuracy(const mublas::vector<double>& prediction, const mublas::vector<double>& label)
{
    int arg_max_hyp, arg_max_ref;
	double result = 0.0;
    arg_max_hyp = std::distance(prediction.begin(), std::max_element(prediction.begin(), prediction.end()));
    arg_max_ref = std::distance(label.begin(), std::max_element(label.begin(), label.end()));
    if (arg_max_hyp == arg_max_ref) result = 1.0;
    return result;
}

mublas::vector<double>& CrossEntropy::costDerivative(const mublas::vector<double>& prediction, const mublas::vector<double>& label, mublas::vector<double>& result)
{
    result = prediction - label;
    return result;
}

mublas::vector<double>& sigmoid(mublas::vector<double>& argv)
{

    for(uint64_t i = 0; i < argv.size(); ++i)argv[i] = 1.0 / (1.0 + exp(-argv[i]));
    return argv;
}

mublas::vector<double> sigmoidPrime(mublas::vector<double>& argv)
{
	//mublas::vector<double>sigmoid_argv = sigmoid(argv);
	return argv - element_prod(argv, argv);
}

mublas::vector<double>& hyperTan(mublas::vector<double>& argv)
{
    for(uint64_t i = 0; i < argv.size(); ++i) argv[i] = tanh(argv[i]);
    return argv;
}

mublas::vector<double>& hyperTanPrime(mublas::vector<double>& argv)
{
    for(uint64_t i = 0; i < argv.size(); ++i) argv[i] = (1 - argv[i] * argv[i]);
    return argv;
}

mublas::vector<double>& ReLU(mublas::vector<double>& argv)
{
    for(uint64_t i = 0; i < argv.size(); ++i) argv[i] = std::max(0.0, argv[i]);
    return argv;
}

mublas::vector<double>& ReLUPrime(mublas::vector<double>& argv, double alpha)
{
    for(uint64_t i = 0; i < argv.size(); ++i)argv[i] = argv[i] < 0.0 ? alpha : 1.0;
    return argv;
}

mublas::vector<double>& ELU(mublas::vector<double>& argv, double alpha)
{
    for(uint64_t i = 0; i < argv.size(); ++i) argv[i] = argv[i] > 0 ? argv[i] : alpha * (exp(argv[i]) - 1);
    return argv;
}

mublas::vector<double>& ELUPrime(mublas::vector<double>& argv, double alpha)
{
    for(uint64_t i = 0; i < argv.size(); ++i)argv[i] = argv[i] > 0.0 ? 1.0 : alpha * exp(argv[i]);
    return argv;
}

mublas::vector<double>& identity(mublas::vector<double>& argv)
{
    return argv;
}

mublas::vector<double>& identityPrime(mublas::vector<double>& argv)
{
    for(uint64_t i = 0; i < argv.size(); ++i) argv[i] = 1.0;
    return argv;
}

mublas::vector<double>& softmax(mublas::vector<double>& argv)
{
    double t_sum = 0.0, max_e = *max_element(argv.begin(), argv.end());
    for(uint64_t i = 0; i < argv.size(); ++i)
    {
        argv[i] = exp(argv[i] - max_e);
        t_sum += argv[i];
    }
    for(uint64_t i = 0; i < argv.size(); ++i)argv[i] = argv[i]/t_sum;
    return argv;
}

mublas::matrix<double>& softmaxPrime(mublas::vector<double>& argv1, mublas::matrix<double>& argv2)
{

    for(uint64_t i = 0; i < argv1.size(); ++i)
        for(uint64_t j = 0; j < argv1.size(); ++j)
        {
            if (i == j) argv2(i, j) = argv1[i] * (1 - argv1[i]);
            else argv2(i, j) = -argv1[i] * argv1[j];
        }
    return argv2;
}

mublas::vector<double>& log_e(mublas::vector<double>& argv)
{
    for(uint64_t i = 0; i < argv.size(); ++i) argv[i] = log(argv[i]);
    return argv;
}

mublas::vector<double>& generateRandomData(mublas::vector<double>& argv, uint64_t input_dim, uint64_t output_dim)
{
    double lower = - sqrt(6.0/(input_dim + output_dim)) * 4;
    double upper = - (lower);
    using value_type = double;
    std::random_device r;
    static std::uniform_real_distribution<value_type> distribution(
        lower,
        upper);
    static std::default_random_engine generator {r()};
    std::generate(argv.begin(), argv.end(), []() { return distribution(generator); });
    return argv;
}
/*vec_ptr_matrix& destroy(vec_ptr_matrix &argv)
{
    if (argv.size() == 0)return argv;
    try{
        for (auto ptr: argv)if(ptr)delete ptr;
    }catch (std::logic_error const& e){
        std::cerr << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
    argv.clear();
    return argv;
}
ptr_matrix& addBias(ptr_matrix &argv, double bias)
{
    uint8_t dim = argv->size1();
    for (uint8_t i=0; i<dim; ++i)
        for (uint8_t j=0; j<dim; ++j)(*argv)(i, j) += bias;
    return argv;
}
double findMax(ptr_matrix argv)
{
    uint8_t dim = argv->size1();
    double m_element = (*argv)(0,0);
    for (uint8_t i=0; i<dim; ++i)
        for (uint8_t j=0; j<dim; ++j)if ((*argv)(i, j) > m_element)m_element = (*argv)(i, j);
    return m_element;
}*/
} // namespace mlearn
