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
#include <vector>
#include "network.h"

namespace mlearn {
template <class T>
Network<T>& Network<T>::addLayer(const Layer<T>& layer)
{
    layers.push_back(layer);
    return *this;
}

template <class T>
Network<T>& Network<T>::connectLayers()
{
    for (std::size_t i = 1; i < layers.size(); ++i)
    {
        layers[i].connect(layers[i - 1]);
    }
    return *this;
}

template <class T>
const NetNode<T>& Network<T>::singleForward(const NetNode<T>& in_data)
{
    layers[0].setInputData(in_data);
    for(std::size_t i = 0; i < layers.size(); ++i)layers[i].forwardProp();
    return layers.back().getOutputData();
}

template <class T>
const NetNode<T>& Network<T>::singleBackward(const NetNode<T>& in_delta)
{
    layers.back().setInputDelta(in_delta);
    for(int i = layers.size() - 1; i >= 0; --i)layers[i].backwardProp();
    return layers[0].getOutputDelta();
}

template <class T>
Network<T>& Network<T>::updateNetwork(
                        double learning_rate,
                        uint32_t batch_size,
                        double lambda,
                        std::string reg,
                        double beta)
{
    for(std::size_t i = 0; i < layers.size(); ++i)
    {
        layers[i].updateParams(learning_rate, batch_size, lambda, reg, beta);
        layers[i].clearDeltas();
    }
    return *this;
}

template <class T>
Network<T>& Network<T>::updateNetwork(
                        double learning_rate,
                        uint32_t batch_size,
                        double lambda,
                        std::string reg,
                        double beta,
                        bool change_rate,
                        std::string id)
{
    for(std::size_t i = 0; i < layers.size(); ++i)
    {
        layers[i].updateParams(learning_rate, batch_size, lambda, reg, beta, change_rate, id);
        layers[i].clearDeltas(beta);
    }
    return *this;
}

template <class T>
void Network<T>::saveModel(std::string model_file)
{
    std::cout << "Saving model..." << std::endl;
    std::ofstream ofs(model_file);
    if (!ofs.good())
    {
        throw std::ios::failure("Error opening file!");
    }
    boost::archive::binary_oarchive oa(ofs);
    oa & layers;
}

template <class T>
Network<T>& Network<T>::loadModel(std::string model_file)
{
    std::cout << "Loading model... " << std::endl;
    std::ifstream ifs(model_file);
    if (!ifs.good())
    {
        throw std::ios::failure("Error opening file!");
    }
    boost::archive::binary_iarchive ia(ifs);
    ia & layers;
    this->connectLayers();
    return *this;
}
//template class Network<float>;
template class Network<double>;
} // namespace mlearn
