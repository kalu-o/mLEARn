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
#include "optimizer.h"

namespace mlearn {
template <class T>
Network<T>& SGDHelper(
                      Network<T>& model,
                      uint32_t batch_size,
                      uint32_t num_epochs,
                      Optimizer<T>* opt,
                      const DataReader<T>* train,
                      const DataReader<T>* validation,
                      std::string id)
{
    auto start = std::chrono::system_clock::now();
    std::time_t start_time = std::chrono::system_clock::to_time_t(start);
    std::cout << "Training starts..." + id + " " << std::ctime(&start_time) << std::endl;
    if(model.getCostFunction()->getId() == "mse")
    {
        if(validation != nullptr)std::cout << "Train MSE" <<"|" << "Validation MSE" << std::endl;
        else std::cout<< "Train MSE" << std::endl;
    }
    else if(model.getCostFunction()->getId() == "mae")
    {
        if(validation != nullptr)std::cout << "Train MAE" <<"|" << "Validation MAE" << std::endl;
        else std::cout<< "Train MAE" << std::endl;
    }
    else if(model.getCostFunction()->getId() == "crossentropy")
    {
        if(validation != nullptr)std::cout << "Train Loss" <<"|" << "Validation Loss" <<"|"
            << "Train Accuracy" <<" %|" << "Validation Accuracy %" << std::endl;
        else std::cout<< "Train Loss" <<"|" << "Train Accuracy" << std::endl;
    }

    NetNode<T> train_pred, val_pred, train_temp_x, train_temp_y, val_temp_x, val_temp_y, in_delta;
    double train_loss = 0.0, val_loss = 0.0, train_accuracy = 0.0,  val_accuracy  = 0.0;
    uint64_t k = 0;
    std::vector<int> indices;
    std::vector< Node<T>*> val_x, train_x = train->getFeatures();
    std::vector< Node<T>*> val_y, train_y = train->getLabels();
    if (validation != nullptr)
    {
        val_x = validation->getFeatures();
        val_y = validation->getLabels();
    }
    for(uint32_t i = 0; i < num_epochs; ++i)
    {
        train->shuffleIndex(indices);
        k = 0;
        train_loss = val_loss = train_accuracy = val_accuracy  = 0.0;
        if (id == "adagrad" || id == "rmsprop" || id == "psdsquare") model.setUpdateRate(true);
        for(uint64_t j = 0; j < indices.size(); ++j)
        {
            train_temp_x = *train_x[indices[j]];

            train_temp_y = *train_y[indices[j]];
            train_pred = model.singleForward(train_temp_x);

            in_delta = model.getCostFunction()->costDerivative(train_pred, train_temp_y, in_delta);
            model.singleBackward(in_delta);
            if ((j + 1) % batch_size == 0) opt->update(model);
            train_loss += model.getCostFunction()->cost(train_pred, train_temp_y);
            if(model.getCostFunction()->getId() == "crossentropy")
                train_accuracy += dynamic_cast<CrossEntropy<T>*>(model.getCostFunction())->accuracy(train_pred, train_temp_y);
            if(validation != nullptr && k < val_y.size())
            {
                val_temp_x = *val_x[k];
                val_temp_y = *val_y[k];
                val_pred = model.singleForward(val_temp_x);
                val_loss += model.getCostFunction()->cost(val_pred, val_temp_y);
                if(model.getCostFunction()->getId() == "crossentropy")
                    val_accuracy += dynamic_cast<CrossEntropy<T>*>(model.getCostFunction())->accuracy(val_pred, val_temp_y);
                ++k;
            }

        }
        train_loss = train_loss/indices.size();
        train_accuracy = train_accuracy * 100/indices.size();
        if(validation != nullptr)
        {
            val_loss = val_loss/val_y.size();
            val_accuracy = val_accuracy * 100/val_y.size();
        }

        if(model.getCostFunction()->getId() == "crossentropy")
        {
            if(validation != nullptr)std::cout << "Epoch " << i + 1 << " : " << std::setprecision (4) << std::fixed << train_loss
                <<"|" <<val_loss  <<"|" << train_accuracy <<" %|" <<val_accuracy <<" %" << std::endl;
            else std::cout << "Epoch " << i + 1 << " : " << std::setprecision (4) << std::fixed << train_loss <<"|" << train_accuracy
            <<" %" << std::endl;
        }
        else
        {
            if(validation != nullptr)std::cout << "Epoch " << i + 1 << " : " << std::setprecision (4) << std::fixed << train_loss <<"|"
                <<val_loss << std::endl;
            else std::cout << "Epoch " << i + 1 << " : " << std::setprecision (4) << std::fixed << train_loss << std::endl;
        }
    }
    auto stop = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = stop - start;
    std::time_t stop_time = std::chrono::system_clock::to_time_t(stop);
    std::cout << std::endl << "Training ends " << std::ctime(&stop_time);
    std::cout << "Elapsed time: " << elapsed_seconds.count()  << std::endl << std::endl;
    return model;
}
template <class T>
Network<T>&  SGD<T>::train(
                           Network<T>& model,
                           std::string model_file,
                           const DataReader<T>* train,
                           const DataReader<T>* validation,
                           std::string id)
{
    //lambda /= train->getRowDim();
    SGDHelper(model, batch_size, num_epochs, this, train, validation, id);
    model.saveModel(model_file);
    return model;
}
template <class T>
double  SGD<T>::predict(Network<T>& model, const DataReader<T>* test, std::string model_file)
{

    auto start = std::chrono::system_clock::now();
    std::time_t start_time = std::chrono::system_clock::to_time_t(start);
    std::cout << "Test starts..." << std::ctime(&start_time) << std::endl;
    double accuracy = 0.0;
    std::vector< Node<T>*> test_x = test->getFeatures();
    std::vector< Node<T>*> test_y = test->getLabels();
    model.loadModel(model_file);
    NetNode<T> test_pred;
    for(uint64_t i = 0; i < test_x.size(); ++i)
    {
        test_pred = model.singleForward(*test_x[i]);
        accuracy += (model.getCostFunction())->accuracy(test_pred, *test_y[i]);
    }
    accuracy /= test_x.size();
    auto stop = std::chrono::system_clock::now();
    std::time_t stop_time = std::chrono::system_clock::to_time_t(stop);
    std::cout << "Test ends..." << std::ctime(&stop_time) << std::endl;
    return accuracy;
}
template <class T>
Network<T>& SGD<T>::update(Network<T>& model)
{
    model.updateNetwork(learning_rate, batch_size, lambda, reg, beta);
    return model;
}
template <class T>
Network<T>&  Adagrad<T>::train(
                               Network<T>& model,
                               std::string model_file,
                               const DataReader<T>* train,
                               const DataReader<T>* validation,
                               std::string id)
{
    //this->lambda /= train->getRowDim();
    SGDHelper(model, this->batch_size, this->num_epochs, this, train, validation, this->id);
    model.saveModel(model_file);
    return model;
}
template <class T>
double  Adagrad<T>::predict(Network<T>& model, const DataReader<T>* test, std::string model_file)
{
    double accuracy = SGD<T>::predict(model, test, model_file);
    return accuracy;
}
template <class T>
Network<T>&  Adagrad<T>::update(Network<T>& model)
{
    bool change_rate = model.getUpdateRate();
    model.updateNetwork(this->learning_rate, this->batch_size, this->lambda, this->reg, this->beta, change_rate, this->id);
    model.setUpdateRate(false);
    return model;
}
template <class T>
Network<T>& RMSProp<T>::train(
                              Network<T>& model,
                              std::string model_file,
                              const DataReader<T>* train,
                              const DataReader<T>* validation,
                              std::string id)
{
    //this->lambda /= train->getRowDim();
    SGDHelper(model, this->batch_size, this->num_epochs, this, train, validation, this->id);
    model.saveModel(model_file);
    return model;
}
template <class T>
Network<T>&  RMSProp<T>::update(Network<T>& model)
{
    bool change_rate = model.getUpdateRate();
    model.updateNetwork(this->learning_rate, this->batch_size, this->lambda, this->reg, this->beta, change_rate, this->id);
    model.setUpdateRate(false);
    return model;
}
template <class T>
double  RMSProp<T>::predict(Network<T>& model, const DataReader<T>* test, std::string model_file)
{
    double accuracy = SGD<T>::predict(model, test, model_file);
    return accuracy;
}
template <class T>
Network<T>& PSDSquare<T>::train(
                              Network<T>& model,
                              std::string model_file,
                              const DataReader<T>* train,
                              const DataReader<T>* validation,
                              std::string id)
{
    //this->lambda /= train->getRowDim();
    SGDHelper(model, this->batch_size, this->num_epochs, this, train, validation, this->id);
    model.saveModel(model_file);
    return model;
}
template <class T>
Network<T>&  PSDSquare<T>::update(Network<T>& model)
{
    bool change_rate = model.getUpdateRate();
    model.updateNetwork(this->learning_rate, this->batch_size, this->lambda, this->reg, this->beta, change_rate, this->id);
    model.setUpdateRate(false);
    return model;
}
template <class T>
double  PSDSquare<T>::predict(Network<T>& model, const DataReader<T>* test, std::string model_file)
{
    double accuracy = SGD<T>::predict(model, test, model_file);
    return accuracy;
}
//template class Optimizer<double>;

template class SGD<double>;
template class Adagrad<double>;
template class RMSProp<double>;
template class PSDSquare<double>;
} // namespace mlearn
