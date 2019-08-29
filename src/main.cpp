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
#include <cmath>
#include <boost/program_options.hpp>
#include "libutil.h"
#include "optimizer.h"

using namespace mlearn;
namespace po = boost::program_options;
/**
    The trainModel functions is responsible for training.

    @param mode Program usage mode: train
    @param optimizer optimization algorithm used: sgd, adagrad, rmsprop
    @param cost_function Cost function used: mse, mae, crossentropy
    @param learning_rate Learning rate used for training
    @param batch_size Batch size used for training
    @param num_epochs Number of training epochs
    @param lambda Regularization parameter used for training
    @param reg Type of regularization (L1, L2 or None)
    @param beta Momentum term/parameter
    @param dataset Path to the train or test file
    @param header Determines whether dataset file contains header
    @param feature_dim Number of input points/features
    @param label_dim Number of output points/labels
    @param test_size % of train dataset used for validation
    @param num_hidden_layers Number of hidden layers
    @param hidden_dim Number of hidden units
    @param hidden_activation Activation function of hidden layers
    @param output_activation Activation function of the output layer
    @param model_file Path to model file (for saving trained model)
*/
int trainModel(
          std::string mode,
          std::string optimizer,
          std::string cost_function,
          double learning_rate,
          uint32_t batch_size,
          uint32_t num_epochs,
          double lambda,
          std::string reg,
          double beta,
          std::string dataset,
          bool header,
          uint64_t feature_dim,
          uint64_t label_dim,
          double test_size,
          uint16_t num_hidden_layers,
          uint64_t hidden_dim,
          std::string hidden_activation,
          std::string output_activation,
          std::string model_file
          );
/**
    The testModel functions is responsible for testing
    on a held-out set with a trained model.

    @param mode Program usage mode: test
    @param cost_function Cost function used: mse, mae, crossentropy
    @param dataset Path to the train or test file
    @param header Determines whether dataset file contains header
    @param feature_dim Number of input points/features
    @param label_dim Number of output points/labels
    @param model_file Path to model file (for saving trained model)
*/
int testModel(
            std::string mode,
            std::string cost_function,
            std::string dataset,
            bool header,
            uint64_t feature_dim,
            uint64_t label_dim,
            std::string model_file
            );
/**
    The is the entry point of the program. It processes user
    command line options and implements the train or test functions.
*/
int main(int argc, char** argv)
{
    std::string mode = "train";
    std::string optimizer = "sgd";
    std::string cost_function = "MSE";
    double learning_rate = 0.1;
    uint32_t batch_size = 10;
    uint32_t num_epochs = 20;
    double lambda = 0.0;
    std::string reg = "None";
    double beta = 0.0;
    std::string dataset = "mnist";
    bool header = false;
    uint64_t feature_dim = 0;
    uint64_t label_dim = 0;
    double test_size = 0;
    uint16_t num_hidden_layers = 1;
    uint64_t hidden_dim = 0;
    std::string hidden_activation = "sigmoid";
    std::string output_activation = "sigmoid";
    std::string model_file = "temp.bin";
    po::variables_map vm;
    try
    {

        po::options_description desc("Training Options");
        desc.add_options()
            ("help,h", "gives information about usage")
            ("mode,m", po::value<std::string>(&mode)->default_value("train"), "program usage mode: either train or test")
            ("optimizer,o", po::value<std::string>(&optimizer)->default_value("sgd"), "optimization algorithm used: sgd, adagrad, rmsprop")
            ("cost_function,c", po::value<std::string>(&cost_function)->default_value("crossentropy"), "cost function used: mse, mae, crossentropy")
            ("learning_rate,r", po::value<double>(&learning_rate)->default_value(0.05), "learning rate used for training")
            ("batch_size,s", po::value<uint32_t>(&batch_size)->default_value(10), "batch size used for training")
            ("num_epochs,n", po::value<uint32_t>(&num_epochs)->default_value(20), "number of training epochs")
            ("lambda,l", po::value<double>(&lambda)->default_value(0.0), "regularization parameter used for training")
            ("reg,R", po::value<std::string>(&reg)->default_value("None"), "type of regularization (L1, L2 or None)")
            ("beta,b", po::value<double>(&beta)->default_value(0.0), "momentum term/parameter")
            ("dataset,d", po::value<std::string>(&dataset)->default_value("mnist"), "the path to the train or test file")
            ("header,H", po::value<bool>(&header)->default_value(false), "determines whether dataset file contains header")
            ("feature_dim,F", po::value<uint64_t>(&feature_dim)->default_value(784), "number of input points/features")
            ("label_dim,L", po::value<uint64_t>(&label_dim)->default_value(10), "number of output points/labels")
            ("test_size,t", po::value<double>(&test_size)->default_value(0.0), "% of train dataset used for validation")
            ("num_hidden_layers,N", po::value<uint16_t>(&num_hidden_layers)->default_value(1), "number of hidden layers")
            ("hidden_dim,D", po::value<uint64_t>(&hidden_dim)->default_value(100), "number of hidden units")
            ("hidden_activation,a", po::value<std::string>(&hidden_activation)->default_value("sigmoid"), "activation function of hidden layers")
            ("output_activation,A", po::value<std::string>(&output_activation)->default_value("softmax"), "activation function of output layer")
            ("model_file,f", po::value<std::string>(&model_file)->default_value("temp.bin"), "path to model file")
            ;

        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);



        if (vm.count("help"))
        {
            std::cout << desc << std::endl;
            return 0;
        }
        if (vm.count("mode"))mode = vm["mode"].as<std::string>();
        if (vm.count("optimizer"))optimizer = vm["optimizer"].as<std::string>();
        if (vm.count("cost_function"))cost_function = vm["cost_function"].as<std::string>();
        if (vm.count("learning_rate"))learning_rate = vm["learning_rate"].as<double>();
        if (vm.count("batch_size"))batch_size = vm["batch_size"].as<uint32_t>();
        if (vm.count("num_epochs"))num_epochs = vm["num_epochs"].as<uint32_t>();
        if (vm.count("lambda"))lambda = vm["lambda"].as<double>();
        if (vm.count("reg"))reg = vm["reg"].as<std::string>();
        if (vm.count("beta"))beta = vm["beta"].as<double>();
        if (vm.count("dataset"))dataset = vm["dataset"].as<std::string>();
        if (vm.count("header"))header = vm["header"].as<bool>();
        if (vm.count("feature_dim"))feature_dim = vm["feature_dim"].as<uint64_t>();
        if (vm.count("label_dim"))label_dim = vm["label_dim"].as<uint64_t>();
        if (vm.count("test_size"))test_size = vm["test_size"].as<double>();
        if (vm.count("num_hidden_layers"))num_hidden_layers = vm["num_hidden_layers"].as<uint16_t>();
        if (vm.count("hidden_dim"))hidden_dim = vm["hidden_dim"].as<uint64_t>();
        if (vm.count("hidden_activation"))hidden_activation = vm["hidden_activation"].as<std::string>();
        if (vm.count("output_activation"))output_activation = vm["output_activation"].as<std::string>();
        if (vm.count("model_file"))model_file = vm["model_file"].as<std::string>();

    }
    catch(std::exception& e)
    {
        std::cerr << "error: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    if(vm["mode"].as<std::string>() == "train")
    {
        std::set<std::string> optimizers{"sgd", "adagrad", "rmsprop", "psdsquare"};
        std::set<std::string> cost_functions{"mse", "mae", "crossentropy"};
        if (optimizers.count(optimizer) == 0)
        {
            std::cerr << "optimizer error: sgd, adagrad, rmsprop, psdsquare" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (cost_functions.count(cost_function) == 0)
        {
            std::cerr << "cost function error: mse, mse, crossentropy" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (cost_function == "crossentropy" && output_activation != "softmax")
        {

            std::cerr << "cost function/output activation error: crossentropy goes with softmax!" << std::endl;
            exit(EXIT_FAILURE);
        }
        std::cout << "-----T r a i n   C o n f i g u r a t i o n-----" << std::endl << std::endl;
        std::cout << "program usage mode: " << vm["mode"].as<std::string>() << std::endl;
        std::cout << "optimization algorithm used: " << vm["optimizer"].as<std::string>() << std::endl;
        std::cout << "cost function used: " << vm["cost_function"].as<std::string>() << std::endl;
        std::cout << "learning rate: " << vm["learning_rate"].as<double>() << std::endl;
        std::cout << "batch size: " << vm["batch_size"].as<uint32_t>() << std::endl;
        std::cout << "number of training epochs: " << vm["num_epochs"].as<uint32_t>() << std::endl;
        std::cout << "regularization parameter lambda: " << vm["lambda"].as<double>() << std::endl;
        std::cout << "type of regularization: " << vm["reg"].as<std::string>() << std::endl;
        std::cout << "momentum term/parameter beta: " << vm["beta"].as<double>() << std::endl;
        std::cout << "dataset: " << vm["dataset"].as<std::string>() << std::endl;
        std::cout << "dataset header info: " << vm["header"].as<bool>() << std::endl;
        std::cout << "input feature dimension: " << vm["feature_dim"].as<uint64_t>() << std::endl;
        std::cout << "output label dimension: " << vm["label_dim"].as<uint64_t>() << std::endl;
        std::cout << "% validation: " << vm["test_size"].as<double>() << std::endl;
        std::cout << "number of hidden layers: " << vm["num_hidden_layers"].as<uint16_t>() << std::endl;
        std::cout << "number of hidden units: " << vm["hidden_dim"].as<uint64_t>() << std::endl;
        std::cout << "activation function of hidden layers: " << vm["hidden_activation"].as<std::string>() << std::endl;
        std::cout << "activation function of output layer: " << vm["output_activation"].as<std::string>() << std::endl;
        std::cout << "path to model file: " << vm["model_file"].as<std::string>() << std::endl;

        trainModel(
                mode,
                optimizer,
                cost_function,
                learning_rate,
                batch_size,
                num_epochs,
                lambda,
                reg,
                beta,
                dataset,
                header,
                feature_dim,
                label_dim,
                test_size,
                num_hidden_layers,
                hidden_dim,
                hidden_activation,
                output_activation,
                model_file
                );
    }
    else if(vm["mode"].as<std::string>() == "test")
    {
        std::cout << "-----T e s t   C o n f i g u r a t i o n-----" << std::endl << std::endl;
        std::cout << "program usage mode: " << vm["mode"].as<std::string>() << std::endl;
        std::cout << "dataset: " << vm["dataset"].as<std::string>() << std::endl;
        std::cout << "dataset header info: " << vm["header"].as<bool>() << std::endl;
        std::cout << "input feature dimension: " << vm["feature_dim"].as<uint64_t>() << std::endl;
        std::cout << "output label dimension: " << vm["label_dim"].as<uint64_t>() << std::endl;
        std::cout << "path to model file: " << vm["model_file"].as<std::string>() << std::endl;

        testModel(mode, cost_function, dataset, header, feature_dim, label_dim, model_file);
    }
    else
    {
        std::cerr << "error: program mode not set/wrong mode" << std::endl;
        exit(EXIT_FAILURE);
    }

    return 0;
}

int trainModel(
          std::string mode,
          std::string optimizer,
          std::string cost_function,
          double learning_rate,
          uint32_t batch_size,
          uint32_t num_epochs,
          double lambda,
          std::string reg,
          double beta,
          std::string dataset,
          bool header,
          uint64_t feature_dim,
          uint64_t label_dim,
          double test_size,
          uint16_t num_hidden_layers,
          uint64_t hidden_dim,
          std::string hidden_activation,
          std::string output_activation,
          std::string model_file
          )
{

    Activation<double> a_hidden(hidden_activation), a_output(output_activation);
    Network<double>* model(nullptr);
    Optimizer<double>* opt(nullptr);
    Layer<double>* layer(nullptr);
    DataReader<double>* train(nullptr);
    DataReader<double>* validation(nullptr);

    if(cost_function == "mse") model = new Network<double>(new MSE<double>);
    else if(cost_function == "mae") model = new Network<double>(new MAE<double>);
    else if(cost_function == "crossentropy") model =  new Network<double>(new CrossEntropy<double>);
    if(num_hidden_layers > 0)
    {
        layer = new Layer<double>(feature_dim, hidden_dim, "hidden", a_hidden);
        model->addLayer(*layer);
        if(layer)delete layer;
    }

    for(uint16_t i = 1; i < num_hidden_layers; ++i)
    {
        layer = new Layer<double>(hidden_dim, hidden_dim, "hidden", a_hidden);
        model->addLayer(*layer);
        if(layer)delete layer;
    }
    if(num_hidden_layers > 0)layer = new Layer<double>(hidden_dim, label_dim, "output", a_output);
    else layer = new Layer<double>(feature_dim, label_dim, "output", a_output);
    model->addLayer(*layer);
    if(layer)delete layer;
    model->connectLayers();
    if(optimizer == "sgd") opt = new SGD<double>(learning_rate, batch_size, num_epochs, lambda, reg, beta);
    else if(optimizer == "adagrad") opt = new Adagrad<double>(learning_rate, batch_size, num_epochs, lambda, reg, beta);
    else if(optimizer == "rmsprop") opt = new RMSProp<double>(learning_rate, batch_size, num_epochs, lambda, reg, beta);
    else if(optimizer == "psdsquare") opt = new PSDSquare<double>(learning_rate, batch_size, num_epochs, lambda, reg, beta);
    else
    {
        std::cerr << "optimizer is not implemented" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (dataset == "mnist")
    {
        train = new MNISTReader<double>("data/mnist_train.csv", ',',  false);
        train->read();
        if(test_size > 0.0)
        {
            validation = new MNISTReader<double>;
            train->trainTestSplit(*validation, test_size);
            opt->train(*model, model_file, train, validation);
        }
        else opt->train(*model, model_file, train);
    }
    else
    {
        train = new GenericReader<double>(dataset, feature_dim, label_dim, ',',  header);
        train->read();
        if(test_size > 0.0)
        {
            validation = new GenericReader<double>;
            train->trainTestSplit(*validation, test_size);
            opt->train(*model, model_file, train, validation);
        }
        else opt->train(*model, model_file, train);

    }
    if(opt)delete opt;
    if(model)delete model;
    if(train)delete train;
    if(validation)delete validation;
    return 0;

}
int testModel(
            std::string mode,
            std::string cost_function,
            std::string dataset,
            bool header,
            uint64_t feature_dim,
            uint64_t label_dim,
            std::string model_file
            )
{
    std::set<std::string> cost_functions{"mse", "mae", "crossentropy"};
    Optimizer<double>* opt = new SGD<double>;
    DataReader<double>* test(nullptr);
    Network<double>* model(nullptr);
    double accuracy;
    std::string test_info;

    if (cost_functions.count(cost_function) == 0)
    {
        std::cerr << "cost function error: mse, mse, crossentropy" << std::endl;
        exit(EXIT_FAILURE);
    }
    if(cost_function == "mse") model = new Network<double>(new MSE<double>);
    else if(cost_function == "mae") model = new Network<double>(new MAE<double>);
    else if(cost_function == "crossentropy") model =  new Network<double>(new CrossEntropy<double>);

    if (dataset == "mnist")
    {
        try
        {
            test = new MNISTReader<double>("data/mnist_test.csv", ',',  false);
            test->read();
        }catch (std::exception &e){
            std::cerr << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        try
        {
            test = new GenericReader<double>(dataset, feature_dim, label_dim, ',',  header);
            test->read();
        }catch (std::exception &e){
            std::cerr << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    accuracy = opt->predict(*model, test, model_file);
    if (cost_function == "mse")test_info = "MSE: " + std::to_string(accuracy);
    else if (cost_function == "mae")test_info = "MAE: " + std::to_string(accuracy);
    else if (cost_function == "crossentropy")test_info = "Classification Accuracy: " + std::to_string(accuracy * 100) + " %";
    if(opt)delete opt;
    if(test)delete test;
    if(model)delete model;
    std::cout << test_info << std::endl;

    return 0;
}
