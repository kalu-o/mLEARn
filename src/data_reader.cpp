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
#include "data_reader.h"

namespace mlearn {
template <class T>
std::vector<int>& DataReader<T>::shuffleIndex(std::vector<int>& indices)const
{
    if (indices.empty())
    {
        for (uint64_t i = 0; i < row_dim; ++i) indices.push_back(i);
    }
    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    std::mt19937 eng(seed);
    std::shuffle(begin(indices), end(indices), eng);
    return indices;
}

template <class T>
DataReader<T>& DataReader<T>::destroy()
{

    if (features.size() == 0) return *this;
    try{
        for (uint64_t i = 0; i < this->row_dim; ++i)
        {
            delete this->features[i];
            delete this->labels[i];
        }
    }catch (std::logic_error const& e){
        std::cerr << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
    this->features.clear();
    this->labels.clear();
    this->feature_dim = this->label_dim = this->row_dim = 0;
    return *this;
}

template <class T>
DataReader<T>& DataReader<T>::trainTestSplit(DataReader<T>& test, double test_size)
{
    std::cout << "Splitting into train and validation set" << std::endl;
    std::set<int> test_indices;
    uint64_t test_row_dim = (uint64_t) row_dim * test_size, i = 0, j = 0;
    using value_type = int;
    std::random_device r;
    std::uniform_int_distribution<value_type> distribution(0, row_dim - 1);
    static std::default_random_engine generator {r()};
    while(i <  test_row_dim)
    {
        j = distribution(generator);
        if(test_indices.count(j) == 0)
        {
            test.features.push_back(new NetNode<T>(*features[j]));
            test.labels.push_back(new NetNode<T>(*labels[j]));
            delete features[j];
            delete labels[j];
            features[j] = nullptr;
            labels[j] = nullptr;
            test_indices.insert(j);
            ++i;
        }
    }
    features.erase(std::remove(features.begin(), features.end(), nullptr), features.end());
    labels.erase(std::remove(labels.begin(), labels.end(), nullptr), labels.end());
    row_dim = labels.size();
    test.row_dim = test_row_dim;
    test.feature_dim = feature_dim;
    test.label_dim = label_dim;
    std::cout << "Splitting ends" << std::endl;
    return test;
}

template <class T>
MNISTReader<T>::MNISTReader(const MNISTReader<T>& arg): DataReader<T>(arg.file_name, arg.sep, arg.header)
{
    for (uint64_t i = 0; i < arg.row_dim; ++i)
    {
        try{
            this->features.push_back(new Node<T>(*arg.features[i]));
            this->labels.push_back(new Node<T>(*arg.labels[i]));
        }catch (std::bad_alloc &e){
            std::cerr << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }

    }
    this->feature_dim = arg.feature_dim;
    this->label_dim = arg.label_dim;
    this->row_dim = arg.row_dim;
}

template <class T>
MNISTReader<T>& MNISTReader<T>::operator=(const MNISTReader<T>& arg)
{
    if(this == &arg) return *this;
    for (uint64_t i = 0; i < arg.row_dim; ++i)
    {
        *this->features[i] = *arg.features[i];
        *this->labels[i] = *arg.labels[i];
    }
    this->feature_dim = arg.feature_dim;
    this->label_dim = arg.label_dim;
    this->row_dim = arg.row_dim;
    return *this;
}

template <class T>
MNISTReader<T>& MNISTReader<T>::read()
{
    std::cout << "Reading data starts..." << std::endl;
    std::string csv_line;
    std::fstream ifile(this->file_name, std::ios::in);
    if (!ifile.good())
    {
        throw std::ios::failure("Error opening file!");
    }
    while(getline(ifile, csv_line))
    {
        std::istringstream csv_stream(csv_line);
        mublas::vector<T> feat(this->feature_dim * 1), label(this->label_dim, 0);
        std::string csv_element;
        uint32_t i = 0;
        while(getline(csv_stream, csv_element, this->sep))
        {
            if (i == 0)
            {
                uint32_t index = atoi(csv_element.c_str());
                label[index] = 1;
            }
            else
            {
                feat[i - 1] = atof(csv_element.c_str())/MAX_MNIST_VALUE;
            }
            ++i;
        }

        try{
            this->features.push_back(new NetNode<double>{feat});
            this->labels.push_back(new NetNode<double>{label});
        }catch (std::bad_alloc &e){
            std::cerr << e.what();
            exit(EXIT_FAILURE);
        }
        feat.clear();
        label.clear();
    }
    this->feature_dim = this->features[0]->getDataSize();
    this->label_dim = this->labels[0]->getDataSize();
    this->row_dim = this->labels.size();
    std::cout << "Reading data ends" << std::endl;
    return *this;
}

template <class T>
GenericReader<T>& GenericReader<T>::read()
{
    std::cout << "Reading data starts..." << std::endl;
    std::string line;
    std::fstream ifile(this->file_name, std::ios::in);
    if (!ifile.good())
    {
        throw std::ios::failure("Error opening file!");
    }
    if(this->header) getline(ifile,line);
    while(getline(ifile, line))
    {
        std::istringstream stream(line);
        mublas::vector<T> feat(this->feature_dim), label(this->label_dim, 0);
        std::string element;
        uint32_t i = 0;
        while(getline(stream, element, this->sep))
        {
            if (i >= this->feature_dim)
            {
                label[i - this->feature_dim] = atof(element.c_str());
            }
            else feat[i] = atof(element.c_str());

            ++i;
        }
        try{
            this->features.push_back(new NetNode<double>{feat});
            this->labels.push_back(new NetNode<double>{label});
        }catch (std::bad_alloc &e){
            std::cerr << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }
        feat.clear();
        label.clear();
    }
    this->feature_dim = this->features[0]->getDataSize();
    this->label_dim = this->labels[0]->getDataSize();
    this->row_dim = this->labels.size();
    std::cout << "Reading data ends" << std::endl;
    return *this;
}

template <class T>
IrisReader<T>& IrisReader<T>::read()
{
    std::cout << "Reading data starts..." << std::endl;
    std::string line;
    std::fstream ifile(this->file_name, std::ios::in);
    if (!ifile.good())
    {
        throw std::ios::failure("Error opening file!");
    }
    this->header = true;
    if(this->header) getline(ifile,line);
    while(getline(ifile, line))
    {
        std::istringstream stream(line);
        mublas::vector<T> feat(this->feature_dim), label(this->label_dim, 0);
        std::string element;
        uint32_t i = 0;
        while(getline(stream, element, this->sep))
        {
            if (i == 0){++i; continue;}
            else if (i == 5)
            {
                if (element == "Iris-setosa") label[2] = 1;
                else if (element == "Iris-versicolor") label[1] = 1;
                else if (element == "Iris-virginica") label[0] = 1;
            }
            else feat[i - 1] = atof(element.c_str())/MAX_IRIS_VALUE;
            ++i;
        }
        try{
            this->features.push_back(new NetNode<double>{feat});
            this->labels.push_back(new NetNode<double>{label});
        }catch (std::bad_alloc &e){
            std::cerr << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }
        feat.clear();
        label.clear();
    }
    this->feature_dim = this->features[0]->getDataSize();
    this->label_dim = this->labels[0]->getDataSize();
    this->row_dim = this->labels.size();
    std::cout << "Reading data ends" << std::endl;
    return *this;
}
template class DataReader<double>;
template class MNISTReader<double>;
template class GenericReader<double>;
} // namespace mlearn








