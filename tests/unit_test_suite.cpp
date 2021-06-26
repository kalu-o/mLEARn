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
#define BOOST_TEST_MODULE unitTest
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <cmath>
#include "libutil.h"
#include "network.h"
#include "data_reader.h"

using namespace mlearn;
template <class T>
/**
    A helper function used to validate/test if 2 vectors contain the same elements.

    @param v1 First vector
    @param v1 Second vector

    @return A boolean value (true or false)
*/
boost::test_tools::predicate_result validateVector(const mublas::vector<T>& v1,  const mublas::vector<T>& v2)
{
    if (v1.size() != v2.size()) return false;
    for (uint64_t i = 0; i < v1.size(); ++i)
    {
        if (fabs(v1[i] - v2[i]) > EPSILON)
        {
            return false;
        }
    }
    return true;
}
/**
    Unit test for the Node class. Tests constructors and operators.
*/
BOOST_AUTO_TEST_SUITE (NodeSuite)
BOOST_AUTO_TEST_CASE(contructor_test)
{
    std::cout << "Start Node class test" << std::endl;
    std::vector<int> sv1 {1, 2, 3, 4};
    std::vector<double> sv2{2, 3, 4.0, 2};
    std::vector<int> sv3{0, 0, 0, 0, 0};

    mublas::vector<int> v1(sv1.size());
    mublas::vector<double> v2(sv2.size());
    mublas::vector<int> v3(sv3.size());

    std::copy(sv1.begin(), sv1.end(), v1.begin());
    std::copy(sv2.begin(), sv2.end(), v2.begin());
    std::copy(sv3.begin(), sv3.end(), v3.begin());

    Node<int> n1(v1);
    Node<double> n2(v2);
    Node<double> n3;
    Node<int> n4{n1};
    Node<int> n5(5);
    BOOST_CHECK(validateVector(v1, n1.getData()));
    BOOST_CHECK(validateVector(v1, n4.getData()));
    BOOST_CHECK(validateVector(v2, n2.getData()));
    BOOST_CHECK(n3.getDataSize() == 0);
    BOOST_CHECK(n4.getDataSize() == n1.getDataSize());
    BOOST_CHECK(validateVector(v3, n5.getData()));
}

BOOST_AUTO_TEST_CASE(operator_test)
{

    std::vector<double> sv1 {1.2, 2.3, 3.0, 4.1};
    std::vector<double> sv2{2, 3, 4.0, 2};
    std::vector<double> sv3{3.2, 5.3, 7.0, 6.1};
    std::vector<double> sv4 {1.2, 2.3, 3, 4.1};
    std::vector<double> sv5 {2.4, 6.9, 12.0, 8.2};
    std::vector<double> sv6 {4.8, 13.8, 24.0, 16.4};

    mublas::vector<double> v1(sv1.size());
    mublas::vector<double> v2(sv2.size());
    mublas::vector<double> v3(sv3.size());
    mublas::vector<double> v4(sv4.size());
    mublas::vector<double> v5(sv5.size());
    mublas::vector<double> v6(sv6.size());

    std::copy(sv1.begin(), sv1.end(), v1.begin());
    std::copy(sv2.begin(), sv2.end(), v2.begin());
    std::copy(sv3.begin(), sv3.end(), v3.begin());
    std::copy(sv4.begin(), sv4.end(), v4.begin());
    std::copy(sv5.begin(), sv5.end(), v5.begin());
    std::copy(sv6.begin(), sv6.end(), v6.begin());

    Node<double> n1(v1);
    Node<double> n2(v2);
    //operator+
    Node<double> n3 = n1 + n2;
    BOOST_CHECK(validateVector(v3, n3.getData()));
    //operator-
    Node<double> n4 = n3 - n2;

    BOOST_CHECK(validateVector(v1, n4.getData()));
    //operator*
    Node<double> n5 = n2 * n4;
    BOOST_CHECK(validateVector(v5, n5.getData()));
    //operator=
    n4 = n5;
    BOOST_CHECK(validateVector(n4.getData(), n5.getData()));
    BOOST_CHECK(n4.getDataSize() == n5.getDataSize());
    //scalarMultiply
    Node<double> n6 = n5.scalarMultiply(2);
    BOOST_CHECK(validateVector(v6, n6.getData()));
    //sum
    BOOST_CHECK(n6.sum() == 59);
    std::cout << "End Node class test" << std::endl;

}
BOOST_AUTO_TEST_SUITE_END()

/**
    Unit test for the NetNode class. Tests constructors and operators.
*/
BOOST_AUTO_TEST_SUITE (NetNodeSuite)
BOOST_AUTO_TEST_CASE(contructor_test)
{
    std::cout << "Start NetNode class test" << std::endl;
    std::vector<int> sv1 {1, 2, 3, 4};
    std::vector<double> sv2{2, 3, 4.0, 2};
    std::vector<int> sv3{0, 0, 0, 0, 0};

    mublas::vector<int> v1(sv1.size());
    mublas::vector<double> v2(sv2.size());
    mublas::vector<int> v3(sv3.size());

    std::copy(sv1.begin(), sv1.end(), v1.begin());
    std::copy(sv2.begin(), sv2.end(), v2.begin());
    std::copy(sv3.begin(), sv3.end(), v3.begin());

    NetNode<int> n1(v1);
    NetNode<double> n2(v2);
    NetNode<double> n3;
    NetNode<int> n4{n1};
    NetNode<int> n5(5);
    BOOST_CHECK(validateVector(v1, n1.getData()));
    BOOST_CHECK(validateVector(v1, n4.getData()));
    BOOST_CHECK(validateVector(v2, n2.getData()));
    BOOST_CHECK(n3.getDataSize() == 0);
    BOOST_CHECK(n4.getDataSize() == n1.getDataSize());
    BOOST_CHECK(validateVector(v3, n5.getData()));
}

BOOST_AUTO_TEST_CASE(operator_test)
{
    std::vector<double> sv1 {1.2, 2.3, 3.0, 4.1};
    std::vector<double> sv2{2, 3, 4.0, 2};
    std::vector<double> sv3{3.2, 5.3, 7.0, 6.1};
    std::vector<double> sv4 {1.2, 2.3, 3, 4.1};
    std::vector<double> sv5 {2.4, 6.9, 12.0, 8.2};
    std::vector<double> sv6 {4.8, 13.8, 24.0, 16.4};
    std::vector<double> sv7 {0.76852478, 0.90887704, 0.95257413, 0.9836975};
    std::vector<double> sv8 {0.17789444, 0.08281957, 0.04517666, 0.01603673};

    mublas::vector<double> v1(sv1.size());
    mublas::vector<double> v2(sv2.size());
    mublas::vector<double> v3(sv3.size());
    mublas::vector<double> v4(sv4.size());
    mublas::vector<double> v5(sv5.size());
    mublas::vector<double> v6(sv6.size());
    mublas::vector<double> v7(sv7.size());
    mublas::vector<double> v8(sv8.size());

    std::copy(sv1.begin(), sv1.end(), v1.begin());
    std::copy(sv2.begin(), sv2.end(), v2.begin());
    std::copy(sv3.begin(), sv3.end(), v3.begin());
    std::copy(sv4.begin(), sv4.end(), v4.begin());
    std::copy(sv5.begin(), sv5.end(), v5.begin());
    std::copy(sv6.begin(), sv6.end(), v6.begin());
    std::copy(sv7.begin(), sv7.end(), v7.begin());
    std::copy(sv8.begin(), sv8.end(), v8.begin());

    NetNode<double> n1(v1);
    NetNode<double> n2(v2);
    //operator+
    NetNode<double> n3 = n1 + n2;
    BOOST_CHECK(validateVector(v3, n3.getData()));
    //operator-
    NetNode<double> n4 = n3 - n2;

    BOOST_CHECK(validateVector(v4, n4.getData()));
    BOOST_CHECK(validateVector(v4, n1.getData()));
    //operator*
    NetNode<double> n5 = n2 * n4;
    BOOST_CHECK(validateVector(v5, n5.getData()));
    //operator=   not working. Fix!
    //n4 = n5;
    //scalarMultiply
    NetNode<double> n6 = n5.scalarMultiply(2);
    BOOST_CHECK(validateVector(v6, n6.getData()));
    // sum
    BOOST_CHECK(n6.sum() == 59);
    //sigmoid
    NetNode<double> n7 = n1.sigmoid();
    BOOST_CHECK(validateVector(v7, n7.getData()));
    //sigmoidPrime
    NetNode<double> n8 = n7.sigmoidPrime();
    BOOST_CHECK(validateVector(v8, n8.getData()));
    std::cout << "End NetNode class test" << std::endl;
}
BOOST_AUTO_TEST_SUITE_END()

/**
    Unit test for the Activation class. Performs general tests.
*/
BOOST_AUTO_TEST_SUITE (ActivationSuite)
BOOST_AUTO_TEST_CASE(general_test)
{
    std::cout << "Start Activation class test" << std::endl;
    std::vector<double> sv1 {1.2, 2.3, 3.0, 4.1};
    std::vector<double> sv2{0.76852478, 0.90887704, 0.95257413, 0.9836975};
    std::vector<double> sv3{1.2, 2.3, 3.0, 4.1};
    std::vector<double> sv4{0.17789444, 0.08281957, 0.04517666, 0.01603673};

    mublas::vector<double> v1(sv1.size());
    mublas::vector<double> v2(sv2.size());
    mublas::vector<double> v3(sv3.size());
    mublas::vector<double> v4(sv4.size());

    std::copy(sv1.begin(), sv1.end(), v1.begin());
    std::copy(sv2.begin(), sv2.end(), v2.begin());
    std::copy(sv3.begin(), sv3.end(), v3.begin());
    std::copy(sv4.begin(), sv4.end(), v4.begin());

    NetNode<double> n1(v1);
    NetNode<double> n3(v3);
    Activation<double> act("sigmoid");
    act.compute(n1);
    BOOST_CHECK(validateVector(v2, n1.getData()));
    //computeDerivative
    act.computeDerivative(n1);
    BOOST_CHECK(validateVector(v4, n1.getData()));
    //softmax
    mublas::vector<double> v5(2), v6(2);
    v5[0] = 1.0, v5[1] = 2.0, v6[0] = 0.26894142, v6[1] = 0.73105858;
    //compute
    Activation<double> soft_act("softmax");
    NetNode<double> n5(v5);
    soft_act.compute(n5);
    BOOST_CHECK(validateVector(v6, n5.getData()));
    //computeDerivative
    mublas::matrix<double> m1(2, 2);
    soft_act.computeDerivative(n5, m1);
    //std::cout <<m1 <<std::endl; //[2,2]((0.196612,-0.196612),(-0.196612,0.196612))
    std::cout << "End Activation class test" << std::endl;
}
BOOST_AUTO_TEST_SUITE_END()

/**
    Unit test for the Layer class. Performs general tests.
*/
BOOST_AUTO_TEST_SUITE (Layer)
BOOST_AUTO_TEST_CASE(general_test)
{
    std::cout << "Start Layer class test" << std::endl;
    std::vector<double> ax{0, 0}, bx{0, 1}, cx{1, 0}, dx{1, 1};
    std::vector<double> ay{0}, by{1}, cy{1}, dy{1};
    std::vector<double> sv1{-0.7, 0.8}, sv2{0.66}, sv3{0, 0}, sv4{0.331812, 0.689974}, sv5{0.763192}, sv6{-0.76};
    std::vector<double> sv7 {-0.14};
    std::vector<double> sv8 {-0.03, -0.01};

    mublas::vector<double> v1(sv1.size());
    mublas::vector<double> v2(sv2.size());
    mublas::vector<double> v3(sv3.size());
    mublas::vector<double> v4(sv4.size());
    mublas::vector<double> v5(sv5.size());
    mublas::vector<double> v6(sv6.size());
    mublas::vector<double> v7(sv7.size());
    mublas::vector<double> v8(sv8.size());

    std::copy(sv1.begin(), sv1.end(), v1.begin());
    std::copy(sv2.begin(), sv2.end(), v2.begin());
    std::copy(sv3.begin(), sv3.end(), v3.begin());
    std::copy(sv4.begin(), sv4.end(), v4.begin());
    std::copy(sv5.begin(), sv5.end(), v5.begin());
    std::copy(sv6.begin(), sv6.end(), v6.begin());
    std::copy(sv7.begin(), sv7.end(), v7.begin());
    std::copy(sv8.begin(), sv8.end(), v8.begin());

    Activation<double> hidden("sigmoid"), output("sigmoid");
    Layer<double> hidden_layer(2, 2, "hidden", hidden);
    Layer<double> output_layer(2, 1, "output", output);
    std::unique_ptr<std::vector<double>> v_ptr1(new std::vector<double>{0.62, 0.55});
    std::unique_ptr<std::vector<double>> v_ptr2(new std::vector<double>{0.42, -0.17});
    std::unique_ptr<std::vector<double>> v_ptr3(new std::vector<double>{0.81, 0.35});
    std::unique_ptr<NetNode<double>> v_ptr4(new NetNode<double>{v1});
    std::unique_ptr<NetNode<double>> v_ptr5(new NetNode<double>{v2});
    std::unique_ptr<NetNode<double>> v_ptr6(new NetNode<double>{v3});
    std::unique_ptr<NetNode<double>> v_ptr7(new NetNode<double>{v6});
    hidden_layer.push_row(0, *v_ptr1);
    hidden_layer.push_row(1, *v_ptr2);
    output_layer.push_row(0, *v_ptr3);
    hidden_layer.setBias(*v_ptr4);
    output_layer.setBias(*v_ptr5);

    output_layer.connect(hidden_layer);
    hidden_layer.setInputData(*v_ptr6);
    NetNode<double> n1 = hidden_layer.forwardProp();
    NetNode<double> n2 = output_layer.forwardProp();
    BOOST_CHECK(validateVector(v4, n1.getData()));
    BOOST_CHECK(validateVector(v5, n2.getData()));
    //backwardProp()
    output_layer.setInputDelta(*v_ptr7);
    NetNode<double> n3 = output_layer.backwardProp();
    NetNode<double> n4 = hidden_layer.backwardProp();

    BOOST_CHECK(validateVector(v7, n3.getData()));
    BOOST_CHECK(validateVector(v8, n4.getData()));
    hidden_layer.clearDeltas();
    mublas::matrix<double> m1 = hidden_layer.getDeltaWeight();
    //std::cout<<m1<<std::endl; //should print zero matrix
    NetNode<double> n5 = hidden_layer.getDeltaBias();
    BOOST_CHECK(validateVector(v3, n5.getData()));
    std::cout << "End Activation class test" << std::endl;
}
BOOST_AUTO_TEST_SUITE_END()
/**
    Unit test for the DataReader class. Performs general tests.
*/
BOOST_AUTO_TEST_SUITE (ReaderSuite)
BOOST_AUTO_TEST_CASE(mnist_test)
{

    std::cout << "Start Reader class test" << std::endl;
    MNIST_CIFARReader<double> mnist("data/mnist_sample.csv", 784, 10, ',',  false), train("data/mnist_train.csv", 784, 10, ',',  false), test, test2;
    mnist.read();
    BOOST_CHECK(mnist.getFeatureDim() == 784);
    BOOST_CHECK(mnist.getLabelDim() == 10);
    BOOST_CHECK(mnist.getRowDim() == 10);
    //constructor
    MNIST_CIFARReader<double> mnist2(mnist);
    BOOST_CHECK(mnist.getFeatureDim() == mnist2.getFeatureDim());
    BOOST_CHECK(mnist.getLabelDim() == mnist2.getLabelDim() );
    BOOST_CHECK(mnist.getRowDim() == mnist2.getRowDim());
    //shuffleIndex
    std::vector<int> indices;
    mnist.shuffleIndex(indices);
    mnist.trainTestSplit(test, 0.2);
    BOOST_CHECK(mnist.getFeatureDim() == 784);
    BOOST_CHECK(mnist.getLabelDim() == 10);
    BOOST_CHECK(mnist.getRowDim() == 8);
    BOOST_CHECK(test.getFeatureDim() == 784);
    BOOST_CHECK(test.getLabelDim() == 10);
    BOOST_CHECK(test.getRowDim() == 2);
    train.read();
    train.trainTestSplit(test2, 0.1);
    BOOST_CHECK(train.getFeatureDim() == 784);
    BOOST_CHECK(train.getLabelDim() == 10);
    BOOST_CHECK(train.getRowDim() == 54000);
    BOOST_CHECK(test2.getFeatureDim() == 784);
    BOOST_CHECK(test2.getLabelDim() == 10);
    BOOST_CHECK(test2.getRowDim() == 6000);
}
BOOST_AUTO_TEST_CASE(general_test)
{
    GenericReader<double> data("data/xor.dat", 2, 1, ' ', false);
    data.read();
    BOOST_CHECK(data.getFeatureDim() == 2);
    BOOST_CHECK(data.getLabelDim() == 1);
    BOOST_CHECK(data.getRowDim() == 12);
    //shuffleIndex
    std::vector<int> indices;
    data.shuffleIndex(indices);
    //////////////////////////////////////////////////////
    GenericReader<double> test("data/xor_header.dat", 2, 2, ' ', true);
    test.read();
    BOOST_CHECK(test.getFeatureDim() == 2);
    BOOST_CHECK(test.getLabelDim() == 2);
    BOOST_CHECK(test.getRowDim() == 12);
    std::cout << "End Reader class test" << std::endl;
}
BOOST_AUTO_TEST_SUITE_END()
