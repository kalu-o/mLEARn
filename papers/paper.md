---
title: 'mLEARn: An Implementation of Multi-layer Perceptron in C++'
tags:
  - neural networks
  - multi-layer perceptron
  - deep learning
  - stochastic gradient descent
  - C++
authors:
  - name: Kalu U. Ogbureke
    orcid: 0000-0002-8152-4863
    affiliation: 1
affiliations:
 - name: GRLMC, Rovira i Virgili University, Tarragona Spain
   index: 1
date: 22 May 2019
bibliography: ref.bib
---

# Abstract

This paper presents ``mLEARn``, an open-source implementation of multi-layer perceptron
in C++. The techniques and algorithms implemented represent existing approaches in
machine learning. ``mLEARn`` is written using simple C++ constructs. The aim of ``mLEARn``
is to provide a simple and extendable machine learning platform for students in courses
involving C++ and machine learning. The source code and documentation can
be downloaded from https://github.com/kalu-o/mLEARn.

# Introduction

Artificial neural networks (ANNs) is an area of deep learning which has been well studied.
This is because of their importance in many areas, from autonomous driving through speech
technologies [@Graves:2013]. The two main categories are usually recurrent (feedback)
and feed-forward architectures [@Bishop:1995]. Deep learning is currently an active research area in machine learning.

# Statement of Need

There are currently available popular deep learning frameworks such as MXNet, Caffe and TensorFlow. Students
often used these as off-the-shelf machine learning tools and have little or no control over the codes. One of the reasons for 
this is because the codes are advanced and production ready. ``mLEARn`` addresses these as it can be used as an off-the-shelf machine learning tool. Furthermore, the coding style makes it easier to apply what was learnt in machine learning/C++ courses and extend the functionalities. These make it easier to understand machine learning algorithms from first principle and extend state-of-the-art.

# Architecture of mLEARn

The classes implemented in ``mLEARn`` are Node, NetNode, Activation, CostFunction, Layer,
Network, DataReader and Optimizer. The Node class is the fundamental data structure
used; and NetNode is an extension of the Node class used for multi-layer perceptron. 

![](node.PNG)  

The Activation class handles activations in the network. Currently, the functions implemented
are sigmoid, tanh, rectified linear unit (ReLU), leaky ReLU, identity, softmax
and exponential linear unit (ELU). The CostFunction class is responsible for objective/loss
functions. Cost functions implemented are mean squared error (MSE), mean absolute
error (MAE) and cross entropy.
The Network class is a classic MLP consisting of sequences of layers, i.e. one or more
hidden layers and an output layer. The DataReader class is the base class responsible for
reading train/test dataset into features and labels. Three different readers are implemented,
namely MNISTReader, GenericReader and IrisReader. The Optimizer class is the base class responsible
for training algorithms. Three optimizers are currently implemented, namely mini-batch stochastic gradient descent [@Kiefer:1952; @Ruder16], adaptive gradient (Adagrad) [@Duchi:2011] and root mean square propagation (RMSProp) [@Tieleman:2012]. A novel implicit adaptive learning rate method (PSDSquare) implemented is currently being considered for publication by IEEE. An experiment showed comparable results in terms of accuracy on the MNIST digit recognition task [@LeCun:1999]

A number of new enhancements such as automatic differentiation, distributed computing and GPU support will be added in future.

# Acknowledgements

The core work in neural networks and convergence was done as part of the DEA thesis "Training Multi-layer Perceptron using Genetic Algorithms", while the author was with GRLMC at Rovira i Virgili University, Tarragona. The author would
like to acknowledge support for the speech and language technologies program.

# References

