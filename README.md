# NeuralNetwork_HW3

Implement Neuralnetwork


- [Background](#background)
- [Usage](#usage)
- [API](#api)
- [Example](#example)

## Background
In this assignment, I will implement a simple neural network with a input layer, a hidden layer, an output layer. Also I will write a K fold validation function and a genetic algorthim for optimization

## Usage 
Common neural network

## API

### Function def rand(a, b):
    '''
    I use the rand function to generate a random number between a and b, (a must be smaller than b) This function is used to initialize the matrix or sampling for some purposes
    '''
### Function makeMatrix(I, J):
    '''
    makeMatrix function is used to generate a I rows J columns matrix
    '''
### def sigmoid(x):
    '''
    Caculation function. sigmoid reuturn the sigmoid function output, and sigmoid_derivative return the derivative based on the output
    '''
### def sigmoid_derivative(s):
    '''
    Caculation function. sigmoid reuturn the sigmoid function output, and sigmoid_derivative return the derivative based on the output
    '''
    
###  class NeuralNetwork:
    def __init__(self, input_layer = 68,hidden_layer = 25, output_layer = 1, activation_function= "sigmoid",lr=0.5,seed=1, iteration=10000,batch_size = 20, mf = 0.1, lr_decay = 0.2,print_log = True, print_frequence = 100):

#### Parameters:

input_layer: the int variable of input nodes plus one [as the bias] 
hidden_layer: the int variable of  hidden_layer
output_layer: the int variable of output_layer
mf: momentum factor
lr: learning rate
batch_size: batch size for the training, can use to speed up the training but should be careful about putting too much samples into the training
lr_decay: the percent of learning reate decrease after certain time of iteration
iteration: the total times of iteration
error_log: a list variable which store the error value after each interation. This value is used with the viz() function
print_log: bool variable, the default is True, which mean it will print error after certain times of iteration to tell us how the training process did
print_frequence: decide how often should the training function print the error value

#### Attributes:

ai: input layer array
ah: hidden layer array
ao: output layer array
       # with no batch, they are one demension array, with the batch size, they are matrix with the col_num = batch_size
input_weights: use the number of nodes in each layer to make calculation matrix for the propagation between each layer. the value was randomly initialized.
output_weights:  use the number of nodes in each layer to make calculation matrix for the propagation between each layer. the value was randomly initialized.
           THese two matrix will be used for matrix calculation
ci:
co:  THese two have the same dimension as the input_weights/output_weights, are storing the changes computed by the back propogation


#### Method:    
	    
* Initialization

PASS 

* def make_weights(self):
 
use the number of nodes in each layer to make calculation matrix for the propagation between each layer. the value was randomly initialized. make matrix with the makeMatrix function

* feedforward(self, inputs):

Perform the matrix mutiplication and compute the next layer with the valuyes of the layer and the related matrix.

* backPropagate(self,targets, N, M):

perform backpropogation, which integrate the errors calculated from this time (bia learning rate) and the direction of last changes from the last interation ( momentum factor) to update the matrix

* train(self, inputs, groundtruth):

The core training function. Perform the feedforward and backpropagate for the a cerntain number  of times (epoch). Update the network with the learning rate.

* test(self, inputs, groundtruth = 0):
 
use the train matrix to predict the data

* viz(self):

viz the plot

### Function encoder(sequence):

The DNA sequence havs A T C G, all need to be treat differently but evenly.
So I decide that for each A, T , C, G we use a vector to represent it. 
A as [1,0,0,0]
T as [0,1,0,0]
C as [0,0,1,0]
G as [0,0,0,1]
input is a list of string, and the return will be a list with all binary values (0,1)

### Function def KFold_split(inputs, groundtruth, k_folds):
    '''
    This function will random split the whole dataset into k parts evenly. 
    For the following process, each time we will the most parts for training but each time left a part for testing
    This function will return a list like this:
    # [[input_set_1, groundtruth_set_1],
    [input_set_1, groundtruth_set_1]
    [input_set_2, groundtruth_set_2]
    [input_set_3, groundtruth_set_3]
    ...
    [input_set_n, groundtruth_set_n]]
    '''

### Function def cross(Father, Mother, training_inputs, training_groundtruth, test_inputs, test_groundtruth ):
    '''
    This cross function is used for the genetic algorithm optimization. the inputs are two individuals, and will return a "child" with a better training performance.
    This function will first generate two children. And for each child's one single feature, it will either be from the 'mother' or the 'father'
    And then a quick training and testing process will be used to evalue the AUROC score.
    And the 'child' with a better performance will be returned.
    '''
 
### FUnction def mutate(individual):
    '''
    for each feature of individuals, the feature's value is shitfed with in 10 percent variance. 
    this is based on our thoughts that mutated should not be a lot.
    '''
    
### def genetic_algorithm(training_inputs, training_groundtruth, test_inputs, test_groundtruth, num_population,times, invasion, hidden_nodes, lr, lr_decay, mf, batch_size, epoch):
    
    '''
    In this function, I will perform the genetic_algorithm to find out the best combination of hyperparameters combination for the training.
    [training_inputs, training_groundtruth, test_inputs, test_groundtruth] is a training, testing set which are obtained for evalute the training performance.
    num_population: the number of random candidates with the random number of feathers.
    times: total times of "cross" for the parents to exchange the features which uis aiming at getting 'progeny" with the better performance.
    invasion: number of new "invasion" individuals, besides the progeny got from the 'cross', for each time I also introduce some new individuals with different features, 
    which is aimming at making the whole population get more information for optimization.
    
    [hidden_nodes, lr, lr_decay, mf, batch_size, epoch] are the six features i planned to use as the hyperparameter features that needs to be optimized
    They are all put in as the two-element list, which represents the range. In the function, I will use rand funvtion to generate a random number between the range.
    '''

## Example

In this turtorial, I will build a autoencoder network 

* Initialization

Load packages

    import glob
    import pandas as pd
    import numpy as np
    import os 
    import sys
    from algs import Ligand, HierarchicalClustering, PartitionClustering,Silhouette_Coefficient, ClusterNode, getEuclidean, Rand_Index
    import matplotlib.pyplot as plt
    from __future__ import print_function
    import time
    import numpy as np
    import pandas as pd
    from sklearn.datasets import fetch_openml
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    %matplotlib inline
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import seaborn as sns
    import json


Take a subset of all the ligands and perform kmeans and hirarchical clustering

    csv = pd.read_csv('../ligand_information.csv', index_col= False)
    Data = list(np.random.randint(1,csv.shape[0],size=100))
    Ligands = []
    for i in Data:

        a_ligand = Ligand(csv[csv.index==i])
        Ligands.append(a_ligand)

		
Perform hirarchical clustering, and visualize the results

    Hiera_C = HierarchicalClustering(Ligands, 5)
    results = Hiera_C.implement_viz()
    Hiera_C.show(results, len(Ligands))
    # run Hiera_C.implement_viz() and Hiera_C.show() to visualize the results. But this can run very slow so please use a small subset.
    
Perform hirarchical clustering, keams clustering

    # Can use a higher 
    Data = list(np.random.randint(1,csv.shape[0],size=1000))
    Ligands = []
    for i in Data:

        a_ligand = Ligand(csv[csv.index==i])
        Ligands.append(a_ligand)
    
    
    Hiera_C = HierarchicalClustering(Ligands, 5)
    Hiera_C.implement()
    
    Kmeans_C = PartitionClustering(Ligands, 5, 10)
    Kmeans_C.implement()
    
    

	  
	




