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

input are a list of ligands and the decided cluster number

* minDist(self,dataset)

only used when you are using a small sample size to visilize the Hierarchical clustering plot
Go over all the nodes and find the smallest distance

* implement_viz(self)
* show(self)

Visualize the hirarchical clusering, only for a small sample size.
First use Implement_viz() and then use show()  
Don't use Implement_viz() for the true analysis with a huge sample

* implement(self)

This is the formal implement function, this function use the parameter K, once the clusters merges into the
into K clusters then this function will break and return the labels for all the elements

* calc_label(self)

Return the analysis result, calculating labels

* leaf_traversal(self)

Travel across all nodes

### PartitionClustering(dataset, k, iteration)

K means is much eaiser considering it don't save the linkage between every two nodes

#### paramemters

Ligands : a list of ligands
k : the number of cluster you decided
iteration : the number of itertation

#### attributes

ID: The ID list
scores: Scores list   
dataset: the score matrix
labels: the cluster label list 
k: the number of cluster you decided
C: use to store all the nodes  

#### Methods

* Initilization

input are a list of ligands and the decided cluster number and the number of iteration

* Implement

Implement() and that's it!

## Example

In this turtorial, we will together initialize a global alignment class and a local alignment class, compute the alignment scores and alignment matrix.
Go to the folder test, open a jupyter notebook. ( Or other editors...)

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
    
    

	  
	




