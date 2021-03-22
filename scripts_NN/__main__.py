#!/usr/bin/env python
# coding: utf-8

# In[34]:


import glob
import os
import sys
import pandas as pd
import numpy as np
from random import sample
from fasta_reader import read_fasta
from NN import NeuralNetwork as NeuralNetwork, rand, KFold_split
import random
from i_o import encoder


# # Part 1: Autoencoder implementation

# In[36]:


# stimutate a 8 elements vector randomly to build the training set

empty_vec = np.zeros(8)
indexs = range(0,80)


    

# generate a dataset with 100 samples
inputs = []
groundtruth = []
for i in range(0,8):
    k =i %8 
    vec = empty_vec.copy()
    vec[k] =1
    
    inputs.append(vec)
    groundtruth.append(vec)
    


# In[50]:


Autoencoder = NeuralNetwork(input_layer=8, hidden_layer= 3, output_layer=8, batch_size= 8, print_frequence=1000,iteration= 50000, lr = 0.5,mf = 0.1)
Autoencoder.make_weights()
Autoencoder.train(inputs, groundtruth)


# In[51]:


# training curve
Autoencoder.viz()


# In[52]:


print("the test for [1,0,0,0,0,0,0,0] is", Autoencoder.test(np.array([1,0,0,0,0,0,0,0])).reshape(8))
print("the test for [0,1,0,0,0,0,0,0] is", Autoencoder.test(np.array([0,1,0,0,0,0,0,0])).reshape(8))
print("the test for [0,0,1,0,0,0,0,0] is", Autoencoder.test(np.array([0,0,1,0,0,0,0,0])).reshape(8))
print("the test for [0,0,0,1,0,0,0,0] is", Autoencoder.test(np.array([0,0,0,1,0,0,0,0])).reshape(8))
print("the test for [0,0,0,0,1,0,0,0] is", Autoencoder.test(np.array([0,0,0,0,1,0,0,0])).reshape(8))
print("the test for [0,0,0,0,0,1,0,0] is", Autoencoder.test(np.array([0,0,0,0,0,1,0,0])).reshape(8))
print("the test for [0,0,0,0,0,0,1,0] is", Autoencoder.test(np.array([0,0,0,0,0,0,1,0])).reshape(8))
print("the test for [0,0,0,0,0,0,0,1] is", Autoencoder.test(np.array([0,0,0,0,0,0,0,1])).reshape(8))


# # Part 2: Adapt for classification, and develop training regime

# ### Describe your process of encoding your training DNA sequences into input vectors in detail. Include a description of how you think the representation might affect your network’s predictions.
# 
# 
# The DNA sequence havs A T C G, all need to be treat differently but evenly.
# So I decide that for each A, T , C, G we use a vector to represent it. 
# A as [1,0,0,0]
# T as [0,1,0,0]
# C as [0,0,1,0]
# G as [0,0,0,1]
# 
# For the full seuqnce we merge them together. Add them one by one based on the sequence of nucleotide.
# 
# This method assme that the contribution of each nucleotide to the prediction are equally to each other. 
# 
# If A, T, C, G are encoded as 1, 2, 3, 4, then these values can affect training not equally, which is a bad idea.
# 
# However, if there are certain pattern of seuqnce which can have special results:
#     for example, if sequence 'ATCGA' affect the result from an expontail scale, I think my encode method won't be able to detect that. 
#     
#     
#     However, based on my following results I think my encoding method works relavately well for the TF bnding site recogniation.
#     So I will go with this for now unless there is more information that can guibe me to weight these four nucleotide differently.
#     
#     
# 
# 

# ### Describe your training regime. How was your training regime designed so as to prevent the negative training data from overwhelming the positive training data?
# 
# 
# 
# For building the training dataset. Since there is a huge amount of negative seuqneces I can use. I need to be careful because 
# if I use all the possible negative sites, the compiutation will be forever, and the worst case would be the model will tend to bindly turn whatever unknwoen sequences into the negative becauuse the negative sites are too overwhleming. 
# 
# To avoid this situation, I decide to use all 137 positive examples but randomly sample 137 negative samples from the negative seuqnces
# Following these rules:
#     1) all are taken from the negative seuqnces
#     2) length are the same
#     3) will at least have two mismatches with all the positive sites
#     
#     
# To avoid the misleading from the a batch. for loading the dataset, every time once I load a positive sites I also load a negative sites.
# In this way I can make sure everytime I train a batch, there is 50 percent of positive sites and 50 percent of negative sites.
# 
# 
# 
# 
# 
# 

# In[53]:


# Train the network to recogize the TF binding site


# positive sites
positive_sites = []
with open('../data/rap1-lieb-positives.txt','r') as F:
    lines = F.readlines()
    for line in lines:
        line = line.split('\n')[0]
        positive_sites.append(line)
        
        
# negative sites
p = 0
negative_background = []
with read_fasta("../data/yeast-upstream-1k-negative.fa") as file:
    for seq in file:
        
        negative_background.append(seq.sequence)
        
        
# Prepare the negative sites:


# define a function return the different nucleotide number between two sequences
def diff_num(string1, string2):
    num = 0
    for i in range(len(string1)):
        if string1[i] != string2[i]:
            
            num +=1 
    return num


# randomly smaple 137 17-neocletide-length seuqnces from the negative dataset
negative_sub_backgroundsample = sample(negative_background,150,) # took a little more than 137, in case need to remove the repeated ones or the ones that very similar to the positive seqs
negative_sites = []
for a_negative_string in negative_sub_backgroundsample:
    length = len(a_negative_string)
    flag= False
    while (flag == False):
        
        start = int((length -17)*random.random())
        a_negative_string = a_negative_string[start: start + 17]
        for positive_site in positive_sites:
            flag= True
            if diff_num(positive_site, a_negative_string) <2:
                flag= False
                break
            
    assert flag == True, 'Something wrong!'
    
    negative_sites.append(a_negative_string)
    negative_sites = list(set(negative_sites))
    if len(negative_sites) == 137:
        break  
    


# In[54]:


positive_sites_encoded = encoder(positive_sites)
negative_sites_encoded = encoder(negative_sites)


# ### Provide an example of the input and output for one true positive sequence and one true negative sequence.

# In[60]:


print('Input positive sequence')
print(positive_sites[0])
print('Output positive seuqnce')
print(positive_sites_encoded[0])


print('Input negative sequence')
print(negative_sites[0])
print('Output negative seuqnce')
print(negative_sites_encoded[0])


# In[61]:



inputs = []
groundtruth = []
for i in range(len(positive_sites_encoded)):
    inputs.append(positive_sites_encoded[i])
    groundtruth.append([1])
    inputs.append(negative_sites_encoded[i])
    groundtruth.append([0])


# In[62]:


TFs_NN = NeuralNetwork(input_layer=68, hidden_layer= 25, output_layer=1, batch_size=137, print_frequence=1000, iteration=5000)


# In[58]:


TFs_NN.make_weights()


# In[59]:


TFs_NN.train(inputs,groundtruth)


# ### Describe your network architecture, and the results of your training. How did your network perform in terms of minimizing error?
# 
# 
# For the design, I used a input layer of 68 nodes with an addtional node as the bias.
# For the hidden layer I used 25 nodes.
# For the output layer, I used one node as the probability of this binding site is a positive binding site. 1 as the True positive.
# 0 as the negative sequences.
# 
# I use the gradient descnet, and for the loss function I use the minimal sqaure error (MSE)
# 
# So far, it looks like the training went very well. But it needs more support from the following k flod validation.
# 

# ### What was your stop criterion for convergence in your learned parameters? How did you decide this?
# 
# I will think the error converge if the MSE doesn't decrease any more.
# Or the MSE is swing around a specfic value for a long time

# # Part 3: Cross-validation

# ### How can you use k-fold cross validation to determine your model’s performance?
# 
# k fold valdation is a powerful method if we believe that my data has a limitation
# Because it ensures that every observation from the original dataset has the chance of appearing in 
# training and test set.
# 
# I will random split my dataset into several pieces. Each time hold one part for the testing and use all the other parts for the training.
# Use this method to evaluate the model performance and make sure that my model can be trained fairly.
# 

# ### Given the size of your dataset, positive and negative examples, how would you select a value for k?
# 
# 
# Usually if you are having lots of dataset. The ideal cluster is 10. However, given that my dataset is not big, I decide to do k = 5. 
# In this way,  my training and testing data are 80% , 20% split. It is also acceptable.
# 
# 

# In[63]:


K_fold_split = KFold_split(inputs, groundtruth, 5)


# ### Using the selected value of k, determine a relevant metric of performance for each fold. Describe how your model performed under cross validation.

# In[8]:


import matplotlib.pyplot as plt


# In[10]:


for i in range(len(K_fold_split)):
    print('The validation: '+str(i+1))
    testing_set = K_fold_split[i]
    # test_set
    test_inputs = testing_set[0]
    test_groundtruth = testing_set[1]
    training_inputs = []
    training_groundtruth = []
    # training_set
    for j in range(len(K_fold_split)):
        if i ==j:
            pass
        else:
            assert i != j, ' something wrong.'
            training_inputs = training_inputs + K_fold_split[j][0]
            training_groundtruth = training_groundtruth + K_fold_split[j][1]
    # train the model
    NN = NeuralNetwork(input_layer=68, hidden_layer= 25, output_layer=1, batch_size=137, iteration=1000)
    NN.make_weights()
    NN.train(training_inputs, training_groundtruth)
    fig = AUROC_cruve(NN,test_inputs, test_groundtruth, Fig= True)
    
    print('Plot the training curve')
    
    f1, ax1= plt.subplots()
    NN.viz()
    #f1.show()
    
    fig.plot()
       
    
    


# # Part 4: Extension

# ### Try something fun to improve your model performance! This should include implementation of alternative optimization methods (particle swarm, genetic algorithms, etc), you can also optionally add changes in the network architecture such as modifying the activation function, changing the architecture, adding regularization etc. For this section, we want to see a description of what you want to try and why. As long as we have this, and some effort towards implementation, you will get full points.
# 
# ● What set of learning parameters works the best? Please provide sample output
# from your system.
# 
# ● What are the effects of altering your system (e.g. number of hidden units or
# choice of kernel function)? Why do you think you observe these effects?
# 
# ● What other parameters, if any, affect performance?

# #### Answers:
# 
# I performed a gnetic algorthim. Using six hyperparameters: 
# 1) nodes of hidden layer 2) learning rate 3) learning decay 4) momentum factor 5) batch size 6) times of iteration
# 
# In theory, since random generate times of interation can make the algorithm take forever. I decide to set it stable for this test mission.
# 
# ps: For this specific case, when I was running I realize with all the combination, I get a AUC score 1. In this case, there is not really something we need to optimize. It will be worthwhile to consider use the MSE as the parameter to improve in the future.

# In[60]:


genetic_algorithm(training_inputs, training_groundtruth, test_inputs, test_groundtruth,
                     4,5,2, [10,100], [0,1], [0,1], [0,1], [1, 274], [100,101])


# 
# 
# [87, 0.29119834160481906, 0.4555150156548764, 0.017018243282391543, 24, 100] are the one that work the best. But I don't buy this too much since overall the model work very well. 
# 
# THese parameters are definitely affect the system. But unfortunately I didn't observe these effect very obviously in this specific case.
# 
# The parameters which I didn't test are the layers of neural network, here I was only using one hidden layer. In the reall situation adding more layer will generally help the model while at the same time makes it more computational expensive. 

# # Part 5: Evaluate your network on the final set.

# ### Select a final model (encoding, architecture, training regime). This can be the same as your model in Part 3, Part 4, or something completely different.

# #### Since all the model work fine. I decide to use my original model, which is 25 hidden layers, lr = 0.05, batch_size=137, iteration=10000, others are as the default.
# 

# In[67]:


TFs_NN = NeuralNetwork(input_layer=68, hidden_layer= 25, output_layer=1, batch_size=137, lr = 0.05,print_frequence=1000, iteration=10000)
TFs_NN.make_weights()
TFs_NN.train(inputs,groundtruth)


# In[68]:


test_sites = []
with open('../data/rap1-lieb-test.txt','r') as F:
    lines = F.readlines()
    for line in lines:
        line = line.split('\n')[0]
        test_sites.append(line)


# In[69]:


test_sites_encoded = encoder(test_sites)


# In[71]:


with open('../test_scores.txt','w') as F:
    index = 0
    for seq_encode in test_sites_encoded:
        
        F.write(test_sites[index]+'\t')
        
        F.write(str(TFs_NN.test(seq_encode)[0][0])+'\n')
        index = index +1


# In[ ]:




