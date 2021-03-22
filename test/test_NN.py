import pytest
import sys
import numpy as np
sys.path.append("..")
from scripts_NN import NN#.NeuralNetwork as NeuralNetwork#, rand
from scripts_NN import i_o#as encoder

def test_encoder():
    seq = 'ATCG'
    encodes = i_o.encoder([seq])
    assert len(encodes[0]) == 16, 'The encoder does not work normally!'
    assert encodes[0] == [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],'The encoder does not work normally!'

def test_dimension():
    '''
    This one check if the Neuralnetwork have the correct dimension
    '''
    random_NN = NN.NeuralNetwork(6,14,6)
    random_NN.make_weights()
    assert random_NN.input_layer  == 7, "The dimension isn't correct !"
    assert random_NN.hidden_layer == 14, "The dimension isn't correct !"
    assert random_NN.output_layer  == 6, "The dimension isn't correct !"
    assert random_NN.input_weights.shape == (7,14),"The dimension isn't correct !"
    assert random_NN.output_weights.shape == (14,6),"The dimension isn't correct !"

def test_training():
    '''   
    This function give a simple test to the network to see if the network can do this drop
    '''
    empty_vec = np.zeros(8)

    inputs = []
    groundtruth = []
    for i in range(0,4):
        k =i %8 
        vec = empty_vec.copy()
        vec[k] =1

        inputs.append(vec)
        groundtruth.append(vec)


    Autoencoder = NN.NeuralNetwork(input_layer=8, hidden_layer= 5, output_layer=8, batch_size= 4, print_frequence=1000, lr = 0.5)
    Autoencoder.make_weights()  
    Autoencoder.train(inputs, groundtruth)
    results = Autoencoder.test(np.array([1,0,0,0,0,0,0,0])).reshape(8)
    assert results[0] > 0.8, 'The training is not successful!'
    assert results[1] < 0.2, 'The training is not successful!'
    assert results[2] < 0.2, 'The training is not successful!'
    assert results[3] < 0.2, 'The training is not successful!'
    assert results[4] < 0.2, 'The training is not successful!'
    assert results[5] < 0.2, 'The training is not successful!'
    assert results[6] < 0.2, 'The training is not successful!'
    assert results[7] < 0.2, 'The training is not successful!'
    




    
    

