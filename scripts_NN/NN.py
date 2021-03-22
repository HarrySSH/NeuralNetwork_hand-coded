import math
import random
import string
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skl_metrics
import statistics
import matplotlib.pyplot as plt


random.seed(10000000000)


def rand(a, b):
    '''
    I use the rand function to generate a random number between a and b, (a must be smaller than b) This function is used to initialize the matrix or sampling for some purposes
    '''
    return (b-a)*random.random() + a

def makeMatrix(I, J):
    '''
    makeMatrix function is used to generate a I rows J columns matrix
    '''
    s = (I,J)
    m = np.zeros(s)
    return m

def sigmoid(x):
    '''
    Caculation function. sigmoid reuturn the sigmoid function output, and sigmoid_derivative return the derivative based on the output
    '''

    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(s):
    '''
    Caculation function. sigmoid reuturn the sigmoid function output, and sigmoid_derivative return the derivative based on the output
    '''
    ds = s*(1-s)
    return ds

class NeuralNetwork:
    def __init__(self, input_layer = 68,hidden_layer = 25, output_layer = 1, activation_function= "sigmoid",lr=0.5,seed=1, iteration=10000,batch_size = 20, mf = 0.1, lr_decay = 0.2,print_log = True, print_frequence = 100):
        '''
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
        '''
        # initialize the parameters
        self.input_layer = input_layer + 1  # the additional one is the bias
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.mf = mf
        self.lr = lr
        self.lr_decay =lr_decay
        self.batch_size = batch_size
        self.iteration= iteration
        self.error_log = []

        self.print_log = print_log
        self.print_frequence = print_frequence

    def make_weights(self):
        #use the number of nodes in each layer to make calculation matrix for the propagation between each layer. the value was randomly initialized.
        # make matrix with the makeMatrix function
        self.input_weights = makeMatrix(self.input_layer, self.hidden_layer)
        self.output_weights = makeMatrix(self.hidden_layer, self.output_layer)
        # randomly initialize the matrix
        for i in range(self.input_layer):
            for j in range(self.hidden_layer):
                self.input_weights[i][j] = rand(-2, 2)

        for j in range(self.hidden_layer):
            for k in range(self.output_layer):
                self.output_weights[j][k] = rand(-2, 2)
        # initialize the changes matrix, no need to use it now. So just fill it with zeros
        self.ci = makeMatrix(self.input_layer, self.hidden_layer)
        self.co = makeMatrix(self.hidden_layer, self.output_layer)

    def feedforward(self, inputs):
        # just check that it will be the correct dimensions
        if inputs.shape[0] != self.input_layer - 1:
            raise  ValueError('The nubmer are different')
        if len(inputs.shape) ==1:
            inputs = inputs.reshape(inputs.shape[0],1)
            
        # initialize the input with the row number based on the nodes number, 
        # the column number is based on the batch size

        self.ai = np.ones((self.input_layer, inputs.shape[1]) )
        self.ah = np.ones((self.hidden_layer, inputs.shape[1])) 
        self.ao = np.ones((self.output_layer, inputs.shape[1]))
        # activate the input layer
        self.ai[0:self.input_layer-1]= inputs[0:self.input_layer-1]  # not initialize the bias 
        
        # activate the hidden layer
        sum = np.dot( self.input_weights.T,self.ai)
        self.ah = sigmoid(sum)

        # activate the output 
        sum = np.dot(self.output_weights.T,self.ah )
        self.ao= sigmoid(sum)

    def backPropagate(self,targets, N, M):
        '''
        perform backpropogation, which integrate the errors calculated from this time (bia learning rate) and the direction of last changes from the 
        last interation ( momentum factor) to update the matrix
        '''
        """back probagation"""
        if targets.shape[0] != self.output_layer:
            raise ValueError('The input is inconsistant with the design')

        # build a array representing the difference of the output layer
        output_deltas = np.zeros((self.output_layer,targets.shape[1])) # initialize the nodes
        error = targets - self.ao  # calculate the error
        output_deltas = sigmoid_derivative(self.ao) * error # compute the error for deltas for the hidden layer

        # build a array representing the difference of the hidden layer
        hidden_deltas = np.zeros((self.hidden_layer,targets.shape[1]))
        error = np.dot(self.output_weights,output_deltas)
        hidden_deltas = sigmoid_derivative(self.ah) * error
        
        # update the output matrix
        for j in range(self.hidden_layer): 
            change = output_deltas*self.ah[j]
            if change.shape[1] == 1:
                change = change.reshape((change.shape[0]))
            else:
                change = np.mean(change, axis=1)
            # N*change  update by what the network learn this time
            # M*self.co[j] update date the network via the privous training
            self.output_weights[j] = self.output_weights[j] + N*change + M*self.co[j]
            self.co[j] = change


        #print(change.shape)
        #print(output_deltas*self.ah[j].shape)
        # update the input matrix
        for i in range(self.input_layer): 
            #print(hidden_deltas.shape)
            #print(self.ai[i].shape)
            change = hidden_deltas * self.ai[i]
             #) /hidden_deltas.shape[1]
            if change.shape[1] == 1:  
                change = change.reshape((change.shape[0]))
            else:

                change = np.mean(change, axis=1)
            self.input_weights[i] = self.input_weights[i] + N*change + M*self.ci[i]
            self.ci[i]= change

        # calculate the error
        error = 0.0
        error = 0.5*(targets-self.ao)**2
        return np.sum(error)

    def train(self, inputs, groundtruth):
        # N：learnning rate
        # M：momentum factor
        N=self.lr
        M=self.mf
        lr_decay = self.lr_decay
        epoch = self.iteration
        #print(epoch)
        for i in range(epoch):
            error = 0.0
            time = 0  # an indicator to indicator how many batches have been taken for calculation from the whole batch dataset
            while( 1== 1):
                if len(inputs) - self.batch_size * (time +1)>0:
                    # get the input based on the size number
                    inputs_batch = inputs[time * self.batch_size:(time+1) * self.batch_size]
                    inputs_batch = np.array(inputs_batch)
                    
                    groundtruth_batch = np.array(groundtruth[time * self.batch_size:(time+1) * self.batch_size])
                    # feedford, need to transpose the matrix to fit the right dimension
                    self.feedforward(inputs_batch.T)
                    # sum the error
                    error = error + self.backPropagate(groundtruth_batch.T, N, M)
                    time = time +1
                else:
                    # just in case that the rest dataset have less samples than the batch size
                    inputs_batch = np.array(inputs[time * self.batch_size:])
                    groundtruth_batch = np.array(groundtruth[time * self.batch_size:])
                    self.feedforward(inputs_batch.T)
                    error = error + self.backPropagate(groundtruth_batch.T, N, M)
                    time = time +1
                    # time to shop this loop
                    break

            self.error_log.append(error)
            if self.print_log == True:
                
                if i % self.print_frequence == 0:
                    # print the error value after certain number of iteration
                    print('Epoch: '+str(i)+' Error %-.5f' % error)
            if i % 2000 == 0:
                # decrease the learning rate
                N = N * lr_decay


    def viz(self):
        # viz the plot
        plt.plot(range(1,self.iteration+1), self.error_log)


    def test(self, inputs, groundtruth = 0):
        # use the train matrix to predict the data
        inputs = np.array(inputs)
        self.feedforward(inputs)
        return self.ao
        



def KFold_split(inputs, groundtruth, k_folds):
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
    assert len(inputs) == len(groundtruth), 'The training set and testing set do not match!'
    indexs = list(range(0,len(inputs)))
    random.shuffle(indexs)  # shuffle the order
    #random.shuffle(indexs)
    #random.shuffle(indexs)
    part_num = int(len(inputs)/k_folds)
    print(part_num)
    K_folds = []
    # run a loop to load the whole dataset into K pieces
    for i in range(k_folds):
        
        if i < k_folds -1:
            
            _index = indexs[i*part_num: (i+1)* part_num]
            print(_index)
            _inputs = [inputs[x] for x in _index]
            _groundtruth = [groundtruth[x] for x in _index]
            a_fold = [_inputs, _groundtruth]
            K_folds.append(a_fold)
        else:
            _index = indexs[i*part_num: ]
            print(_index)
            _inputs = [inputs[x] for x in _index]
            _groundtruth = [groundtruth[x] for x in _index]
            a_fold = [_inputs, _groundtruth]
            K_folds.append(a_fold)            
    return K_folds
    
    
    
    
# plot the AUROC curve
import sklearn.metrics as skl_metrics
def AUROC_cruve(trained_NN, inputs, outputs, Fig = False):
    '''
    This function leverage the trained neurla network 
    to predict the probability of the target seuqnce is the candidate or not
    And will draw the ROC plot 
    '''
    results = []
    for i in range (len(inputs)):
        
        results.append(trained_NN.test(inputs[i])[0][0])
    #results = results.reshape((result.shape[0]))
    # draw the AUC cruve
    fpr, tpr, _ = skl_metrics.roc_curve(outputs, results, pos_label=1)
    roc_display = skl_metrics.RocCurveDisplay(fpr=fpr, tpr=tpr)
    # decide if the function show the score or show the plot
    if Fig == False:
        
        return skl_metrics.roc_auc_score(outputs, results)
    else:
        
        return roc_display
    #roc_display = skl_metrics.RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
