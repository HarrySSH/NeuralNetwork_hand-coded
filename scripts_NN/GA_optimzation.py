from NN import rand, NeuralNetwork, AUROC_cruve
import random


def cross(Father, Mother, training_inputs, training_groundtruth, test_inputs, test_groundtruth ):
    '''
    This cross function is used for the genetic algorithm optimization. the inputs are two individuals, and will return a "child" with a better training performance.
    This function will first generate two children. And for each child's one single feature, it will either be from the 'mother' or the 'father'
    And then a quick training and testing process will be used to evalue the AUROC score.
    And the 'child' with a better performance will be returned.
    '''
    Child_1 = []
    Child_2 = []
    for i in range(6):
        coin = rand(-1,1)
        if coin>= 0:
            Child_1.append(Father[i])
            Child_2.append(Mother[i])
        else: 
            Child_1.append(Mother[i])
            Child_2.append(Father[i])
    Child_1 = mutate(Child_1)
    Child_2 = mutate(Child_2)
    

    
    # build the network, make weights, and train it
    NN_1 = NeuralNetwork(input_layer=68, hidden_layer= Child_1[0], output_layer=1,
                            lr = Child_1[1], lr_decay= Child_1[2], iteration= Child_1[5],
                            batch_size= Child_1[4], mf= Child_1[3])
        
    NN_2 = NeuralNetwork(input_layer=68, hidden_layer= Child_2[0], output_layer=1,
                            lr = Child_2[1], lr_decay= Child_2[2], iteration= Child_2[5],
                            batch_size= Child_2[4], mf= Child_2[3])
        
    NN_1.make_weights()
    NN_2.make_weights()

    NN_1.train(training_inputs, training_groundtruth)
    NN_2.train(training_inputs, training_groundtruth)
        
    Score_1 = AUROC_cruve(NN_1, test_inputs, test_groundtruth, Fig=False)
    Score_2 = AUROC_cruve(NN_2, test_inputs, test_groundtruth, Fig=False)
    if Score_1 > Score_2:
        return Child_1, Score_1
    else: 
        return Child_2, Score_2
    



def mutate(individual):
    '''
    for each feature of individuals, the feature's value is shitfed with in 10 percent variance. 
    this is based on our thoughts that mutated should not be a lot.
    '''
    for i in range(6):
        if i in [0,4,5]:
            
            coin = rand(-1,1)
        
            individual[i] = int(individual[i] * (1 + 0.1*coin))  # 10%  variantion of the features
            
        else:
            coin = rand(-1,1)
            individual[i] = individual[i] * (1 + 0.1*coin)
        
    return individual
def genetic_algorithm(training_inputs, training_groundtruth, test_inputs, test_groundtruth,
                     num_population,times, invasion, hidden_nodes, lr, lr_decay, mf, batch_size, epoch):
    
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
    
    
    # generate the parents population
    print('generating '+ str(num_population)+' individuals')
    
    # makeing sure the input are correct
    assert hidden_nodes[0] < hidden_nodes[1], 'something went wrong!'
    assert lr[0] < lr[1], 'something went wrong!'
    assert lr_decay[0] < lr_decay[1], 'something went wrong!'
    assert mf[0] < mf[1], 'something went wrong!'
    assert batch_size[0] < batch_size[1], 'something went wrong!'
    assert epoch[0] < epoch[1], 'something went wrong!'
    

    # generating a bunch of individuals based on the range that provided
    individuals_genom = []
    individuals_phyno = []
    for i in range(num_population):
        # randonly generate the feature
        _hidden_nodes = int(rand(hidden_nodes[0],hidden_nodes[1]))
        _lr = rand(lr[0], lr[1])
        _lr_decay = rand(lr_decay[0], lr_decay[1])
        _mf = rand(mf[0],mf[1])
        _batch_size = int(rand(batch_size[0],batch_size[1] ))
        _epoch = int(rand(epoch[0], epoch[1]))
        
        # build an individual and put it into the whole set
        individuals_genom.append( [_hidden_nodes, _lr, _lr_decay, _mf, _batch_size, _epoch])
        NN = NeuralNetwork(input_layer=68, hidden_layer= _hidden_nodes, output_layer=1,
                           lr = _lr, lr_decay= _lr_decay, iteration= _epoch,
                           batch_size= _batch_size, mf= _mf)
        NN.make_weights()
        NN.train(training_inputs, training_groundtruth)
        # also store the individual's performance in a list vector
        individuals_phyno.append(AUROC_cruve(NN, test_inputs, test_groundtruth, Fig=False))
        
        
    # take the best performance people and keep it for the next generation
    my_phyno = max(individuals_phyno)
    idx = individuals_phyno.index(my_phyno)
    my_genome = individuals_genom[idx]
    
    n = invasion
    # begin the cross, do N times of cross
    for t in range(times):
        print('For the time '+str(t)+' the best candidates and the best result is')
        print(my_genome)
        print(my_phyno)

        if t >=1:   
            if len(individuals_genom) % 2 == 0:  # add the new invasion people,
                                                 # make sure that the number is even 
                add = n
            else:
                add = n+1
            for i in range(add):
        
                _hidden_nodes = int(rand(hidden_nodes[0],hidden_nodes[1]))
                _lr = rand(lr[0] , lr[1])
                _lr_decay = rand(lr_decay[0], lr_decay[1])
                _mf = rand(mf[0],mf[1])
                _batch_size = int(rand(batch_size[0],batch_size[1] ))
                _epoch = int(rand(epoch[0], epoch[1]))

                individuals_genom.append( [_hidden_nodes, _lr, _lr_decay, _mf, _batch_size, _epoch])
                NN = NeuralNetwork(input_layer=68, hidden_layer= _hidden_nodes, output_layer=1,
                                       lr = _lr, lr_decay= _lr_decay, iteration= _epoch,
                                       batch_size= _batch_size, mf= _mf)
                
                
                NN.make_weights()
                NN.train(training_inputs, training_groundtruth)
                individuals_phyno.append(AUROC_cruve(NN, test_inputs, test_groundtruth, Fig=False))
                    
        next_generation_genom = []  
        next_generation_phyno = []
        next_generation_genom.append(my_genome)
        next_generation_phyno.append(my_phyno)
        assert len(individuals_genom) %2 == 0, 'something wrong!'
        for i in range(int(len(individuals_genom)/2)):

            child_genom, child_phyno = cross(individuals_genom[i*2],individuals_genom[ i*2 +1],
                                             training_inputs, training_groundtruth, test_inputs, test_groundtruth)
            # get the child, put the child into the next generation
            next_generation_genom.append(child_genom)
            next_generation_phyno.append(child_phyno)
          
        # next is now the parents
        individuals_genom = next_generation_genom
        individuals_phyno = next_generation_phyno
        # shift it randomly
        index = list(range(len(individuals_genom)))
        random.shuffle(index)
        individuals_phyno = [individuals_phyno[x] for x in index]
        individuals_genom = [individuals_genom[x] for x in index]
        # still get the best performance
        my_phyno = max(individuals_phyno)
        idx = individuals_phyno.index(my_phyno)
        my_genome = individuals_genom[idx]
    
    print('The whole crossing process is done')
    my_phyno = max(individuals_phyno)
    idx = individuals_phyno.index(my_phyno)
    my_genome = individuals_genom[idx]   
    print('the best candidates and the best result is')
    print(my_genome)
    print(my_phyno)
        