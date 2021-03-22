def encoder(sequence):
    '''
    The DNA sequence havs A T C G, all need to be treat differently but evenly.
    So I decide that for each A, T , C, G we use a vector to represent it. 
    A as [1,0,0,0]
    T as [0,1,0,0]
    C as [0,0,1,0]
    G as [0,0,0,1]
    '''
    # encode all the sequences

    A = [1,0,0,0]
    T = [0,1,0,0]
    C = [0,0,1,0]
    G = [0,0,0,1]

    sequence_encoded = []
    for a_string in sequence:
        encode = []
        for letter in a_string:
            if letter == 'A':
                encode = encode + A
            elif letter == 'T':
                encode = encode + T
            elif letter == 'C':
                encode = encode + C
            elif letter == 'G':
                encode = encode + G
            else:
                assert 1==2, 'There is a mistake here'
        #assert len(encode) == 68, 'There is a mistake here'
        sequence_encoded.append(encode)
    return sequence_encoded
