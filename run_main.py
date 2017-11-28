import os
import sys
import timeit, time
import pickle, gzip
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
sys.path.append(sys.path[0]+"/ann_theano")
from mlp_theano import *
from csv2gz import *


if __name__ == '__main__':
    t1 = time.time()
    #===================================
    print(sys.argv)
    a = int(sys.argv[1]) # 1 is training, 2 is predict
    training_file = sys.argv[2]
    n_in = int(sys.argv[3])
    n_hidden = int(sys.argv[4])
    n_out = int(sys.argv[5])
    learning_rate = 0.3
    batch_size = 200
    #===================================
    if a == 1:
        csv2gz(training_file)
        os.remove(training_file+'.pkl')
        test_mlp(n_in=n_in, n_out=n_out, learning_rate=learning_rate, L1_reg=0.00, L2_reg=0.00, n_epochs=1000,
                 dataset=training_file+'.pkl.gz', batch_size=batch_size, n_hidden=n_hidden)
        predict(training_file+'.pkl.gz', n_hidden=n_hidden, n_in=n_in, n_out=n_out)
    elif a == 2:
        csvtogz('data.csv')
        predict_data_set('data.csv.pkl.gz', n_hidden=n_hidden, n_in=n_in, n_out=n_out)
    else:
        print('Wrong input!')

    t2 = time.time()
    print('time = ', t2 - t1, ' s ')
    print('time = ', (t2 - t1) / 60, ' mins ')
    print(time.asctime(time.localtime(time.time())))

