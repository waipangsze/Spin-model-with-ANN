""" use> python csv2gz.py filename
for training set """

import theano
import pickle
import numpy
import gzip
import sys, time

def csv2gz(filename):
    t1 = time.time()
    print('='*80)
    print('The', filename, 'will be converted to', filename +'.pkl.gz', '\n')

    my_train_x = []
    my_train_y = []
    my_valid_x = []
    my_valid_y = []
    my_test_x = []
    my_test_y = []

    # check the size of csv.file
    data = numpy.genfromtxt(filename, delimiter=',')
    numberD, numberinput = data.shape
    print('The training set number and size of input = ', data.shape)
    print('This set will be divided into three parts as training set (0.8), validation set(0.1) and test est(0.1)')
    print('The size of training set = ', 0.8*numberD)
    print('The size of validation set = ', 0.1*numberD)
    print('The size of test set = ', 0.1*numberD, '\n')

    # for i, line in enumerate(fp):
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                print('The first one size = ', len(line.split(',')))
            if i < 0.8*numberD:
                my_list = line.split(',') # replace with your own separator instead
                my_train_x.append(my_list[1:]) # omitting identifier in [0] and target in [-1]
                my_train_y.append(my_list[0])
            elif 0.8*numberD <= i < 0.9*numberD:
                my_list = line.split(',')  # replace with your own separator instead
                my_valid_x.append(my_list[1:])  # omitting identifier in [0] and target in [-1]
                my_valid_y.append(my_list[0])
            else:
                my_list = line.split(',')  # replace with your own separator instead
                my_test_x.append(my_list[1:])  # omitting identifier in [0] and target in [-1]
                my_test_y.append(my_list[0])

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    print('training set = ', len(my_train_x), len(my_train_x[0]), len(my_train_y), len(my_train_y[0]))
    print('valid set = ', len(my_valid_x), len(my_valid_x[0]), len(my_valid_y), len(my_valid_y[0]))
    print('test set = ', len(my_test_x), len(my_test_x[0]), len(my_test_y), len(my_test_y[0]))

    f = open(filename+'.pkl', 'wb')
    pickle.dump([(my_train_x, my_train_y), (my_valid_x, my_valid_y), (my_test_x, my_test_y)], f)
    f.close()

    f_in = open(filename+'.pkl', 'rb')
    f_out = gzip.open(filename+'.pkl.gz', 'wb')
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()

    t2 = time.time()
    print('time = ', t2 - t1, ' s ')
    print('time = ', (t2 - t1) / 60, ' mins ')
    print(time.asctime(time.localtime(time.time())))
    print('='*80)
