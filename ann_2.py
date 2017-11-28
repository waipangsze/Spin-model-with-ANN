import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import ann


def main():
    nin = 5 * 5 + 1
    m = 2 # ...

    data = ann.build_input('t1.csv')  # 1 + 20*20
    tD, nin = data.shape
    trainning = int(tD*0.8)
    labels = np.copy(data[0:trainning, 0]) # D*1
    data = data[0:trainning, :]
    data[:,0] = 1   # D*(n+1) , n+1=785, it's bias
    D, nin = data.shape  # D, nplus
    labels = ann.make_labels(labels, D, m)

    hnode = 100
    batchsize = 10
    epsilon = 1.0
    epoches = 20
    w1, w2 = ann.learn_neuron_ann2_mini_sgd(data, labels, hnode, m, batchsize, epsilon, epoches)
    np.savetxt('w1.csv', w1, delimiter=',')
    np.savetxt('w2.csv', w2, delimiter=',')
	
    test_data = ann.build_input('t1.csv')
    tD, nin = test_data.shape
    trainning = int(tD * 0.8)
    test_labels = np.copy(test_data[trainning+1:, 0]) # D*1
    test_data = test_data[trainning+1:, :]
    test_data[:, 0] = 1
    D, nin = test_data.shape  # D, nplus
    test_labels = ann.make_labels(test_labels, D, m)
    ann.test_neuron_ann2(test_data, test_labels, nin, m, D, w1, w2)



if __name__ == '__main__':
    main()

