# refernce : http://iamtrask.github.io/2015/07/12/basic-python-network/
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import copy

def build_input(filename):
    return np.loadtxt(open(filename, "r+b"), delimiter=",")

def make_labels(labels, D, m):
    output = np.zeros((D, m))
    for i in range(D):
        output[i, int(labels[i])] = 1.0
    return output

def forward(x, w):
    z = np.dot(x, w)
    zbar = z < -200
    zbar = zbar.astype(int)
    z = (z * zbar) / np.min(z) - 200 * zbar + z * (1 - zbar)
    return 1 / (1 + np.exp(-z))

def learn_neuron(data, labels, nplus, m, D):
    w = np.random.randn(nplus, m)
    epsilon, reg = 0.1, 0.0

    x = data
    t = labels
    acc = 0
    i = 0

    while True:
        i += 1
        if (acc > 0.87): break
        y = forward(x, w)

        acc = 0
        for p1 in range(D):
            if np.argmax(y[p1, :]) == np.argmax(t[p1, :]):
                acc += 1
        acc = acc/D

        L = 0.5*np.sum(np.square(y-t))/D
        if (np.mod(i, 10) == 0):
            print(i, np.sum(L), acc)

        # # compute dw, db, dLbydz = dLbydy*y*(1-y)
        dLbydy = (y - t)
        dLbydw = np.dot(x.T, dLbydy)

        dw = -epsilon*dLbydw/D

        w = w + dw + (-reg*epsilon*(w))/D

    return w

def learn_neuron_mini_sgd(data, labels, nplus, m, D):
    w = np.random.randn(nplus, m)
    epsilon, reg = 0.1, 0.0

    x = data
    t = labels
    acc = 0
    i = 0
    batchsize = 400

    for i in range(300):
        print(i, acc)
        np.random.shuffle(x)
        for p1 in range(int(D / batchsize)-1):
            mini_batch = x[p1 * batchsize : p1 * batchsize + batchsize, :]
            mini_batch_t = t[p1 * batchsize : p1 * batchsize + batchsize, :]
            y, dLbydw = mini_sgd(mini_batch, w, mini_batch_t)

            acc = 0
            for p2 in range(batchsize):
                if np.argmax(y[p2, :]) == np.argmax(mini_batch_t[p2, :]):
                    acc += 1
            acc = acc / batchsize

            dw = -epsilon * dLbydw / batchsize
            w = w + dw
    return w

def mini_sgd(mini_batch, w, mini_batch_t):
    y = forward(mini_batch, w) # (batch_size, m)
    dLbydy = (y - mini_batch_t)
    dLbydw = np.dot(mini_batch.T, dLbydy)

    return y, dLbydw

def learn_neuron_ann2(data, labels, nplus, m, D):
    hnode = 100
    w1 = 5*np.random.randn(nplus, hnode)
    w2 = 5*np.random.randn(hnode, m)
    epsilon = 50.0

    x = data
    t = labels
    acc = 0.0
    i = 0
    print(D, epsilon)

    while True:
        i += 1
        if (acc > 0.90): break

        y, dLbydw1, dLbydw2 = ann2_mini_sgd(x, w1, w2, t)

        acc = 0.0
        for p1 in range(D):
            if np.argmax(y[p1, :]) == np.argmax(t[p1, :]):
                acc += 1.0
        acc = acc/D

        L = 0.5*np.sum(np.square(y-t))/D
        if (np.mod(i, 100) == 0):
            print(i, np.sum(L), acc)

        # compute dw, db ..... *y*(1-y)
        # dLbydy = (y-t)
        # dLbydw2 = np.dot(h.T, dLbydy)   # (D,d)^T * (D,m)
        #
        # dLbydh = np.dot(dLbydy, w2.T) # (D, d)
        # delta_j = h*(1-h)*dLbydh
        # dLbydw1 = np.dot(x.T, delta_j)

        dw1 = -epsilon*dLbydw1/D
        dw2 = -epsilon*dLbydw2/D

        w1 = w1 + dw1
        w2 = w2 + dw2

    return w1, w2

def learn_neuron_ann2_mini_sgd(data, labels, hnode, m, batchsize, epsilon, epoches):
    D, nplus = data.shape  # D, nplus
    w1 = np.random.randn(nplus, hnode)
    w2 = np.random.randn(hnode, m)

    x = data
    t = labels

    acc = 0.0
    print(nplus, hnode, m, D, batchsize, epsilon)

    for epoch in range(epoches):
        # np.random.shuffle(data)  # bug: to shuffle both x and t !
        print(epoch, acc)
        for p1 in range(int(D / batchsize)):
            mini_batch = x[p1 * batchsize: p1 * batchsize + batchsize, :]
            mini_batch_t = t[p1 * batchsize: p1 * batchsize + batchsize, :]
            y, dLbydw1, dLbydw2 = ann2_mini_sgd(mini_batch, w1, w2, mini_batch_t)

            acc = 0
            for p2 in range(batchsize):
                if np.argmax(y[p2, :]) == np.argmax(mini_batch_t[p2, :]):
                    acc += 1.0
            acc = acc / batchsize

            dw1 = -epsilon * dLbydw1 / batchsize
            dw2 = -epsilon * dLbydw2 / batchsize
            w1 = w1 + dw1
            w2 = w2 + dw2

    return w1, w2

def ann2_mini_sgd(mini_batch, w1, w2, mini_batch_t):
    h = forward(mini_batch, w1)
    h[:, 0] = 1.0
    y = forward(h, w2)

    # compute dw, db ..... *y*(1-y)
    dLbydy = (y - mini_batch_t)
    dLbydw2 = np.dot(h.T, dLbydy)  # (D,d)^T * (D,m)

    dLbydh = np.dot(dLbydy, w2.T)  # (D, d)
    delta_j = h * (1 - h) * dLbydh
    dLbydw1 = np.dot(mini_batch.T, delta_j)

    return y, dLbydw1, dLbydw2


def test_neuron_ann1(data, labels, nplus, m, D, w):
    x = data
    t = labels
    y = forward(x, w)

    acc = 0
    for p1 in range(D):
        if np.argmax(y[p1, :]) == np.argmax(t[p1, :]):
            acc += 1
    acc = acc/D

    L = 0.5*np.sum(np.square(y-t))/D
    print('Test result: ')
    print(' L = ', np.sum(L), ' acc = ', acc)

def test_neuron_ann2(data, labels, nplus, m, D, w1, w2):
    x = data
    t = labels
    h = forward(x, w1)
    h[:, 0] = 1.0
    y = forward(h, w2)

    acc = 0.0
    for p1 in range(D):
        if np.argmax(y[p1, :]) == np.argmax(t[p1, :]):
            acc += 1.0
    acc = acc/D

    L = 0.5*np.sum(np.square(y-t))/D
    print('Test result: ')
    print(' L = ', np.sum(L), ' acc = ', acc)

def make_neuron_ann2(data, w1, w2):
    x = data
    h = forward(x, w1)
    h[:, 0] = 1.0
    y = forward(h, w2)
    return y

def neuron_ann1(data, w):
    x = data
    y = forward(x, w)
    print('neuron: ')
    print(np.argmax(y))


def neuron_ann2(data, w1, w2):
    hnode = 21
    x = data
    h = np.zeros((x.shape[0], hnode))
    h[:, 0] = 1

    h = forward(x, w1)
    h[:, 0] = 1.0
    y = forward(h, w2)
    print('neuron: ')
    print(np.argmax(y))


