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

data = np.genfromtxt('validationerror.csv', dtype=float, delimiter=',')
plt.plot(data, 'bo--')
plt.title('Validation Error')
plt.grid()
plt.savefig('validationerror.png')
plt.close()

data = np.genfromtxt('predict_output.csv', dtype=float, delimiter=',')
sdx, sdy = data.shape
meandata = np.zeros((10, 2))
m = 3000
listT = np.linspace(1.0, 3.5, 10)
print(data.shape)
for p1 in range(10):
    meandata[p1, 0] = numpy.mean(data[p1 * m: (p1 + 1) * m, 0])
    meandata[p1, 1] = numpy.mean(data[p1 * m: (p1 + 1) * m, 1])
print(meandata)
plt.plot(listT, meandata, 'bo--')
plt.grid()
plt.savefig('predict.png')
plt.close()
