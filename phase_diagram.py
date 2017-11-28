import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import ann
from numpy.random import rand

def generating(n, T, numlist):
    data = np.zeros((numlist, n * n + 1))
    h = 2 * np.random.randint(2, size=(n, n)) - 1
    pt = 0

    for p1 in range(100 * n * n):
        x1 = np.random.randint(n)
        x2 = np.random.randint(n)
        s = h[x1, x2]
        nb = h[(x1 + 1) % n, x2] + h[x1, (x2 + 1) % n] + h[(x1 - 1) % n, x2] + h[x1, (x2 - 1) % n]
        cost = 2 * s * nb
        if cost < 0:
            s *= -1
        elif rand() < np.exp(-cost * (1.0 / T)):
            s *= -1

        h[x1, x2] = s

    for p1 in range(numlist * n * n):
        x1 = np.random.randint(n)
        x2 = np.random.randint(n)
        s = h[x1, x2]
        nb = h[(x1 + 1) % n, x2] + h[x1, (x2 + 1) % n] + h[(x1 - 1) % n, x2] + h[x1, (x2 - 1) % n]
        cost = 2 * s * nb
        if cost < 0:
            s *= -1
        elif rand() < np.exp(-cost * (1.0 / T)):
            s *= -1

        h[x1, x2] = s

        if (np.mod(p1, n * n) == 0):
            data[pt, :] = np.append(1, np.asarray(h.reshape(-1)))
            pt += 1

    return data



def main():
    n = 5
    nin = n * n + 1
    m = 2 # ...
    print('processing...')

    w1 = ann.build_input('w1.csv')
    w2 = ann.build_input('w2.csv')

    # make phase diagram
    outdata = np.zeros((10,2))
    listT = np.linspace(1.0, 3.5, 10)
    f = open('data.csv', 'wb')
    for p1 in range(10):
        nT = listT[p1]
        make_data = generating(n, nT, 3000)
        np.savetxt(f, make_data, fmt='%s', delimiter=',')
        temp_output = ann.make_neuron_ann2(make_data, w1, w2)
        outdata[p1, :] = sum(abs(temp_output))/3000
    f.close()
    print(temp_output.shape, outdata.shape)
    np.savetxt('outdata.txt', outdata)
    fig = plt.figure()
    plt.plot(listT, outdata[:, 0], '-bo', label='label 0')
    plt.plot(listT, outdata[:, 1], '-ro', label='label 1')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
    fig.savefig('phase.png', dpi=fig.dpi)


if __name__ == '__main__':
    main()
