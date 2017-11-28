import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand

def energy(n, h):
    energy = 0
    for p1 in range(n):
        for p2 in range(n):
            x1 = p1
            x2 = p2
            energy = energy - h[x1, x2] * (h[(x1+1)%n, x2] + h[x1, (x2+1)%n])
    return energy

def generating(n, T, numlist, label):
    check = np.zeros((numlist,1))
    data = np.zeros((numlist , n*n + 1))
    h = 2*np.random.randint(2, size=(n,n))-1
    print(h)
    pt = 0

    for p1 in range(100 * n * n):
        x1 = np.random.randint(n)
        x2 = np.random.randint(n)
        s = h[x1, x2]
        nb = h[(x1+1)%n, x2] + h[x1, (x2+1)%n] + h[(x1-1)%n, x2] + h[x1, (x2-1)%n]
        cost = 2*s*nb
        if cost < 0:
            s *= -1
        elif rand() <np.exp(-cost*(1.0/T)):
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

        if( np.mod(p1, n*n) ==  0):
            data[pt, :] = np.append(label, np.asarray(h.reshape(-1)))
            check[pt] = energy(n, h)
            pt += 1
    #plt.plot(check)
    #plt.show()

    return data

def main():
    print('hi')
    n = 5
    numlist = 1000
    data1 = generating(n, 0.1, numlist, 0)
    data2 = generating(n, 0.2, numlist, 0)
    data = np.concatenate((data1, data2))
    listT = np.linspace(0.3, 1.0, 8)
    for p1 in range(8):
        data1 = generating(n , listT[p1], numlist , 0)
        data = np.concatenate((data, data1))

    listT = np.linspace(2.9, 6.0, 10)
    for p1 in range(10):
        data1 = generating(n, listT[p1], numlist, 1)
        data = np.concatenate((data, data1))

    for p1 in range(3):
        print(data[p1,:])
    print(data1.shape, data2.shape, data.shape)

    np.random.shuffle(data)

    data = data.astype(int)
    np.savetxt('t1.csv', data, fmt='%s', delimiter=',')
if __name__ == '__main__':
    main()
