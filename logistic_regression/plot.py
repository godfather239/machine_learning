#!/usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from DataParser import *


def scatter_sample():
    parser = DataParser('./watermelon_data_3a.txt', ',', True)
    dataset = parser.load_data()

    X = dataset[:, 0:2]
    y = dataset[:, 2]
    f1 = plt.figure(1)
    plt.title('watermelon_3a')
    plt.xlabel('density')
    plt.ylabel('sugar_content')
    plt.scatter(X[y == 0,0], X[y == 0,1], marker = 'o', color = 'k', s=100, label = 'bad')
    plt.scatter(X[y == 1,0], X[y == 1,1], marker = 'o', color = 'g', s=100, label = 'good')
    plt.legend(loc = 'upper right')
    plt.show()


def meshgrid():
    x = np.arange(-5, 5, 0.1)
    y = np.arange(-5, 5, 0.1)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
    h = plt.contourf(x,y,z)

if __name__ == '__main__':
    # scatter_sample()
    meshgrid()
