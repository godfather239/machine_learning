#!/usr/bin/env python
# -*- coding:utf-8 -*-
import random


class UniqRandom:
    """
    Uniq random class in [0, rand_max). The rand_max should not be a large number since
    the main procedure is based on an array with length of rand_max. Mass
    memory cost is caused by huge number of rand_max.
    """

    def __init__(self, rand_max):
        self.rand_max = max(rand_max, 1)
        self.arr = None
        random.seed()

    def rand(self):
        if self.arr is None:
            self.arr = range(self.rand_max)
        if self.rand_max == 1:
            return self.arr[0]
        pos = random.randint(0, self.rand_max - 1)
        val = self.arr[pos]
        self.arr[pos], self.arr[self.rand_max - 1] = self.arr[self.rand_max - 1], self.arr[pos]
        self.rand_max -= 1
        return val
