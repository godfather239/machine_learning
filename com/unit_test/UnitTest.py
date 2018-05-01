#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
sys.path.append('../uniq_random')
import unittest
from UniqRandom import *

class ComUnitTest(unittest.TestCase):
    def test_UniqRandom(self):
        rand_max = 10
        rander = UniqRandom(rand_max)
        for i in range(rand_max * 2):
            print 'random value: %d' % rander.rand()


if __name__ == '__main__':
    unittest.main()