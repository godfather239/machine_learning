#!/usr/bin/env python
# -*- coding:utf-8 -*-
import unittest
from DataParser import *


class MyUnitTest(unittest.TestCase):
    def test_load_data(self):
        parser = DataParser('./watermelon_data_3a.txt', ',', True)
        dataset = parser.load_data()
        row = dataset[0]
        assert (row[0]-0.697) < 0.1
        assert row[2] == 1


if __name__ == '__main__':
    unittest.main()
