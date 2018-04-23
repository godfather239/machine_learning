#!/usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from DataParser import *



def scatter_sample():
    parser = DataParser('./watermelon_data_3a.txt', ',', True)
    dataset = parser.load_data()