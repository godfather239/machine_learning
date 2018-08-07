#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Author:     wenjieg@jumei.com
@Date:       24/07/2018
@Copyright:  Jumei Inc
"""
import math
from sklearn.tree import DecisionTreeClassifier
import numpy as np


class MyDecisionTreeClassifier:
    def __init__(self, critiron = 'id3'):
        self.critiron = critiron
        self.root = None
        # TODO

    def fit(self, X, y):
        self.generate_decision_tree(X, y)
        pass

    def predict(self, y):
        pass

    def generate_decision_tree(self, X, y):
        self.root = TreeNode()
        X.astype(float, copy=False)
        # feature preprocess
        Z = self.calc_split_points(X)
        self.generate_recursive(self.root, X, y, Z)
        pass

    def calc_split_points(self, X):
        """
        Calculate split points of each attribute
        """
        Z = np.zeros((X.shape[0]-1, X.shape[1]))
        for col in range(X.shape[1]):
            for i in range(X.shape[0] - 1):
                Z[i, col] = (X[i, col] + X[i+1, col]) / 2.0
            Z[:, col] = np.unique(Z[:, col])
        return Z


    def generate_recursive(self, node, X, y, Z):
        classes, cnts = np.unique(y)
        if len(classes) == 1:
            node.set_val(classes[0])
            return
        if len(classes) == 0:
            if node.parent is not None:
                node.set_val(node.parent.val)
            return
        entropy_y = self.calc_entropy(cnts, y.shape[0])
        best_attr = self.get_best_attr(X, y, entropy_y, Z)
        attr_vals = np.unique(X[:, best_attr])
        for attr_val in attr_vals.tolist():
            sub_node = TreeNode(node)
            node.add_child(sub_node)
            sub_X, sub_y = self.get_sub_set(X, y, best_attr, attr_val)
            if len(sub_y) == 0:
                sub_node.set_val(node.val)
            else:
                self.generate_recursive(sub_node, sub_X, sub_y, Z)

    def get_classes(self, y):
        return np.unique(y)

    def get_best_attr(self, X, y, entropy_y, Z):
        """
        Information Gain algorithm

        for each attribute:
            calculate info_gain
            if info_gain > info_gain_max:
                info_gain_max = info_gain
                best_attr = curr_attr
        """
        max_info_gain = float('-inf')
        for col in range(X.shape[1]):
            info_gain = self.calc_info_gain(X[:, col], y, entropy_y, Z)


    def calc_info_gain(self, data, y, entropy_y, Z):
        labels = {}
        for i in range(data.shape[0] - 1):
            split_pt = (data[i] + data[i+1]) / 2.0
            labels[split_pt] = 1 if split_pt not in labels else labels[split_pt] + 1



    @staticmethod
    def calc_entropy(sub_cnts, total_cnt):
        res = 0
        for sub_cnt in sub_cnts:
            prob = 1.0 * sub_cnt / total_cnt
            res += -prob * math.log(prob, 2)
        return res






class TreeNode:
    """
    Multi-branches tree algorithm
    """
    def __init__(self, parent=None):
        self.val = None
        self.parent = parent
        self.children = []

    def set_val(self, val):
        self.val = val

    def is_leaf(self):
        return len(self.children) == 0

    def add_child(self, node):
        self.children.append(node)
