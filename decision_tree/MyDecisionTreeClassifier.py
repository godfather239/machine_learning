#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Author:     wenjieg@jumei.com
@Date:       24/07/2018
@Copyright:  Jumei Inc
"""
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
        self.generate_recursive(self.root, X, y)
        pass

    def generate_recursive(self, node, X, y):
        classes = get_classes(y)
        if len(classes) == 1:
            node.set_val(classes[0])
            return
        if len(classes) == 0:
            if node.parent is not None:
                node.set_val(node.parent.val)
            return
        best_attr = get_best_attr(X, y)
        attr_vals = np.unique(X[:, best_attr])
        for attr_val in attr_vals.tolist():
            sub_node = TreeNode(node)
            node.add_child(sub_node)
            sub_X, sub_y = get_sub_set(X, y, best_attr, attr_val)
            if len(sub_y) == 0:
                sub_node.set_val(node.val)
            else:
                self.generate_recursive(sub_node, sub_X, sub_y)






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
