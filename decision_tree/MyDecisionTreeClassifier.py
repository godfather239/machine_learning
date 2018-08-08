#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Author:     wenjieg@jumei.com
@Date:       24/07/2018
@Copyright:  Jumei Inc
"""
import math
import copy
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn import model_selection
from sklearn import metrics
import numpy as np


class MyDecisionTreeClassifier:
    def __init__(self, critiron='id3'):
        self.critiron = critiron
        self.root = None

    def fit(self, X, y):
        data_type = type(X)
        if data_type != list and data_type != np.ndarray:
            print 'Invalid data input'
            return
        if data_type == list:
            X = np.array(X)
            y = np.array(y)
        self.generate_decision_tree(X, y)
        return self

    def predict(self, X):
        data_type = type(X)
        if data_type != list and data_type != np.ndarray:
            print 'Invalid data input'
            return
        if self.root is None:
            return None
        X.astype(float, copy=False)
        res = []
        for i in range(X.shape[0]):
            res.append(self.predict_recursive(self.root, X[i]))
        return res if data_type == list else np.array(res)

    @staticmethod
    def predict_recursive(root, X):
        if root.is_leaf():
            return root.val
        # find sub child
        for node in root.children:
            if X[root.attr] < node.split_pt:
                return MyDecisionTreeClassifier.predict_recursive(node, X)
        return root.val

    def generate_decision_tree(self, X, y):
        self.root = TreeNode()
        X.astype(float, copy=False)
        feature_names = {}
        for i in range(X.shape[1]):
            feature_names[i] = i
        # Z = self.calc_split_points(X)
        self.generate_recursive(self.root, X, y, feature_names)

    def calc_split_points(self, X):
        """
        Calculate split points of each attribute
        """
        Z = np.zeros((X.shape[0] - 1, X.shape[1]))
        for col in range(X.shape[1]):
            for i in range(X.shape[0] - 1):
                Z[i, col] = (X[i, col] + X[i + 1, col]) / 2.0
            Z[:, col] = np.unique(Z[:, col])
        return Z

    def generate_recursive(self, node, X, y, feature_names):
        classes, cnts = np.unique(y, return_counts=True)
        if len(classes) == 1:
            node.set_val(classes[0])
            return
        if len(classes) == 0:
            if node.parent is not None:
                node.set_val(node.parent.val)
            return
        entropy_y = self.calc_entropy(cnts, y.shape[0])
        best_attr, split_pts = self.get_best_attr(X, y, entropy_y)
        node.set_attr(feature_names[best_attr])
        for pt in split_pts:
            sub_node = TreeNode(node, pt)
            node.add_child(sub_node)
            sub_X, sub_y, X, y = self.extract_sub_set(X, y, best_attr, pt)
            if len(sub_y) == 0:
                target_class = classes[0]
                target_sample_cnt = cnts[0]
                for i in range(1, len(classes)):
                    if cnts[i] > target_sample_cnt:
                        target_sample_cnt = cnts[i]
                        target_class = classes[i]
                sub_node.set_val(target_class)
            else:
                # tmp = copy.copy(feature_names)
                # tmp.remove(feature_names[best_attr])
                self.generate_recursive(sub_node, sub_X, sub_y, feature_names)
                # attr_vals = np.unique(X[:, best_attr])
                # for attr_val in attr_vals.tolist():
                #     sub_node = TreeNode(node)
                #     node.add_child(sub_node)
                #     sub_X, sub_y = self.get_sub_set(X, y, best_attr, attr_val)
                #     if len(sub_y) == 0:
                #         sub_node.set_val(node.val)
                #     else:
                #         self.generate_recursive(sub_node, sub_X, sub_y)

    def extract_sub_set(self, X, y, best_attr, split_pt):
        """
        Extract sub samples set with attribute values on best_attr small than split_pt
        """
        sub_X, sub_y, remain_X, remain_y = [], [], [], []
        for i in range(X.shape[0]):
            if X[i, best_attr] < split_pt:
                sub_X.append(X[i])
                sub_y.append(y[i])
            else:
                remain_X.append(X[i])
                remain_y.append(y[i])
        return np.array(sub_X), np.array(sub_y), np.array(remain_X), np.array(remain_y)

    def get_best_attr(self, X, y, entropy_y):
        """
        Information Gain algorithm

        for each attribute:
            calculate info_gain
            if info_gain > info_gain_max:
                info_gain_max = info_gain
                best_attr = curr_attr
        """
        max_info_gain = float('-inf')
        best_attr = -1
        best_split_pts = None
        for col in range(X.shape[1]):
            info_gain, split_pts = self.calc_info_gain(X[:, col], y, entropy_y)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_attr = col
                best_split_pts = split_pts
        return best_attr, best_split_pts

    def calc_info_gain(self, data, y, entropy_y):
        """
        return
            max information gain
            split points
        """
        labels = {}
        for i in range(data.shape[0] - 1):
            split_pt = (data[i] + data[i + 1]) / 2.0
            labels[split_pt] = 1 if split_pt not in labels else labels[split_pt] + 1
        cumu_cnt = 0
        for split_pt, cnt in labels.items():
            cumu_cnt += cnt
            labels[split_pt] = cumu_cnt
        max_info_gain = float('-inf')
        for split_pt, cumu_cnt in labels.items():
            # Split data to 2 sets, one is with value smaller than split_pt, the other otherwise
            lhs_X, rhs_X, lhs_y, rhs_y = self.partition_data(data, y, split_pt)
            # calculate information gain for this partition
            tmp, lhs_classes_cnts = np.unique(lhs_y, return_counts=True)
            tmp, rhs_classes_cnts = np.unique(rhs_y, return_counts=True)
            info_gain = entropy_y - self.calc_entropy(lhs_classes_cnts, lhs_y.shape[0]) - \
                        self.calc_entropy(rhs_classes_cnts, rhs_y.shape[0])
            max_info_gain = max(max_info_gain, info_gain)
        split_pts = labels.keys()
        split_pts.sort()
        return max_info_gain, split_pts

    def partition_data(self, data, y, split_pt):
        lhs_X, rhs_X, lhs_y, rhs_y = [], [], [], []
        for i in range(data.shape[0]):
            if data[i] < split_pt:
                lhs_X.append(data[i])
                lhs_y.append(y[i])
            else:
                rhs_X.append(data[i])
                rhs_y.append(y[i])
        return np.array(lhs_X), np.array(rhs_X), np.array(lhs_y), np.array(rhs_y)

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

    def __init__(self, parent=None, split_pt=None):
        self.val = None
        self.attr = None
        self.split_pt = split_pt
        self.parent = parent
        self.children = []

    def set_val(self, val):
        self.val = val

    def set_attr(self, attr):
        self.attr = attr

    def is_leaf(self):
        return len(self.children) == 0

    def add_child(self, node):
        self.children.append(node)


if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    train_X, test_X, train_y, test_y = model_selection.train_test_split(X, y, test_size=0.3)
    classifier = MyDecisionTreeClassifier()
    classifier.fit(train_X, train_y)
    res = classifier.predict(test_X)
    print(metrics.classification_report(test_y, res))
