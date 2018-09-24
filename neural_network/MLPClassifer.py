#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Author:     wenjieg@jumei.com
@Date:       23/09/2018
@Copyright:  Jumei Inc
"""
import random
import math
import numpy as np
from sklearn.datasets import load_iris
from sklearn import model_selection
from sklearn import metrics
from sklearn import neural_network
import matplotlib.pyplot as plt


class MLPClassifier:
    def __init__(self, hidden_layer_size=100, learning_rate=0.1, max_iter=200):
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        # 输入特征维度
        self.input_dim = 0
        # 输出节点数量
        self.output_dim = 0
        # 输入层权重矩阵
        self.v = None
        # 输出层权重矩阵
        self.w = None
        # 隐藏层阈值矩阵
        self.gamma = None
        # 输出层阈值矩阵
        self.theta = None
        # 最大迭代次数
        self.max_iter = max_iter
        # tolerance of optimizing
        self.tol = 0.0001
        # 隐藏层节点输出矩阵, 即b_h
        self.hlayer_output = np.zeros(self.hidden_layer_size)
        # 输出层节点输出矩阵
        self.olayer_output = None

        self.classes = None
        self.class_map = {}
        self.loss_trace = []

    def fit(self, X, y):
        """
        首先初始化参数
        迭代并使用BP更新参数
        结束
        """
        X, y = self.check_data_type(X, y)
        self.classes, cnts = np.unique(y, return_counts=True)
        y = self.extend_label(y, self.classes)
        self.initialize(X.shape[1], self.classes.shape[0])
        loss_prev = 0
        for iter in range(0, self.max_iter):
            for idx, sample in enumerate(X):
                # 随机选择一个样本,计算Yk
                # idx = random.randint(0, X.shape[0] - 1)
                # 计算输出值矩阵
                y_p = self.calc_y(X[idx])
                # 计算误差矩阵
                error = y[idx] - y_p
                loss = np.sum(np.power(error - loss_prev, 2))
                self.loss_trace.append(loss)
                # print 'current loss: %f' % error
                # print self.theta
                # if error <= self.tol:
                #     print 'tol is reached, convergence stopped!'
                #     break
                # else:
                #     print 'current loss: %f' % error
                loss_prev = error
                # 更新参数矩阵
                # 首先更新隐藏层参数
                g = y_p * (1 - y_p) * error
                # l * q 维矩阵
                delta_w = self.learning_rate * np.dot(g.reshape(g.shape[0], 1),
                                                      self.hlayer_output.reshape(1, self.hlayer_output.shape[0]))
                # 更新theta
                delta_theta = -1.0 * self.learning_rate * g
                # 更新输入层参数
                e_h = self.hlayer_output * (1 - self.hlayer_output) * np.dot(g.T, self.w)
                # q * d 维矩阵
                delta_v = self.learning_rate * np.dot(e_h.reshape(e_h.shape[0], 1), X[idx].reshape(1, X[idx].shape[0]))
                # 更新gamma, q * 1
                delta_gamma = -1.0 * self.learning_rate * e_h

                self.w += delta_w
                self.theta += delta_theta
                self.v += delta_v
                self.gamma += delta_gamma
            print self.theta
        return self

    def predict(self, X):
        data_type = type(X)
        if data_type != list and data_type != np.ndarray:
            print 'Invalid data input'
            return
        X.astype(float, copy=False)
        res = []
        for sample in X:
            y_p = self.calc_y(sample)
            res.append(self.classify(y_p))
        return res if data_type == list else np.array(res)

    def classify(self, y_p):
        c = y_p[0]
        label = self.classes[0]
        for i in range(1, y_p.shape[0]):
            if y_p[i] > c:
                c = y_p[i]
                label = self.classes[i]
        return label

    def extend_label(self, y, classes):
        """
        将单个label映射为向量,每一列代表一个输出神经元,以支持多分类
        """
        for idx, c in enumerate(classes):
            self.class_map[c] = idx
        res = []
        for i in range(0, y.shape[0]):
            row = np.zeros(classes.shape[0])
            row[self.class_map[y[i]]] = 1
            res.append(row)
        return np.array(res)

    def calc_y(self, sample):
        # 首先计算隐藏层输出值
        for i in range(0, self.hlayer_output.shape[0]):
            val = np.dot(sample, self.v[i]) - self.gamma[i]
            self.hlayer_output[i] = self.sigmoid(val)
        # 再计算输出层值
        for i in range(0, self.olayer_output.shape[0]):
            val = np.dot(self.hlayer_output, self.w[i]) - self.theta[i]
            self.olayer_output[i] = self.sigmoid(val)
        return self.olayer_output

    @staticmethod
    def sigmoid(val):
        return 1.0 / (1 + math.exp(-1.0 * val))

    @staticmethod
    def check_data_type(X, y):
        data_type = type(X)
        if data_type != list and data_type != np.ndarray:
            print 'Invalid data input'
            return
        if data_type == list:
            X = np.array(X)
            y = np.array(y)
        return X, y

    def initialize(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.v = (np.random.rand(self.hidden_layer_size, self.input_dim) * 2) - 1.0
        self.w = (np.random.rand(self.output_dim, self.hidden_layer_size) * 2) - 1.0
        # self.gamma = np.random.rand(self.hidden_layer_size) * self.input_dim
        # self.theta = np.random.rand(self.output_dim) * self.hidden_layer_size
        self.gamma = np.random.rand(self.hidden_layer_size)
        self.theta = np.random.rand(self.output_dim)
        self.olayer_output = np.zeros(output_dim)

    def visualize(self):
        # x = np.array(range(0, len(self.loss_trace)))
        res = []
        for i, val in enumerate(self.loss_trace):
            if i % 10 == 0:
                res.append(val)
        plt.plot(res)
        plt.show()


if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    train_X, test_X, train_y, test_y = model_selection.train_test_split(X, y, test_size=0.3)
    model = MLPClassifier(hidden_layer_size=100, max_iter=200)
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    print(metrics.classification_report(test_y, pred))
    model.visualize()
    # model = neural_network.MLPClassifier()
    # model.fit(train_X, train_y)
    # pred = model.predict(test_X)
    # print(metrics.classification_report(test_y, pred))
