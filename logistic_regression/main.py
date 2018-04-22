#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn import model_selection as model_sel
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from DataParser import *


def sigmoid(score):
    return 1.0 / (1 + np.exp(-1.0 * score))


def log_likelihood(features, labels, weights):
    scores = np.dot(features, weights)
    ll = np.sum(labels * scores - np.log(1.0 + np.exp(scores)))
    return ll


def logistic_regression(features, labels, steps=10000, lr=1.0):
    """
    Train logistic regression model and return weights vector
    Stop iteration when abs(ll(curr) - ll(prev)) <= 0.00001
    """
    # extend feature vector
    intercpt = np.ones((features.shape[0], 1))
    features = np.hstack((intercpt, features))

    # init weights vector
    weights = np.zeros(features.shape[1])

    ll_prev = 0
    # iteration
    for step in range(steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)
        output_errors = labels - predictions
        gradient = np.dot(features.T, output_errors)
        weights += gradient * lr
        ll = log_likelihood(features, labels, weights)
        # if step % 100 == 0:
        #     print ll
        if abs(ll - ll_prev) <= 0.00001:
            break
        ll_prev = ll

    return weights


def predict(features, weights):
    intercpt = np.ones((features.shape[0], 1))
    features = np.hstack((intercpt, features))
    scores = np.dot(features, weights)
    predictions = sigmoid(scores)
    res = []
    for predict in predictions:
        res.append(1 if predict > 0.5 else 0)
    return res


def read_data(file_path):
    parser = DataParser(file_path, ',', True)
    features = []
    labels = []
    for row in parser.readrows():
        feature = list()
        feature.append(float(row['density']))
        feature.append(float(row['sugar_content']))
        features.append(feature)
        labels.append(int(row['label']))
    return features,labels


def check_my_classifier():
    features,labels = read_data('./watermelon_data3.0a.txt')
    weights = logistic_regression(np.array(features), np.array(labels), 10000, 0.05)
    print "weights: %s" % str(weights.tolist())

    # predict
    res = predict(np.array(features), weights)
    print "predict: %s" % str(res)

    features = np.array(features)
    labels = np.array(labels)
    X_train,X_test,y_train,y_test = model_sel.train_test_split(features, labels, test_size=0.5)
    print "y_train: %s" % str(y_train.tolist())
    weights = logistic_regression(X_train, y_train)
    y_predict = np.array(predict(X_test, weights))
    # summarize the accuracy of fitting
    print(metrics.confusion_matrix(y_test, y_predict))
    print(metrics.classification_report(y_test, y_predict))


def check_sklearn_classifier():
    features,labels = read_data('./watermelon_data3.0a.txt')
    features = np.array(features)
    labels = np.array(labels)
    X_train,X_test,y_train,y_test = model_sel.train_test_split(features, labels, test_size=0.5)
    classifier = LogisticRegression(max_iter=10000)
    classifier.fit(X_train, y_train)
    print "weights: %s" % str(classifier.coef_)

    y_pred = classifier.predict(X_test)
    print "y_train: %s" % str(y_train.tolist())
    # summarize the accuracy of fitting
    print(metrics.confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))


if __name__ == "__main__":
    check_sklearn_classifier()
