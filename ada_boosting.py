# -*- coding: utf-8 -*-

"""
Implementar o algoritmo AdaBoost (nos mesmos moldes que fizemos com o algoritmo Bagging).
    – Podem escolher qualquer tipo de classificador (MLP, SVM, etc)

Processar os dados presente no arquivo sonar.all-data

Realizar treinamento e teste usando valida¸c˜ao cruzada com 10 folds.

Avaliar os resultados em termos de acur´acia, recall e precis˜ao.

"""
from __future__ import division
import pandas as pd
import numpy as np



""" HELPER FUNCTION: GET ERROR RATE ========================================="""
def get_error_rate(pred, Y):
    return sum(pred != Y) / float(len(Y))


sonar = pd.read_csv('sonar.all-data.csv', header=None)
print sonar.head()

labels = sonar.iloc[:,-1]
data = sonar.iloc[:,:-1]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

print (np.min(data[:,0]), np.max(data[:,0]) )

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(data, labels, train_size=0.75, test_size=0.25, stratify=labels)

iterations = 10

n_train, n_test = len(X_train), len(X_test)

w = np.ones(n_train) / n_train

pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]

print(len(X_train), len(w))

from sklearn.neural_network import MLPClassifier

for i in range(iterations):

    mlp = MLPClassifier()

    mlp.fit(X_train, Y_train)

    pred_train_i = mlp.predict(X_train)
    pred_test_i = mlp.predict(X_test)

    # Indicator function
    miss = [int(x) for x in (pred_train_i != Y_train)]
    # Equivalent with 1/-1 to update weights
    miss2 = [x if x == 1 else -1 for x in miss]
    # Error
    err_m = np.dot(w, miss) / sum(w)
    # Alpha
    alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))
    # New weights
    w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
    # Add to prediction
    pred_train = [sum(x) for x in zip(pred_train,
                                      [x * alpha_m for x in pred_train_i])]
    pred_test = [sum(x) for x in zip(pred_test,
                                     [x * alpha_m for x in pred_test_i])]

pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)


print(get_error_rate(pred_train, Y_train))
print(get_error_rate(pred_test, Y_test))


