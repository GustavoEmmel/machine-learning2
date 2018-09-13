# -*- coding: utf-8 -*-
"""
Gustavo Reichelt Emmel

Metaheuristicas baseadas em solução unica
IMPLEMENTAÇAO:
- Simuleted Annealing + SVM
    - Sigma - Gamma (Kernel RBG) >= 0
    - C >= 0
"""

# this import cast all numbers to float
from __future__ import division
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from copy import copy
from tqdm import tqdm

# receives a solution ( s = [gamma, C] )
def generate_neighboors(s, r=0.15):
    ns = copy(s)
    ns[0] += (2 * np.random.rand() - 1) * r * ns[0]
    ns[1] += (2 * np.random.rand() - 1) * r * ns[1]
    return ns


# avaliation funcion
def f(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


# receives two values and time
def acceptance_prob(v1, v2, t):
    return np.exp(-(v1 -v2)/t)


# dataset
diabetes = np.genfromtxt('diabetes.csv', delimiter=',')
data = diabetes[:, :-1]
labels = diabetes[:, -1]

# separa treino <-> teste
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.3)
# separa porção do treino para validacao
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1)

# Simulated Annealing implementation
solution = [1, 10]
temp = 100
SAmax = 100
alpha = 0.9

iterT = 0

value = 0
old_value = - 1

accuracy = []

while value - old_value > 1e-3:
    pbar = tqdm(total=SAmax)
    while iterT <= SAmax:
        iterT += 1
        pbar.update(1)
        
        # generate a randon solution basead on the neightborhood from the current solution
        new_solution = generate_neighboors(solution)

        # treinamento
        actual_svm = SVC(C=solution[1], gamma=solution[0])
        new_svm = SVC(C=new_solution[1], gamma=new_solution[0])

        actual_svm.fit(X_train, Y_train)
        new_svm.fit(X_train, Y_train)

        y_actual_pred = actual_svm.predict(X_val)
        y_new_predict = new_svm.predict(X_val)

        # avalicao teste de acuracia
        f_actual = f(Y_val, y_actual_pred)
        f_new = f(Y_val, y_new_predict)

        if f_new >= f_actual:
            solution = copy(new_solution)

            old_value = value
            value = f_new
        else:
            # calcula probalidade de aceitacao
            v = np.random.rand()
            prob = acceptance_prob(f_new, f_actual, temp)

            if v <= prob:
                old_value = value
                value = f_new
    iterT = 0
    temp *= alpha
    accuracy.append(value)



print generate_neighboors([1, 10])
print 1 / 2
print accuracy
print "done"