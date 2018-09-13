# -*- coding: utf-8 -*-

import numpy as np
import copy
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def vizinhanca(gamma, C):
    gamma_copy = copy.copy(gamma)
    C_copy = copy.copy(C)

    gamma = gamma + (2 * np.random.rand() - 1) * gamma
    C = C + (2 * np.random.rand() - 1) * C

    while gamma <= 0 or C <= 0:
        gamma = gamma_copy
        C = C_copy
        gamma = gamma + (2 * np.random.rand() - 1) * gamma
        C = C + (2 * np.random.rand() - 1) * C

    return gamma, C


def prob_aceitacao(valor_s, valor_s1, T):
    return np.exp(-(valor_s1 - valor_s) / T)


def f(y_true, y_pred):
    return np.linalg.norm(y_true - y_pred)


def simulated_annealing(gamma_inicial, C_inicial, T, alpha, SAmax, max_iter):
    breast_cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(breast_cancer['data'],
                                                        breast_cancer['target'],
                                                        test_size=0.33)
    iterT = 0
    i = 0
    gamma = gamma_inicial
    C = C_inicial

    melhor_gamma = gamma
    melhor_C = C
    melhor_f = 1e12

    for i in range(max_iter):
        pbar = tqdm(total=SAmax)
        while iterT < SAmax:
            iterT += 1
            pbar.update(1)
            svm_s = SVC(gamma=gamma, C=C)

            gamma_s1, C_s1 = vizinhanca(gamma, C)
            svm_s1 = SVC(gamma=gamma_s1, C=C_s1)

            svm_s = svm_s.fit(X_train, y_train)
            svm_s1 = svm_s1.fit(X_train, y_train)

            y_pred_s = svm_s.predict(X_train)
            y_pred_s1 = svm_s1.predict(X_train)

            f_s = f(y_train, y_pred_s)
            f_s1 = f(y_train, y_pred_s1)

            if f_s1 <= f_s:
                gamma = gamma_s1
                C = C_s1
                if f_s1 < melhor_f:
                    melhor_f = f_s1
                    melhor_gamma = gamma
                    melhor_C = C
            else:
                v = np.random.rand()
                if v <= prob_aceitacao(f_s, f_s1, T):
                    gamma = gamma_s1
                    C = C_s1
        iterT = 0
        T = alpha * T

    print('SA terminado!\n')
    print('Melhor gamma: {}'.format(melhor_gamma))
    print('Melhor C: {}'.format(melhor_C))

    print('Treinando modelo final....')
    final_svm = SVC(gamma=melhor_gamma, C=melhor_C)
    final_svm = final_svm.fit(X_train, y_train)
    print('Avaliando acurácia no conjunto de teste...')
    accuracy = final_svm.score(X_test, y_test)
    print('Acurácia: {}'.format(accuracy))

simulated_annealing(0.1, 10, 100, 0.5, 100, 10)