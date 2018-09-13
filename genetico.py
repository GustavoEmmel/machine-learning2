# -*- coding: utf-8 -*-

import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# gamma: 0.01 - 10
# C:     1 - 1000

"""
MLP
Numero de camadas ocultas 1 - 3 (representacao binaria 2)
Numero de neuronios 1 - 1000 (representacao binaria tamanho 10)
"""
def generate_population(n_ind):
    population = []
    for i in range(n_ind):
        n_camadas = np.random.randint(1, 4)
        n_neuronios = np.random.randint(1, 1001)

        bin_n_camadas = np.binary_repr(n_camadas, width=2)
        bin_n_neuronios = np.binary_repr(n_neuronios, width=10)

        crom = list(bin_n_camadas + bin_n_neuronios)
        population.append(crom)

    return population


"""
    individuo: cromossomo
    x_train: dados para treino
    y_train: labels do treino
    x_test: dados para teste
    y_test: labels para teste
"""
def avaliate_subject(ind, x_train, y_train, x_test, y_test):
    #gambiarra para transforma array em uma string
    info_n_camadas = int(''.join(ind[:2]), 2)
    info_n_neuronios = int(''.join(ind[2:]), 2)

    hidden_layers = (info_n_neuronios for i in range(info_n_camadas))

    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers)
    mlp.fit(x_train, y_train)

    y_predict = mlp.predict(x_test)

    return accuracy_score(y_test, y_predict)


def crossover(ind1, ind2):
    ponto_corte = np.random.randint(1, 11)

    ind3 = ind1[:ponto_corte] + ind2[ponto_corte:]
    ind4 = ind2[:ponto_corte] + ind1[ponto_corte:]

    return ind3, ind4


print generate_population(10)[0][2:]