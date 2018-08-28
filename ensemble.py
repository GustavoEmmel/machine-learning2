import numpy as np

data = np.genfromtxt('diabetes.csv', delimiter=',')

labels = data[:, -1]
data = data[:, :-1]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

#print (np.min(data[:,0]), np.max(data[:,0]) )

# modelos
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#cross validation
from sklearn.model_selection import KFold

#metricas
from sklearn.metrics import accuracy_score, recall_score, precision_score

performance = {}

modelos = {'mlp', 'svm', 'random_forest', 'gradient_boosting'}

for modelo in modelos:
    performance[modelo] = {
        'acuracia': [],
        'recall': [],
        'precisao': []
    }

kf = KFold(n_splits=10)

for train_index, test_index in kf.split(data):
    X_train, X_test = data[train_index], data[test_index]
    Y_train, Y_test = labels[train_index], labels[test_index]

    mlp = MLPClassifier()
    svm = SVC(C=10, gamma=0.1)
    rf = RandomForestClassifier()
    bt = GradientBoostingClassifier(n_estimators=10)

    #treino
    mlp.fit(X_train, Y_train)

    #avaliacao
    mlp_predict = mlp.predict((X_test))

    mlp_acc = accuracy_score(Y_test, mlp_predict)
    mlp_recall = recall_score((Y_test, mlp_predict))
    mlp_prec = precision_score(Y_test, mlp_predict)

    performance['mlp']['acuracia'].append(mlp_acc)