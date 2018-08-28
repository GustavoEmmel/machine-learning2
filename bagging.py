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
from collections import Counter

def balancear_dados(x, y, estrategia='oversampling'):
    if estrategia == 'oversampling':
        cnt = Counter()
        for cat in y:
            cnt[cat] += 1

        classe_majoritaria = cnt.most_common()[0][0]
        num_samples = cnt.most_common()[0][1]

        dados_bal = []
        labels_bal = []
        for classes in np.unique(y):
            if not classes == classe_majoritaria:
                dados = x[y == classes]
                label = y[y == classes]

                sampled_dados, sampled_label = resample(dados, label, n_samples=num_samples)

                dados_bal.append(sampled_dados)
                labels_bal.append(sampled_label)
            else:
                dados_bal.append(x[y == classe_majoritaria])
                labels_bal.append(y[y == classe_majoritaria])

    return np.vstack(dados_bal), np.hstack(labels_bal)


performance = {}

modelos = {'mlp', 'svm', 'random_forest', 'gradient_boosting'}

for modelo in modelos:
    performance[modelo] = {
        'acuracia': [],
        'recall': [],
        'precisao': []
    }

from sklearn.utils import resample

n_modelos = 5
kf = KFold(n_splits=10)
num_samples = 400

for train_index, test_index in kf.split(data):
    X_train, X_test = data[train_index], data[test_index]
    Y_train, Y_test = labels[train_index], labels[test_index]

    predictions = []
    ensemble_predictions = []

    for n in range(n_modelos):
        sample_X_train, sample_Y_train = resample(X_train, Y_train, n_samples=num_samples)

        #balanceamento
        sample_X_train, sample_Y_train = balancear_dados(sample_X_train, sample_Y_train)

        mlp = MLPClassifier()
        # treino
        mlp.fit(sample_X_train, sample_Y_train)

        # avaliacao
        predictions.append(mlp.predict(X_test))

    predictions = np.vstack(predictions)

    #voto majoritario
    cnt = Counter()
    for col in range(predictions.shape[1]):
        votes = predictions[:, col]
        for vote in votes:
            cnt[vote] += 1

        ensemble_predictions.append(cnt.most_common()[0][0])

    # calculo de metricas
    ens_acc = accuracy_score(Y_test, ensemble_predictions)
    ens_rec = recall_score(Y_test, ensemble_predictions)
    ens_prec = precision_score(Y_test, ensemble_predictions)



print(ensemble_predictions)
print(ens_acc, ens_rec, ens_prec)