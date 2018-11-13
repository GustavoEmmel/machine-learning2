# -*- coding: utf-8 -*-

"""
Everton Thomas e Gustavo Reichelt
"""

# dependencies
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils

# load data
text = (open("bv00180a.txt").read())
text=text.lower()


# word mapping
characters = sorted(list(set(text)))

n_to_char = {n:char for n, char in enumerate(characters)}
char_to_n = {char:n for n, char in enumerate(characters)}

# pre processing
X = []
Y = []
length = len(text)
seq_length = 100

for i in range(0, length-seq_length, 1):
    sequence = text[i:i + seq_length]
    label =text[i + seq_length]
    X.append([char_to_n[char] for char in sequence])
    Y.append(char_to_n[label])

X_modified = np.reshape(X, (len(X), seq_length, 1))
X_modified = X_modified / float(len(characters))
Y_modified = np_utils.to_categorical(Y)

# building model

model = Sequential()
model.add(LSTM(128, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(Y_modified.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(X_modified, Y_modified, epochs=100, batch_size=50)

# saving model to use whithout need of compile it every time

model.save_weights('text_generator.h5')
model.load_weights('text_generator.h5')

# generate literature
string_mapped = X[99]
full_string = [n_to_char[value] for value in string_mapped]
# generating characters
for i in range(400):
    x = np.reshape(string_mapped,(1,len(string_mapped), 1))
    x = x / float(len(characters))

    pred_index = np.argmax(model.predict(x, verbose=0))
    seq = [n_to_char[value] for value in string_mapped]
    full_string.append(n_to_char[pred_index])

    string_mapped.append(pred_index)
    string_mapped = string_mapped[1:len(string_mapped)]

txt=""
for char in full_string:
    txt = txt+char
print txt

text_file = open("Output.txt", "w")
text_file.write(txt)
text_file.close()

"""
Resultado:
ui do bairro, que eu conheço de vista e de chapéu. cumprimentou-me, sentou-se ao pé de mim,  
fazer a casa de capitu e a minha mãe e a minha mãe e a minha mãe e a minha mãe e a  
contertar a casa de capitu e a minha mãe e a minha mãe e a minha mãe e a minha mãe e a minha  
mãe e a minha mãe e a minha mãe e a minha mãe e a minha mãe e a minha mãe e a minha mãe  
desce a contersação de capitu e a minha mãe e a minha mãe e a minha mãe e a minha mãe  
de capitu e a minha
"""