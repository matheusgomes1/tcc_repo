from __future__ import print_function
import keras, json
from keras.layers import Dense, Flatten
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.models import Sequential
import matplotlib.pylab as plt
import numpy as np

batch_size = 5
num_classes = 10
epochs = 2

with open('embedded2intent.json') as f:
    intent_dict= json.load(f)

x=[]
y=[]
for tuples in intent_dict["embeddedclass"]:
    #pegando os x's e transformando em numpyarray
    nparray=np.asarray(tuples[0])
    nparray=np.reshape(nparray, (300, 1))
    x.append(nparray)
    #pegando os y's e transformando em numpyarray
    nparray=np.asarray(tuples[1])
    y.append(nparray)

x= np.asarray(x)
y= np.asarray(y)

print(x) 
print(y)

model = Sequential()
model.add(Conv1D (kernel_size = (11), filters = 20, input_shape=(300, 1), activation='relu'))
print(model.input_shape)
print(model.output_shape)
model.add(MaxPooling1D(pool_size = (10), strides=(1)))
print(model.output_shape)
model.add(Flatten())
print(model.output_shape)
model.add(Dense(num_classes, activation='softmax',activity_regularizer=keras.regularizers.l2()))
print(model.output_shape)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x, y, batch_size=100, epochs=50)
#score = model.evaluate(x_test, y_test, batch_size=16)