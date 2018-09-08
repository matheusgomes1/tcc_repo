from __future__ import print_function
import keras
from keras.layers import Dense, Flatten, Dropout, Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.models import Sequential
import matplotlib.pylab as plt
import numpy as np

batch_size = 5
num_classes = 10
epochs = 2

# Generate dummy data
x_train = np.random.random((100, 300, 1))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
x_test = np.random.random((100, 300, 1))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

y_train= np.reshape(y_train, (100, 10))
y_test= np.reshape(y_test, (100, 10))
print(y_train) 
print(x_train)

model = Sequential()
model.add(Conv1D (kernel_size = (11), filters = 20, input_shape=(300, 1), activation='relu'))
print(model.input_shape)
print(model.output_shape)
model.add(MaxPooling1D(pool_size = (10), strides=(1)))
print(model.output_shape)
model.add(Flatten())
print(model.output_shape)
model.add(Dense(10, activation='softmax',activity_regularizer=keras.regularizers.l2()))
print(model.output_shape)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=50, epochs=1)
score = model.evaluate(x_test, y_test, batch_size=16)