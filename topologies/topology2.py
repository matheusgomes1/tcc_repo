#Epoch 100/100
#197/197 [==============================] - 2s 8ms/step - loss: 0.4940 - acc: 1.0000 - val_loss: 2.9083 - val_acc: 0.5455
#109/109 [==============================] - 0s 3ms/step
#
#teste
#loss = 3.149069 , acc = 0.522936

import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.models import Sequential

class Topology2():
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv1D (kernel_size = (11), filters = 20, input_shape=(300, 1), activation='relu'))
        self.model.add(MaxPooling1D(pool_size = (11), strides=(1)))
        self.model.add(Conv1D (kernel_size = (51), filters = 40, activation='relu'))
        self.model.add(MaxPooling1D(pool_size = (31), strides=(1)))
        self.model.add(Conv1D (kernel_size = (51), filters = 60, activation='relu'))
        self.model.add(MaxPooling1D(pool_size = (31), strides=(1)))
        self.model.add(Conv1D (kernel_size = (51), filters = 10, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(10, activation='softmax',activity_regularizer=keras.regularizers.l2()))
    def get_model(self):
        return self.model