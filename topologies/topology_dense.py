#Epoch 50/50
#186/186 [==============================] - 0s 127us/step - loss: 0.4734 - acc: 1.0000 - val_loss: 0.4171 - val_acc: 0.9394
#109/109 [==============================] - 0s 48us/step
#
#teste
#loss = 0.492383 , acc = 0.899083


import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.models import Sequential

class TopologyDense():
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(300, input_shape=(300,)))
        self.model.add(Dense(300))
        self.model.add(Dense(100))
        self.model.add(Dense(10, activation='softmax',activity_regularizer=keras.regularizers.l2()))
    def get_model(self):
        return self.model