import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.models import Sequential

class Topology3():
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv1D (kernel_size = (11), filters = 20, input_shape=(300, 1), activation='relu'))
        self.model.add(MaxPooling1D(pool_size = (11), strides=(1)))
        self.model.add(Conv1D (kernel_size = (81), filters = 60, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(10, activation='softmax',activity_regularizer=keras.regularizers.l2()))
    def get_model(self):
        return self.model