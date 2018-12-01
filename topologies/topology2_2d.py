import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential

class Topology2_2D():
    def __init__(self, qntd_classes):
        self.model = Sequential()
        self.model.add(Conv2D(8, kernel_size=(1, 31), activation='relu', input_shape=(8, 300, 1)))
        self.model.add(MaxPooling2D(pool_size=(1, 2)))
        self.model.add(Conv2D(16, kernel_size=(2, 31), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(1, 2)))
        self.model.add(Conv2D(32, kernel_size=(3, 31), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 3)))
        self.model.add(Flatten())
        self.model.add(Dense(qntd_classes, activation='softmax',activity_regularizer=keras.regularizers.l2()))
    def get_model(self):
        return self.model