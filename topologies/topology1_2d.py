import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential

class Topology1_2D():
    def __init__(self, qntd_classes):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(1, 30), activation='relu', input_shape=(8, 300, 1)))
        self.model.add(Flatten())
        self.model.add(Dense(qntd_classes, activation='softmax',activity_regularizer=keras.regularizers.l2()))
    def get_model(self):
        return self.model