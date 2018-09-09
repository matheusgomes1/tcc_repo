#-*- encoding: UTF-8 -*-
from __future__ import print_function
import keras, json
from keras.layers import Dense, Flatten
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.models import Sequential
import matplotlib.pylab as plt
import numpy as np

index2label={
    0:"ShareCurrentLocation",
    1:"ComparePlaces",
    2:"GetPlaceDetails",
    3:"SearchPlace",
    4:"BookRestaurant",
    5:"RequestRide",
    6:"GetDirections",
    7:"ShareETA",
    8:"GetTrafficInformation",
    9:"GetWeather"
}

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

model.fit(x, y, batch_size=100, epochs=10, validation_split=0.5)
#score = model.evaluate(x_test, y_test, batch_size=16)

#faz a predição de uma query em específico
def query_predict(query):
    #busca query no dicionario intent_dict
    for block in intent_dict["embeddedclass"]:
        if query in block:
            pred_array=np.asarray(block[0])
            pred_array=np.reshape(pred_array, (1, 300, 1))
            
            prediction=model.predict(pred_array , batch_size=1)
            predict = prediction[0]
            predict = predict.tolist()
            indx=predict.index(max(predict))
            print('index= %d , label= %s , '%(indx, index2label[indx])+'vec_prediction= '+str(predict))

query_predict("Find me a table for four for dinner tonight")
query_predict("Book a table for today's lunch at Eggy's Diner for 3 people")
query_predict("Order a taxi")
query_predict("I need a taxi for 6 to go to Audrey and Sam's wedding")
query_predict("Send my ETA to the guests of my apartment")