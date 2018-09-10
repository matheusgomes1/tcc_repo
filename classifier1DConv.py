#-*- encoding: UTF-8 -*-
from __future__ import print_function
import keras, json, random
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

size_validation_batch= 50
num_classes = 10
epochs = 100

#extrai alguns elementos de x e y e cria um novo conjunto de validação e de treinamento
def gen_validation_batch(batch_val_size, x, y):
    x_val=[]
    y_val=[]
    
    #evolve random indexes
    random_list = random.sample(xrange(len(x)), batch_val_size)
    random_list.sort(reverse=True)
    for indx in random_list:
        elem_x = x.pop(indx)
        elem_y = y.pop(indx)
        x_val.append(elem_x)
        y_val.append(elem_y)
    
    return (x_val, y_val),(x, y)

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

(x_val, y_val),(x, y)= gen_validation_batch(size_validation_batch, x, y)
print("len do x= "+str(len(x)))
print("len do x_val= "+str(len(x_val)))
x= np.asarray(x)
y= np.asarray(y)

x_val= np.asarray(x_val)
y_val= np.asarray(y_val)

model = Sequential()
model.add(Conv1D (kernel_size = (11), filters = 20, input_shape=(300, 1), activation='relu'))
print(model.input_shape)
print(model.output_shape)
model.add(MaxPooling1D(pool_size = (11), strides=(1)))
print(model.output_shape)
model.add(Conv1D (kernel_size = (51), filters = 40, activation='relu'))
print(model.output_shape)
model.add(Flatten())
print(model.output_shape)
model.add(Dense(num_classes, activation='softmax',activity_regularizer=keras.regularizers.l2()))
print(model.output_shape)

#optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
#optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])

tensorboard_callback=keras.callbacks.TensorBoard(log_dir='/home/matheusgs/TCC/tcc_repo/logs/', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)

#o parâmetro validation_split separa uma porcentagem do conjunto de treinamento para teste
model.fit(x, y, batch_size=50, epochs=epochs, validation_split=0.5, callbacks=[tensorboard_callback])
score = model.evaluate(x_val, y_val, batch_size=32, verbose=1)

print('\n validation')
print('%s = %f , %s = %f'%(model.metrics_names[0], score[0], model.metrics_names[1], score[1]))

#query_predict("Find me a table for four for dinner tonight")
#query_predict("Book a table for today's lunch at Eggy's Diner for 3 people")
#query_predict("Order a taxi")
#query_predict("I need a taxi for 6 to go to Audrey and Sam's wedding")
#query_predict("Send my ETA to the guests of my apartment")
