#-*- encoding: UTF-8 -*-
from __future__ import print_function
import keras, json, random
from keras.layers import Dense, Flatten
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.models import Sequential
import matplotlib.pylab as plt
import numpy as np
from sklearn.model_selection import train_test_split
########## my modules#######################
from topologies.topology1 import Topology1
from topologies.topology2 import Topology2
from topologies.topology3 import Topology3
from topologies.topology_dense import TopologyDense

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

#test_set_size= 50
num_classes = 10
epochs = 50

#extrai alguns elementos de x e y e cria um novo conjunto de teste e de treinamento
def gen_test_set(batch_val_size, x, y):
    x_test=[]
    y_test=[]
    
    #evolve random indexes
    random_list = random.sample(xrange(len(x)), batch_val_size)
    random_list.sort(reverse=True)
    for indx in random_list:
        elem_x = x.pop(indx)
        elem_y = y.pop(indx)
        x_test.append(elem_x)
        y_test.append(elem_y)
    
    return (x_test, y_test),(x, y)

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
            print('\nindex= %d , label= %s'%(indx, index2label[indx])+'\nvec_prediction= '+str(predict))

with open('embedded2intent.json') as f:
    intent_dict= json.load(f)

x=[]
y=[]

#carrega os dados do json embedded2intent e coloca no formato (n, 300, 1) que é o compatível para a Conv1D
def load_dataset2conv1D():
    for tuples in intent_dict["embeddedclass"]:
        #pegando os x's e transformando em numpyarray
        nparray=np.asarray(tuples[0])
        nparray=np.reshape(nparray, (300, 1))
        x.append(nparray)
        #pegando os y's e transformando em numpyarray
        nparray=np.asarray(tuples[1])
        y.append(nparray)

#carrega os dados e coloca no formato para dense layer (n, 300)
def load_dataset2dense():
    for tuples in intent_dict["embeddedclass"]:
        nparray=np.asarray(tuples[0])
        x.append(nparray)
        nparray=np.asarray(tuples[1])
        y.append(nparray)

#load_dataset2conv1D()
load_dataset2dense()

#(x_test, y_test),(x, y)= gen_test_set(test_set_size, x, y)
x, x_test, y, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
print("x size= "+str(len(x)))
print("x_test size= "+str(len(x_test)))

x= np.asarray(x)
y= np.asarray(y)
x_test= np.asarray(x_test)
y_test= np.asarray(y_test)

#pega modelo do modulo topologies
#model= Topology1().get_model()
#model= Topology2().get_model()
#model= Topology3().get_model()
model= TopologyDense().get_model()
model.summary()

#optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
#optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])

tensorboard_callback=keras.callbacks.TensorBoard(log_dir='/home/matheusgs/TCC/tcc_repo/logs/', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)

#o parâmetro validation_split separa uma porcentagem do conjunto de treinamento para teste
model.fit(x, y, batch_size=50, epochs=epochs, validation_split=0.15, callbacks=[tensorboard_callback], shuffle=True)
score = model.evaluate(x_test, y_test, batch_size=32, verbose=1)

print('\nteste')
print('%s = %f , %s = %f'%(model.metrics_names[0], score[0], model.metrics_names[1], score[1]))

#query_predict("Find me a table for four for dinner tonight")
#query_predict("Book a table for today's lunch at Eggy's Diner for 3 people")
#query_predict("Order a taxi")
#query_predict("I need a taxi for 6 to go to Audrey and Sam's wedding")
#query_predict("Send my ETA to the guests of my apartment")
