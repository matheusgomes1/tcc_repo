#-*- encoding: UTF-8 -*-
from __future__ import print_function
import keras, json, random, argparse
from keras.layers import Dense, Flatten
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.models import Sequential
from keras.callbacks import Callback
# import matplotlib.pylab as plt
import numpy as np
from sklearn.model_selection import train_test_split

########## my modules#######################
from topologies.topology1 import Topology1
from topologies.topology2 import Topology2
from topologies.topology3 import Topology3
from topologies.topology4 import Topology4
from topologies.topology_dense import TopologyDense
from topologies.topology1_2d import Topology1_2D
from topologies.topology2_2d import Topology2_2D
from topologies.topology3_2d import Topology3_2D
from topologies.topology4_2d import Topology4_2D
from topologies.topology5_15x300 import Topology5_15x300

from metrics import Metrics

index2intent = {
    0:'GetWeather',
    1:'RateBook',
    2:'SearchCreativeWork',
    3:'SearchScreeningEvent',
    4:'PlayMusic',
    5:'AddToPlaylist',
    6:'BookRestaurant'
}
parser = argparse.ArgumentParser(description='Plataform for test in neural networks')
parser.add_argument('--seed', type=int, default= 42, help='seed for random generator')
parser.add_argument('--dataset', type=str, default= 'embedded2intent.json', help='dataset path')
parser.add_argument('--epochs', type=int, default= 5, help='number of epochs to fit the model')
####################################GLOBAL VARIABLES#############################################################
#test_set_size= 50
epochs = parser.parse_args().epochs
seed = parser.parse_args().seed
datasetPath = parser.parse_args().dataset
#################################################################################################################

#faz a predição de uma query em específico
def query_predict(query):
    #busca query no dicionario intent_dict
    for block in intent_dict:
        if query in block["query"]:
            pred_array=np.asarray(block["meanEmbedded"])
            pred_array=np.reshape(pred_array, (1, 300))
            prediction=model.predict(pred_array , batch_size=1)
            predict = prediction[0]
            predict = predict.tolist()
            indx=predict.index(max(predict))
            print('\npredict index= %d , predict intent= %s'%(indx, index2intent[indx])+'\nvec_prediction= '+str(predict))
            print("gabarito = ",block["label"], " intent = ", block["intent"])
            break

def query_predict_2d():
	n=0
	n_acc=0
	n_err=0
	#busca query no dicionario intent_dict
	for block in intent_dict:
		n=n+1
		pred_array=np.asarray(block["embeddedsMatrix"])
		pred_array=np.reshape(pred_array, (1, 8, 300, 1))
		prediction=model.predict(pred_array , batch_size=1)
		predict = prediction[0]
		predict = predict.tolist()
		indx=predict.index(max(predict))
		
		if(index2intent[indx] != block["intent"]):
			try:
				print("#falha:\n\tquery = ",block["query"],"\n\tintent = ", block["intent"], "\n\tpredicted = ", index2intent[indx])
			except UnicodeEncodeError:
				print("## UNICODE ENCODE ERROR")
			n_err = n_err+1
		else:
			n_acc = n_acc+1
		#print('\npredict index= %d , predict intent= %s'%(indx, index2intent[indx])+'\nvec_prediction= '+str(predict))
		#print("gabarito = ",block["label"], " intent = ", block["intent"])
	print("## numero acertos = ", n_acc)
	print("## numero erros = ", n_err)
	print("## taxa acc = ", float(n_acc)/n)

with open(datasetPath) as f:
    intent_dict= json.load(f)

num_classes = len(intent_dict[0]["label"])

x=[]
y=[]

#carrega os dados do json embedded2intent e coloca no formato (n, 300, 1) que é o compatível para a Conv1D
def load_dataset2conv1D():
    for block in intent_dict:
        #pegando os x's e transformando em numpyarray
        nparray=np.asarray(block["meanEmbedded"])
        nparray=np.reshape(nparray, (300, 1))
        x.append(nparray)
        #pegando os y's e transformando em numpyarray
        nparray=np.asarray(block["label"])
        y.append(nparray)

def load_dataset2conv2D(dim):
    for block in intent_dict:
        nparray=np.asarray(block["embeddedsMatrix"])
        nparray=np.reshape(nparray, (dim[0], dim[1], 1))
        x.append(nparray)
        nparray=np.asarray(block["label"])
        y.append(nparray)

#carrega os dados e coloca no formato para dense layer (n, 300)
def load_dataset2dense():
    for block in intent_dict:
        nparray=np.asarray(block["meanEmbedded"])
        x.append(nparray)
        nparray=np.asarray(block["label"])
        y.append(nparray)

#load_dataset2conv1D()
#load_dataset2dense()
load_dataset2conv2D((15, 300))

x, x_test, y, y_test = train_test_split(x, y, test_size=0.33, random_state=seed)

print("x_train + validation size= "+str(len(x)))
print("x_test size= "+str(len(x_test)))

x= np.asarray(x)
y= np.asarray(y)
x_test= np.asarray(x_test)
y_test= np.asarray(y_test)

#pega modelo do modulo topologies
#model= Topology1(num_classes).get_model()
#model= Topology2().get_model()
#model= Topology3().get_model()
#model= Topology4().get_model()
#model= TopologyDense(num_classes).get_model()
#model= Topology1_2D(num_classes).get_model()
#model= Topology2_2D(num_classes).get_model()
#model = Topology3_2D(num_classes).get_model()
#model = Topology4_2D(num_classes).get_model()
model = Topology5_15x300(num_classes).get_model()

model.summary()

#optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
#optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy', Metrics().f1])

#tensorboard_callback=keras.callbacks.TensorBoard(log_dir='logs_tensorboard/', histogram_freq=0, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)

#o parâmetro validation_split separa uma porcentagem do conjunto de treinamento para teste
model.fit(x, y, batch_size=32, epochs=epochs, validation_split=0.3, shuffle=True)
score = model.evaluate(x_test, y_test, batch_size=32, verbose=1)

print('\nteste')
print('%s = %f , %s = %f, %s = %f' %(model.metrics_names[0], score[0], model.metrics_names[1], score[1], model.metrics_names[2], score[2]))

#query_predict_2d()
