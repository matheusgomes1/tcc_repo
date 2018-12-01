#-*- encoding: UTF-8 -*-
import json
import os, io 
import numpy as np
import gensim
import argparse
from pprint import pprint

intent2hotvect = {
    'GetWeather':[1, 0, 0, 0, 0, 0, 0],
    'RateBook':[0, 1, 0, 0, 0, 0, 0],
    'SearchCreativeWork':[0, 0, 1, 0, 0, 0, 0],
    'SearchScreeningEvent':[0, 0, 0, 1, 0, 0, 0],
    'PlayMusic':[0, 0, 0, 0, 1, 0, 0],
    'AddToPlaylist':[0, 0, 0, 0, 0, 1, 0],
    'BookRestaurant':[0, 0, 0, 0, 0, 0, 1],
}

parser = argparse.ArgumentParser(description='dictionary generator to join utterances with your respective embeddeds')
parser.add_argument('--out', type=str, default= 'embedded2intent_dict.json', help='name of out file')
parser.add_argument('--embeddeds', type=str, default='/home/matheusgs/Downloads/WordVectorsDatasets/wiki-news-300d-1M-subword.vec', help='name of embeddeds dataset')

jsonOutName = parser.parse_args().out
dataInName = parser.parse_args().embeddeds

def load_model(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
#model = gensim.models.KeyedVectors.load_word2vec_format(dataInName, binary=True) 
model = load_model(dataInName)

def get_vec_mean(embedded_list):
    embedded_sum=0
    for emb in embedded_list:
        embedded_sum=embedded_sum+emb
    
    return embedded_sum/len(embedded_list)

def get_utterance_matrix(embedded_list):
    matrix=[]
    lin = 8
    idx = 0
    while(len(matrix)<lin):
        if(idx > len(embedded_list)-1):
            matrix.append(embedded_list[-1].tolist())    
        else:
            matrix.append(embedded_list[idx].tolist())
        idx=idx+1
    
    if (idx < len(embedded_list)-1):
        matrix[lin-1]=get_vec_mean(embedded_list[(lin-1):]).tolist()
    
    print('len_matrix %d'%(len(matrix)))
    return matrix
           
def word2embedded(query):
    embedded_list=[]
    for word in query.split(' '):
        try:
            clean_word= word.replace('?', '')
            clean_word= clean_word.replace(',', '')
            clean_word= clean_word.replace('.', '')
            clean_word= clean_word.replace('\n', '')
            #print('\t\t'+clean_word)
        except:
            print("It has an issue in replace method")
        #adicionando cada embedded referente a uma palavra a uma lista de embeddeds 
        try:
            embedded_list.append(model[clean_word])
        except KeyError:
            print("word \"%s\" not in vocabulary" %(clean_word))
    

    return embedded_list

root_path= '../nlu-benchmark/2017-06-custom-intent-engines/'

list_dirs= list(os.walk(root_path))[0][1]
print(list_dirs)

dataset=[]

#iterando sobre cada diretorio 
for dir_name in list_dirs:
    print(root_path+dir_name+'/train_'+dir_name+'_full.json')

    with open(root_path+dir_name+'/train_'+dir_name+'_full.json') as f:
        file_data = json.load(f)

        #acessando cada dicionario de textos de queries em [intencao]
        for elemData in file_data[dir_name]:
            query = ""
            
            for elemText in elemData["data"]:
                query = query + elemText["text"]

            embedded_list = word2embedded(query)
            #embedded_mean = get_vec_mean(embedded_list)
            matrix_vecs = get_utterance_matrix(embedded_list)

            #dataset.append(dict(query=query, intent=dir_name, meanEmbedded=embedded_mean.tolist(), label=intent2hotvect[dir_name]))
            dataset.append(dict(query=query, intent=dir_name, embeddedsMatrix=matrix_vecs, label=intent2hotvect[dir_name]))

del model

with open(jsonOutName, 'w') as embclass_file:
    embclass_file.write(json.dumps(dataset))