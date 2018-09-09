import json
import numpy as np
import gensim
from pprint import pprint

intent2hotvect={
    "ShareCurrentLocation":[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "ComparePlaces":[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "GetPlaceDetails":[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "SearchPlace":[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "BookRestaurant":[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "RequestRide":[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "GetDirections":[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "ShareETA":[0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "GetTrafficInformation":[0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "GetWeather":[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
}

embedded_class={"embeddedclass":[]}

# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin.gz', binary=True) 

with open('/home/matheusgs/Downloads/nlu-benchmark-master/2016-12-built-in-intents/benchmark_data.json') as f:
    data = json.load(f)

def get_vec_mean(embedded_list):
    pass
    embedded_sum=0
    for emb in embedded_list:
        embedded_sum=embedded_sum+emb
    
    return embedded_sum/len(embedded_list)

def make_dict():
    n_intent=0
    n_query=0

    for domain in data["domains"]:
        #print domain["description"]
        for intent in domain["intents"]:
            n_intent=n_intent+1
            print "--intent name: "+ intent["name"] + "    |    intent description:" + intent["description"]
            for query in intent["queries"]:
                n_query=n_query+1
                embedded_list=[]
                for word in query["text"].split(' '):
                    try:
                        clean_word= word.replace('?', '')
                        clean_word= clean_word.replace(',', '')
                        clean_word= clean_word.replace('.', '')
                        #print('\t\t'+clean_word)
                    except:
                        print("It has an issue in replace method")
                    #adicionando cada embedded referente a uma palavra a uma lista de embeddeds 
                    try:
                        embedded_list.append(model[clean_word])
                    except KeyError:
                        print("word %s not in vocabulary" %(clean_word))

                mean_array = get_vec_mean(embedded_list)
                #adiciona uma tupla com o emebedded e sua label (em hot vector), a frase e a label (string)
                embedded_class["embeddedclass"].append((mean_array.tolist(), intent2hotvect[intent["name"]], query["text"], intent["name"]))
            
            print("intent: %s has %d queries"%(intent["name"], n_query))
            n_query=0

    with open('embedded2intent.json', 'w') as embclass_file:
        json.dump(embedded_class, embclass_file)

if __name__=="__main__":
    make_dict()