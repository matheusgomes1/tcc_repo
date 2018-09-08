import json
import numpy as np
import gensim
from pprint import pprint

frase_intent={}

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
    n_text=0

    for domain in data["domains"]:
        print domain["description"]
        for intent in domain["intents"]:
            n_intent=n_intent+1
            print "--intent name: "+ intent["name"] + "    |    intent description:" + intent["description"]
            for query in intent["queries"]:
                n_text=n_text+1
                embedded_list=[]
                for word in query["text"].split(' '):
                    try:
                        clean_word= word.replace('?', '').lower()
                        print('\t\t'+clean_word)
                    except:
                        print("It has an issue in replace method")
                    
                    try:
                        embedded_list.append(model[clean_word])
                    except KeyError:
                        print("word %s not in vocabulary" %(clean_word))

                mean_array = get_vec_mean(embedded_list)
                

    print '\tnumero de texts: '+ str(n_text)

if __name__=="__main__":
    make_dict()