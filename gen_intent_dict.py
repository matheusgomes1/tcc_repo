import json
import numpy as np
import gensim
from pprint import pprint

frase_intent={}

# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True) 

with open('/home/matheusgs/Downloads/nlu-benchmark-master/2016-12-built-in-intents/benchmark_data.json') as f:
    data = json.load(f)

n_intent=0
n_text=0

for domain in data["domains"]:
    print domain["description"]
    for intent in domain["intents"]:
        n_intent=n_intent+1
        print "--intent name: "+ intent["name"] + "    |    intent description:" + intent["description"]
        for query in intent["queries"]:
            n_text=n_text+1
            print "\t---text: "+str(query["text"].split(' '))

        print '\tnumero de texts: '+ str(n_text)
        n_text=0
