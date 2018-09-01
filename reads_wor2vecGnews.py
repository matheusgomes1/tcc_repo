import gensim

# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True) 
print('distance between dog and cat is: '),
print(model.distance('dog', 'cat'))