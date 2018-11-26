import io, json

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

 
with open('file_data.json', 'w') as f:
    f.write(json.dumps(load_vectors('wiki-news-300d-1M-subword.vec')))
