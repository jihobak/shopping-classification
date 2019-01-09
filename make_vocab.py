import os, json
from six.moves import cPickle

def make_vocab(load_path='cate1.json', write_path='./data'):
    file_format = '{cate}_vocab.cPickle'
    cate1 = json.loads(open(load_path, 'rb').read().decode('utf-8'))
    for c in cate1.keys():
        file_name = file_format.format(cate=c)
        vocab = {value:label for label, value in enumerate(cate1[c].values())}
        cPickle.dump(vocab, open(os.path.join(write_path, file_name), 'wb'), 2)


if __name__ == '__main__':
    make_vocab(load_path='./cate1.json', write_path='./data')