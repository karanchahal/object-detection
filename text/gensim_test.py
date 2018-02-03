import gensim

import gensim
model = gensim.models.Word2Vec.load_word2vec_format('path-to-vectors.txt', binary=False)
# if you vector file is in binary format, change to binary=True
sentence = ["London", "is", "the", "capital", "of", "Great", "Britain"]
vectors = [model[w] for w in sentence]