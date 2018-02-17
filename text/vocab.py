import gensim
import nltk
import pickle
from nltk.tokenize import RegexpTokenizer
import coloredlogs, logging
import torch
import numpy as np

# Create a logger object.
logger = logging.getLogger(__name__)
PROJECT_DIR = './'
use_cuda = torch.cuda.is_available()

class Vocab:
    
    def __init__(self):
        self.id2word = []
        self.word2id = {}

        self.id2word.append('<PAD>')
        self.id2word.append('<SOS>')
        self.id2word.append('<EOS>')
        self.id2word.append('<UNK>')
        
        self.word2id['<PAD>'] = 0
        self.word2id['<SOS>'] = 1
        self.word2id['<EOS>'] = 2
        self.word2id['<UNK>'] = 3

        self.id = 4
        self.examples = 0

    def add(self,word):
        if word not in self.word2id:
            self.word2id[word] = self.id
            self.id2word.append(word)
            self.id +=1
    
    def word(self,id):
        return self.id2word[id]
    
    def idx(self,word):
        if word in self.word2id:
            return self.word2id[word]
        else:
            return self.word2id['<UNK>']
    
    def length(self):
        return self.id

class WordModel:
    
    def __init__(self):
        self.word_model = None
        self.word_err_file = None
        self.word_errors = {}
        self.vocab = Vocab()

    def load_embeddings(self):
        logger.warning("Loading word model")
        self.word_model = gensim.models.KeyedVectors.load_word2vec_format('lex.vectors', binary=False)

    def write_to_log(self,sentence,filename):
        logger.warning(str(sentence) + ' writing to log')
        self.word_err_file = open(filename, "a+")
        self.word_err_file.write(sentence + '\n')

    def close_log(self):
        self.word_err_file.close()
        self.word_err_file = None

    def collect_errors(self,word,filename):
        if word not in self.word_errors:
            self.word_errors[word] = 1
            self.write_to_log(word,filename)

    def build_vocab(self,captions):
        logger.warning("Building vocabulary")

        for i in range(len(captions)):
            caption = captions[i]

            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(caption)
            for word in tokens:
                word = word.lower()
                if word not in self.word_model:
                    logger.error(str(word) + ' does not exist')
                    self.collect_errors(word,filename=PROJECT_DIR + '/errors/word_error.log')
                else:
                    self.vocab.add(word)
            self.vocab.examples += 1
    
    def tensor(self,array):
        # if use_cuda:
        #     return torch.cuda.LongTensor(array)
        # else:
        return torch.LongTensor(array)

    def parse(self,caption):
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(caption)
        ids = []
        for word in tokens:
            word = word.lower()
            ids.append(self.vocab.idx(word))
        ids.append(self.vocab.idx('<EOS>'))
        ids = self.tensor(ids)
        return ids

    def captionloader(self,captions):
        num_examples = []
        for i in range(len(captions)):
            batch_captions = []
            for j in range(len(captions[i])):
                ids = self.parse(captions[i][j])
                batch_captions.append(ids)
            num_examples.append(batch_captions)
        return num_examples
            
    def save(self,filename):
        pickle.dump(self.vocab,open(filename,"wb"))
    
    def load(self,filename):
        self.vocab = pickle.load(open(filename,"rb"))